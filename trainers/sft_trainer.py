import os

import torch
import torch.nn as nn
import wandb
from dataset import get_dataloader
from tqdm import tqdm
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          get_linear_schedule_with_warmup)
from utils import InfIterator, get_decay_parameter_names
import numpy as np
from peft import LoraConfig, get_peft_model

class SFTTrainer(object):
    def __init__(self, args) -> None:
        self.args = args

        wandb.init(reinit=True, config=args.as_dict(),
                   project=args.wandb_project, name=args.exp_name)

        self.device = torch.cuda.current_device()
        config = AutoConfig.from_pretrained(args.model_name)
        config.use_cache = True

        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16 if self.args.lora else None,
            config=config,
            device_map=self.device)
        
        if self.args.lora:
            lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )

            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()


        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, padding_side="left")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


        decay_parameters = get_decay_parameter_names(self.model)
        
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters() if (n in decay_parameters and p.requires_grad)
                ],

                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, args.num_warmup_steps, args.train_steps)

        self.tr_dataloader, self.val_dataloader = get_dataloader("sft", self.tokenizer, 
                                                                prompt_file=args.prompt_file, 
                                                                sft_file=args.few_shot_file, 
                                                                batch_size=args.batch_size)
        
        self.train_iter = InfIterator(self.tr_dataloader)
    
    def get_position_ids(self, attention_mask):
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        return position_ids

    def validation(self):
        self.model.eval()
        all_loss =[]
        for batch in tqdm(self.val_dataloader, leave=False, dynamic_ncols=True, desc="eval"):
            
            chunks = {k:torch.chunk(v, self.args.grad_acc_steps, dim=0) for k,v in batch.items()}
            num_chunks = len(chunks["input_ids"])
            for i in tqdm(range(num_chunks),  dynamic_ncols=True, leave=False):
                mini_batch = {k:v[i].to(self.device) for k,v in chunks.items()}
                
                # need expllicit position_ids because of left-padding
                attention_mask = mini_batch["attention_mask"]
                position_ids = self.get_position_ids(attention_mask)
                
                with torch.no_grad():
                    loss = self.model(**mini_batch, position_ids=position_ids).loss

                    all_loss.append(loss.item())
        
        return np.mean(all_loss)
    
    def save(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
    def train(self):
        t = tqdm(range(1, self.args.train_steps+1), desc="training", dynamic_ncols=True, leave=False)
        for global_step in t:
            batch_loss = []
            batch = next(self.train_iter)
            chunks = {k:torch.chunk(v, self.args.grad_acc_steps, dim=0) for k,v in batch.items()}
            num_chunks = len(chunks["input_ids"])
            
            self.model.train()
            self.model.zero_grad()
            for i in tqdm(range(num_chunks), desc="gradient step", dynamic_ncols=True, leave=False):
                mini_batch = {k:v[i].to(self.device) for k,v in chunks.items()}    
                
                # need expllicit position_ids because of left-padding
                attention_mask = mini_batch["attention_mask"]
                position_ids = self.get_position_ids(attention_mask)
                
                loss = self.model(**mini_batch, position_ids=position_ids).loss
                loss = loss / self.args.grad_acc_steps
                
                loss.backward()
                batch_loss.append(loss.item())
            
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.args.max_norm)

            self.optimizer.step()
            self.scheduler.step()

            # logging
            wandb.log({"ce-loss/train": sum(batch_loss)}, step=global_step)

            t.set_description(
                f"Step {global_step}: {sum(batch_loss): .4f}")

            if global_step % 10 ==0:
                val_loss = self.validation()
                wandb.log({"ce-loss/validation": val_loss}, step=global_step)
            
            if global_step % self.args.eval_period == 0:
                output_dir = os.path.join(
                    self.args.save_dir, 
                    f"{self.args.exp_name}/{global_step}")
                self.save(output_dir)
        
        output_dir = os.path.join(self.args.save_dir, self.args.exp_name, "latest")
        if self.args.lora:
            self.model = self.model.merge_and_unload()
        self.save(output_dir)
        wandb.finish()


