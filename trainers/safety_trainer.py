import os

import torch
import torch.nn as nn
import wandb
from dataset import get_dataloader
from peft import LoraConfig, get_peft_model
from tqdm import tqdm, trange
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          get_linear_schedule_with_warmup)
from utils import get_decay_parameter_names


class SafetyTrainer(object):
    def __init__(self, args) -> None:
        self.args = args

        wandb.init(reinit=True, config=args.as_dict(),
                   project=args.wandb_project, name=args.exp_name)

        self.device = torch.cuda.current_device()

        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device)
        
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
        self.dataloader = get_dataloader("safety-tuning", self.tokenizer, 
                                        prompt_file=args.prompt_file, 
                                        batch_size=args.batch_size)
        
    
    def get_position_ids(self, attention_mask):
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        return position_ids

    def save(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
    def train(self):
        t = tqdm(range(1, self.args.train_steps+1), desc="training", dynamic_ncols=True, leave=False)
        self.model.train()
        global_step = 1
        for epoch in trange(self.args.epochs):
            for batch in tqdm(self.dataloader, dynamic_ncols=True):
                batch_loss = []
                
                chunks = {k:torch.chunk(v, self.args.grad_acc_steps, dim=0) for k,v in batch.items()}
                num_chunks = len(chunks["input_ids"])
                self.model.zero_grad()
                self.model.train()
                for i in tqdm(range(num_chunks), desc="gradient step", dynamic_ncols=True, leave=False):
                    mini_batch = {k:v[i].to(self.device) for k,v in chunks.items()}    
                    
                    # need expllicit position_ids because of left-padding
                    loss = self.model(**mini_batch).loss
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
                    f"Epoch: {epoch}, Step {global_step}: {sum(batch_loss): .4f}")

                global_step += 1
        
        output_dir = os.path.join(self.args.save_dir, self.args.exp_name, "latest")
        self.save(output_dir)
        wandb.finish()


