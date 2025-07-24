import json
import logging
import math
import os
import random
from collections import defaultdict
from typing import Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from csv_logger import CsvLogger
from dataset import get_dataloader
from peft import LoraConfig, PeftModel, get_peft_model
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          get_linear_schedule_with_warmup)
from utils import (CosineRelayBuffer, InfIterator, LlamaToxicClassifier,
                   ReplayBuffer, RobertaClassifier, base_to_lora,
                   batch_cosine_similarity_kernel, formatted_dict,
                   lora_to_base)
from vllm import LLM, SamplingParams

from w00.utils import PresidioClassifier

def avg_pooling(last_hidden, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(last_hidden.size()).float()
    denom = torch.clamp(input_mask_expanded.sum(1), min=1)
    avg_pool = torch.sum(last_hidden * input_mask_expanded, 1) / denom
    return avg_pool


def generate_and_return_z_logprob(model, prompt_ids, prompt_attention_mask,  eos_token_id, temperature, max_len=30):
    active_seqs = torch.ones(prompt_ids.size(0)).bool().to(prompt_ids.device)
    actions = prompt_ids.clone()
    state = prompt_ids.clone()
    sum_logpf = torch.zeros(prompt_ids.size(0)).float().to(prompt_ids.device)
    attention_mask = prompt_attention_mask.clone()

    # past_key_values = DynamicCache()
    if "Llama" in model.config.architectures[0]:
        cache_position = torch.arange(
            state[:, :-1].size(1), dtype=torch.int64, device=prompt_ids.device)
    else:
        cache_position = None

    inputs = {
        "input_ids": state[:, :-1],
        "attention_mask": attention_mask[:, :-1],
        "output_hidden_states": True,
        "use_cache": True,
        # "past_key_values": past_key_values,
    }
    if cache_position is not None:
        inputs["cache_position"] = cache_position

    outputs = model(**inputs)
    hidden_states = outputs["hidden_states"][-1]
    past_key_values = outputs["past_key_values"]
    for i in range(max_len):
        # update cache position
        if cache_position is not None:
            cache_position = cache_position[-1:] + 1
        if i == 0:
            # If past_key_values is used, attention_mask needs to contain the masking strategy that was used for past_key_values.
            # In other words, the attention_mask always has to have the length: len(past_key_values) + len(input_ids)
            # https://huggingface.co/docs/transformers/en/model_doc/gpt2#transformers.GPT2Model.forward.attention_mask
            inputs = {"input_ids": state[:, -1:],
                      "attention_mask": attention_mask,
                      "past_key_values": past_key_values,
                      "output_hidden_states": True,
                      "use_cache": True,
                      }
            if cache_position is not None:
                inputs["cache_position"] = cache_position
            output = model(**inputs)

            last_hidden = output.hidden_states[-1]
            last_hidden = torch.cat([hidden_states, last_hidden], dim=1)
            avg_pool = avg_pooling(last_hidden, prompt_attention_mask)

            log_z = model.proj_z(avg_pool).squeeze(-1)
        else:
            inputs = {
                "input_ids": state[:, -1:],
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
            }
            if cache_position is not None:
                inputs["cache_position"] = cache_position
            output = model(**inputs)
        past_key_values = output["past_key_values"]

        logits = output["logits"][:, -1, :]
        if i == 0:
            logits[..., eos_token_id] = -torch.inf
        with torch.no_grad():
            modified_logits = logits.clone()
            prob = F.softmax(modified_logits / temperature, dim=-1)
            token_ids = torch.multinomial(prob, num_samples=1)

        logprob = F.log_softmax(logits, dim=-1)
        logprob = torch.gather(logprob, -1, token_ids).squeeze(-1)
        logprob = torch.where(active_seqs, logprob, torch.zeros_like(logprob))
        sum_logpf = sum_logpf + logprob

        token_ids = torch.where(
            active_seqs.unsqueeze(-1), token_ids, torch.ones_like(token_ids) * eos_token_id)

        # update action, state, mask
        masks = torch.where(active_seqs.unsqueeze(-1),
                            torch.ones_like(token_ids), torch.zeros_like(token_ids))
        attention_mask = torch.cat([attention_mask.long(), masks], dim=1)
        actions = torch.cat([actions, token_ids], dim=1)
        state = torch.cat([actions, token_ids], dim=1)

        # check if all sequences have generated eos
        active_seqs = active_seqs * (token_ids != eos_token_id).squeeze(-1)
        if torch.all(~active_seqs):
            break
    # add EOS token to penalize incomplete sentences.
    eos_tokens = torch.ones(actions.size(
        0), dtype=torch.long, device=actions.device) * eos_token_id
    actions = torch.cat([actions, eos_tokens.unsqueeze(1)], dim=1)
    results = {"actions": actions, "sum_logpf": sum_logpf, "log_z": log_z}

    return results


class GFNTrainer(object):
    def __init__(self, args) -> None:
        self.args = args

        wandb.init(reinit=True, config=args.as_dict(),
                   project=args.wandb_project, name=args.exp_name)

        self.device = torch.cuda.current_device()
        config = AutoConfig.from_pretrained(args.model_name)
        config.use_cache = True

        self.model = AutoModelForCausalLM.from_pretrained(
            args.sft_ckpt,
            config=config,
            device_map=self.device)

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            # target_modules=["c_attn"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        model_config = self.model.config

        if "gpt2" in args.model_name:
            n_dim = model_config.n_embd
        else:
            n_dim = model_config.hidden_size

        self.model.proj_z = nn.Linear(n_dim, 1).to(self.device)

        output_dir = os.path.join(self.args.save_dir, self.args.exp_name)

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.sft_ckpt, padding_side="left")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.victim_model = LLM(
            args.victim_model, dtype=args.dtype, gpu_memory_utilization=args.gpu_memory_utilization)

        self.victim_model_tokenizer = AutoTokenizer.from_pretrained(
            args.victim_model, padding_side="left")

        self.victim_model_tokenizer.pad_token_id = self.victim_model_tokenizer.eos_token_id

        if "Llama-3" in args.victim_model:
            stop_token_ids = [self.tokenizer.eos_token_id,
                              self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        else:
            stop_token_ids = [self.tokenizer.eos_token_id]

        self.sampling_params = SamplingParams(
            n=args.num_r_samples,
            top_p=args.victim_top_p,
            temperature=args.victim_temp,
            stop_token_ids=stop_token_ids,
            max_tokens=args.victim_max_len)

        self.toxicity_fn = PresidioClassifier(device=self.device)

        self.sentence_encoder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device=self.device)

        self.dataloader = get_dataloader(
            "gfn", self.tokenizer, prompt_file=args.prompt_file,
            batch_size=args.batch_size, shuffle=True)

        self.train_iter = InfIterator(self.dataloader)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)
        t_total = args.train_steps * args.grad_acc_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, args.num_warmup_steps, t_total)

        # initialize buffer
        if args.metric == "edit":
            print("edit distance for buffer")
            self.rbuffer = ReplayBuffer(
                self.tokenizer.eos_token_id,
                self.args.buffer_size,
                prioritization=self.args.prioritization,
                compare=self.args.compare)
        elif args.metric == "cosine":
            print("cosine similarity for buffer")
            self.rbuffer = CosineRelayBuffer(
                self.tokenizer.eos_token_id,
                self.args.buffer_size,
                prioritization=self.args.prioritization,
                compare=self.args.compare)

        self.start = self.load(output_dir, self.model,
                               self.optimizer, self.scheduler, self.rbuffer)

        delimiter = ","
        self.csvlogger = CsvLogger(filename=f"logs/{args.exp_name}.csv",
                                   delimiter=delimiter,
                                   level=logging.INFO,
                                   add_level_nums=None,
                                   fmt=f'%(asctime)s{delimiter}%(message)s',
                                   datefmt='%Y/%m/%d %H:%M:%S',
                                   header=["date", "output", "c_log_reward", "lm_log_reward"])
        # prompt format
        if "gpt" in args.victim_model or "dolly" in args.victim_model:
            self.prompt_fn = self.make_prompt
        else:
            self.prompt_fn = self.make_chat_prompt

    def get_total_reward_temp(self, step):
        args = self.args
        diff = args.reward_sched_end - args.reward_sched_start
        temp = args.reward_sched_start + diff * \
            min(1, step / args.reward_sched_horizon)
        return temp

    def get_lm_reward_temp(self, step):
        diff = self.args.lm_sched_end - self.args.lm_sched_start
        temp = self.args.lm_sched_start + diff * \
            min(1, step / self.args.lm_sched_horizon)
        return temp

    @torch.no_grad()
    def get_avg_pairwise_cossim(self, sentences):
        embeddings = self.sentence_encoder.encode(
            sentences, convert_to_tensor=True)
        cos_sim = F.cosine_similarity(embeddings.unsqueeze(
            0), embeddings.unsqueeze(1), -1).cpu()
        off_diag = cos_sim.masked_select(
            ~torch.eye(cos_sim.size(0), dtype=bool)).view(-1)
        avg_sim = torch.mean(off_diag).item()

        return avg_sim

    def simulate_experience(self, batch,  rbuffer, beta, max_len):
        policy = random.randint(0, 1)  # integer from [0,1]
        if policy == 0:
            choice = random.randint(0, 1)
            temp = random.uniform(self.args.temp_low,
                                  self.args.temp_high) if choice == 0 else 1.0
            results = self.get_online_samples(
                batch, max_len=max_len, temp=temp)

            c_log_reward = results["c_log_reward"]
            lm_log_reward = results["lm_log_reward"]
            log_reward = lm_log_reward + (c_log_reward / beta)

            prompt_ids = batch["input_ids"]

            prompts_responses = results["prompts_responses"]
            responses = prompts_responses[:, prompt_ids.size(1):]

            decoded_responses = results["decoded_responses"]

            response_embs = self.sentence_encoder.encode(
                decoded_responses, convert_to_tensor=True)

            rbuffer.add_batch(responses, decoded_responses, response_embs,
                              c_log_reward, lm_log_reward, log_reward)

            for i in range(responses.size(0)):
                self.csvlogger.info(
                    ['"'+decoded_responses[i]+'"', c_log_reward[i].item(), lm_log_reward[i].item()])

        else:
            bs = batch["input_ids"].size(0)
            # sample from buffer
            response_batch, reward_batch = rbuffer.sample(bs)
            results = self.get_offline_samples(
                batch, response_batch, reward_batch)

        return results

    def get_logpf_and_logz(self, prompt_batch, response_batch):
        prompt_len = prompt_batch["input_ids"].size(1)

        # gpu allocation
        prompt_batch = {k: v.to(self.device) for k, v in prompt_batch.items()}
        response_batch = {k: v.to(self.device)
                          for k, v in response_batch.items()}

        concat_inputs = dict()
        for k in prompt_batch.keys():
            concat_inputs[k] = torch.cat(
                [prompt_batch[k], response_batch[k]], 1)

        outputs = self.model(**concat_inputs, output_hidden_states=True)

        # compute z
        last_hidden = outputs.hidden_states[-1]
        prompt_hidden = last_hidden[:, :prompt_len]
        prompt_attention_mask = prompt_batch["attention_mask"]

        avg_pool = avg_pooling(prompt_hidden, prompt_attention_mask)
        log_z = self.model.proj_z(avg_pool).squeeze(-1)

        logits = outputs.logits[:, prompt_len-1:-1]
        responses = response_batch["input_ids"]

        log_prob = F.log_softmax(logits, dim=-1)
        log_prob = torch.gather(
            log_prob, -1, responses.unsqueeze(-1)).squeeze(-1)
        log_prob = log_prob.masked_fill(
            response_batch["attention_mask"] == 0, 0.0)
        sum_logpf = torch.sum(log_prob, dim=1)

        return sum_logpf, log_z

    def get_offline_samples(self, prompt_batch, response_batch, reward_batch):
        sum_logpf, log_z = self.get_logpf_and_logz(
            prompt_batch, response_batch)
        reward_batch = {k: v.to(self.device) for k, v in reward_batch.items()}

        decoded_responses = self.tokenizer.batch_decode(
            response_batch["input_ids"], skip_special_tokens=True)

        results = {
            "log_z": log_z,
            "sum_logpf": sum_logpf,
            "decoded_responses": decoded_responses
        }
        results.update(reward_batch)

        return results

    def get_logreward(self,
                      prompt_inputs: Dict[str, Union[List, torch.LongTensor]],
                      prompts_responses: torch.LongTensor
                      ):
        # prompt_inputs: input_ids, attention_mask of prompt
        # prompts_responses: concatenation of prompt and response
        prompt_len = prompt_inputs["input_ids"].size(1)
        only_responses = prompts_responses[:, prompt_len:]
        # the first pad token is EOS
        pad_mask = (only_responses ==
                    self.tokenizer.pad_token_id).cumsum(1) > 1
        attention_mask = torch.cat(
            [prompt_inputs["attention_mask"], (~pad_mask).long()], 1)

        # llh from reference model
        with torch.no_grad():
            lora_to_base(self.model)
            outputs = self.model(input_ids=prompts_responses,
                                 attention_mask=attention_mask)
            logits = outputs.logits[:, prompt_len-1:-1]
            log_prob = F.log_softmax(logits, dim=-1)
            labels = prompts_responses[:, prompt_len:]

            lm_logreward = torch.gather(
                log_prob, -1, labels.unsqueeze(2)).squeeze(2)
            lm_logreward = torch.where(pad_mask, 0.0, lm_logreward)
            lm_logreward = torch.sum(lm_logreward, 1)
            base_to_lora(self.model)

        # length penalty
        response_lengths = torch.sum((~pad_mask).long(), 1)
        lm_logreward = torch.where(
            response_lengths < self.args.min_len, -500, lm_logreward)

        # reward model
        decoded_responses = self.tokenizer.batch_decode(only_responses,
                                                        skip_special_tokens=True)

        victim_prompts = [self.prompt_fn(x) for x in decoded_responses]

        llm_outputs = self.victim_model.generate(
            victim_prompts, self.sampling_params, use_tqdm=False)
        attack_prompts = []
        victim_responses = []
        for i, output in enumerate(llm_outputs):
            # for each prompt we get multiple response
            for response in output.outputs:
                victim_responses.append(response.text)
                attack_prompts.append(decoded_responses[i])


        # 보상함수 교체하기
        c_log_reward = self.toxicity_fn.compute(responses=victim_responses)
        chunks = torch.split(c_log_reward, self.args.num_r_samples, dim=0)
        c_log_reward = torch.stack(chunks, dim=0)  # [b, r]
        # c_log_reward = torch.log(c_reward)

        avg_c_log_reward = c_log_reward.mean(1).to(self.device)

        return lm_logreward, avg_c_log_reward, decoded_responses

    def get_online_samples(self, batch, max_len, temp=1.0):
        # input_ids is left-side padded
        outputs = generate_and_return_z_logprob(
            model=self.model,
            prompt_ids=batch["input_ids"],
            prompt_attention_mask=batch["attention_mask"],
            eos_token_id=self.tokenizer.eos_token_id,
            temperature=temp,
            max_len=max_len
        )
        prompts_responses = outputs["actions"]
        log_z = outputs["log_z"]
        sum_logpf = outputs["sum_logpf"]

        lm_logreward, c_log_reward, decoded_responses = self.get_logreward(
            batch, prompts_responses)

        results = {"lm_log_reward": lm_logreward,
                   "c_log_reward": c_log_reward,
                   "log_z": log_z,
                   "sum_logpf": sum_logpf,
                   "prompts_responses": prompts_responses,
                   "decoded_responses": decoded_responses,
                   }

        return results

    @staticmethod
    def make_prompt(instruction):
        prompt_template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"

        return prompt_template.format(instruction=instruction.rstrip())

    def make_chat_prompt(self, instruction):
        return self.victim_model_tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction.rstrip()}],
            tokenize=False,
            add_generation_prompt=True)

    def compute_tb_loss(self, log_z, sum_logpf, log_reward):
        delta = log_z + sum_logpf - log_reward
        losses = delta**2
        return losses

    def get_batch_metrics(self, batch, step, rbuffer,  max_len, beta,  train=True):
        metrics = {}
        train_test = 'train' if train else 'eval'

        results = self.simulate_experience(
            batch,  rbuffer, beta=beta, max_len=max_len)

        if step % 100 == 0:
            cos_sim = self.get_avg_pairwise_cossim(
                results["decoded_responses"])
            metrics["cos-sim"] = [cos_sim]

        c_log_reward = results["c_log_reward"]
        lm_log_reward = results["lm_log_reward"]

        gamma = self.get_lm_reward_temp(step)
        log_reward = (lm_log_reward / gamma) + (c_log_reward / beta)

        rew_temp = self.get_total_reward_temp(step)
        tempered_log_reward = log_reward / rew_temp

        losses = self.compute_tb_loss(
            results["log_z"], results["sum_logpf"], tempered_log_reward)

        metrics[f"log_z"] = results["log_z"].detach().tolist()
        metrics[f"c_log_reward/{train_test}"] = c_log_reward.tolist()
        metrics[f"lm_log_reward/{train_test}"] = lm_log_reward.tolist()
        metrics[f"log_reward/{train_test}"] = log_reward.tolist()
        metrics[f"loss/{train_test}"] = losses.detach().tolist()

        return losses.mean(), metrics

    def eval(self):
        num_samples = math.ceil(
            self.args.eval_batch_size / self.args.batch_size)
        all_log_reward = []
        all_lm_log_reward = []
        all_c_log_reward = []
        all_decoded_responses = []
        gamma = self.get_lm_reward_temp(self.args.train_steps)
        for _ in tqdm(range(num_samples), desc="eval"):
            batch = next(self.train_iter)
            batch = batch.to(self.device)
            batch = {k: v.repeat(self.args.batch_size, 1)
                     for k, v in batch.items()}

            self.model.eval()
            prompts_responses = self.model.generate(
                **batch,
                do_sample=True,
                max_new_tokens=self.args.max_len,
                temperature=1.0,
                min_new_tokens=self.args.min_len,
                pad_token_id=self.tokenizer.pad_token_id
            )
            lm_log_reward, c_log_reward, decoded_responses = self.get_logreward(
                batch, prompts_responses)

            log_reward = (lm_log_reward / gamma) + \
                (c_log_reward / self.args.beta)
            all_log_reward.append(log_reward.cpu())
            all_lm_log_reward.append(lm_log_reward.cpu())
            all_c_log_reward.append(c_log_reward.cpu())
            all_decoded_responses.extend(decoded_responses)

        all_log_reward = torch.cat(all_log_reward)
        all_lm_log_reward = torch.cat(all_lm_log_reward)
        all_c_log_reward = torch.cat(all_c_log_reward)

        embs = self.sentence_encoder.encode(all_decoded_responses)
        embs = torch.from_numpy(embs)
        cos_sim = batch_cosine_similarity_kernel(embs)

        asr = torch.mean((all_c_log_reward > math.log(0.5)).float())
        metrics = {
            "cos_sim/eval": cos_sim,
            "c_log_reward/eval": all_c_log_reward.mean().item(),
            "lm_log_reward/eval": all_lm_log_reward.mean().item(),
            "log_reward/eval": all_log_reward.mean().item(),
            "asr": asr.item()
        }
        return metrics

    def save(self, output_dir, rbuffer, step):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        ckpt = {"global_step": step,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "proj_z": self.model.proj_z.state_dict()}
        ckpt_file = os.path.join(output_dir, "ckpt.pt")
        torch.save(ckpt, ckpt_file)

        rbuffer.save(os.path.join(output_dir, "buffer.pkl"))

    def load(self, output_dir, model, optimizer, scheduler, rbuffer):
        # load checkpoint and return starting step
        if not os.path.exists(output_dir):
            return 1
        dirs = sorted(os.listdir(output_dir))
        if len(dirs) == 0:
            return 1
        else:
            dirs = [int(x) for x in dirs if x.isdigit()]
            dirs = sorted(dirs, reverse=True)
            ckpt_dir = os.path.join(output_dir, str(dirs[0]))
            _model = AutoModelForCausalLM.from_pretrained(self.args.sft_ckpt)
            _model = PeftModel.from_pretrained(_model, ckpt_dir)
            # we do not load proj_z here
            msg = model.load_state_dict(_model.state_dict(), strict=False)
            print(msg)

            # load optimizer, scheduler, and proj_z
            ckpt = torch.load(os.path.join(ckpt_dir, "ckpt.pt"))
            model.proj_z.load_state_dict(ckpt["proj_z"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])

            # load buffer
            buffer_ckpt = os.path.join(ckpt_dir, "buffer.pkl")
            rbuffer.load(buffer_ckpt)

            return ckpt["global_step"] + 1

    def init_buffer(self, prompt_ids, attention_mask, rbuffer):
        # initialize the buffer with sft dataset
        with open(self.args.few_shot_file, "r") as f:
            data = json.load(f)
            instructions = [" " + x["instruction"].strip() for x in data]
        orig_padding_side = self.tokenizer.padding_side

        self.tokenizer.padding_side = "right"
        inputs = self.tokenizer(
            instructions, padding=True, add_special_tokens=False, return_tensors="pt")
        self.tokenizer.padding_side = orig_padding_side

        # add eos_token
        responses = inputs["input_ids"]

        eos_tokens = torch.ones(responses.size(
            0), 1, dtype=torch.long) * self.tokenizer.eos_token_id
        responses = torch.cat([responses, eos_tokens], 1)

        ds = TensorDataset(responses)
        dataloader = DataLoader(ds, self.args.batch_size, shuffle=False)
        for batch in tqdm(dataloader, desc="init buffer with sft dataset"):
            batch_response = batch[0]

            batch_response = batch_response.to(prompt_ids.device)
            batch_prompts = prompt_ids.repeat(batch_response.size(0), 1)
            prompts_responses = torch.cat(
                [batch_prompts, batch_response], dim=1)

            batch = {"input_ids": batch_prompts,
                     "attention_mask": attention_mask.repeat(prompts_responses.size(0), 1)
                     }

            lm_log_reward, c_log_reward, decoded_responses = self.get_logreward(
                batch, prompts_responses)

            log_reward = lm_log_reward + (c_log_reward / self.args.beta)

            embs = self.sentence_encoder.encode(
                decoded_responses, convert_to_tensor=True)
            rbuffer.add_batch(responses, decoded_responses, embs,
                              c_log_reward, lm_log_reward, log_reward
                              )

    def train(self):
        # if the buffer is empty, we seed it with inital policy
        if self.rbuffer.size() == 0:
            lora_to_base(self.model)
            # we alyways have the same prompt
            batch = next(self.train_iter)
            batch = batch.to(self.device)

            self.init_buffer(batch["input_ids"],
                             batch["attention_mask"], self.rbuffer)
            batch = {k: v.repeat(self.args.batch_size, 1)
                     for k, v in batch.items()}

            for _ in tqdm(range(10), desc="init buffer", leave=False):
                prompts_responses = self.model.generate(
                    **batch,
                    do_sample=True,
                    max_new_tokens=self.args.max_len,
                    temperature=0.7,
                    top_p=0.9,
                    min_new_tokens=self.args.min_len,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                lm_log_reward, c_log_reward, decoded_responses = self.get_logreward(
                    batch, prompts_responses)

                log_reward = lm_log_reward + (c_log_reward / self.args.beta)
                prompt_ids = batch["input_ids"]
                responses = prompts_responses[:, prompt_ids.size(1):]

                eos_tokens = torch.ones(prompt_ids.size(
                    0), 1, dtype=torch.long) * self.tokenizer.eos_token_id
                eos_tokens = eos_tokens.to(prompt_ids.device)

                responses = torch.cat([responses, eos_tokens], 1)

                embs = self.sentence_encoder.encode(
                    decoded_responses, convert_to_tensor=True)
                self.rbuffer.add_batch(responses, decoded_responses, embs,
                                       c_log_reward, lm_log_reward, log_reward
                                       )
            base_to_lora(self.model)

        # prepare a prompt
        batch = next(self.train_iter)
        batch = batch.to(self.device)
        batch = {k: v.repeat(self.args.batch_size, 1)
                 for k, v in batch.items()}

        t = tqdm(range(self.start, self.args.train_steps+1),
                 desc="training", dynamic_ncols=True)
        for global_step in t:
            batch_metrics = defaultdict(list)

            self.model.train()
            self.optimizer.zero_grad()
            for _ in range(self.args.grad_acc_steps):
                loss, metrics = self.get_batch_metrics(
                    batch, global_step, self.rbuffer,
                    self.args.max_len, self.args.beta)

                for k, v in metrics.items():
                    batch_metrics[k].extend(v)
                loss = loss / self.args.grad_acc_steps
                loss.backward()

            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.args.max_norm)

            self.optimizer.step()
            self.scheduler.step()

            # logging
            batch_metrics = {k: sum(v) / float(len(v))
                             for k, v in batch_metrics.items()}
            wandb.log(batch_metrics, step=global_step)

            t.set_description(
                f"Step {global_step}: {formatted_dict(batch_metrics)}")

            if global_step % self.args.eval_period == 0:
                output_dir = os.path.join(
                    self.args.save_dir, f"{self.args.exp_name}/{global_step}")
                self.save(output_dir, self.rbuffer, global_step)

        output_dir = os.path.join(
            self.args.save_dir, self.args.exp_name, "latest")
        self.save(output_dir, self.rbuffer, global_step)

        eval_metrics = self.eval()
        wandb.log(eval_metrics, step=global_step)
        wandb.finish()

"""
export SFT_CKPT=save/gpt2-sft-position-final/latest

python main.py \
--exp_name llama-gfn-pii-lora \
--sim_tolerance 0.3 \
--victim_model ./save/email-lora-v2/latest \
--lr 1e-4 \
--reward_sched_horizon 1000 \
--train_steps 5000 \
--buffer_size 5000 \
--seed 42 \
--max_len 20 \
--temp_low 0.7 \
--temp_high 2.0 \
--lm_sched_end 1.2 \
--lm_sched_horizon 2000 \
--sft_ckpt $SFT_CKPT \
--compare c_reward \
--prioritization c_reward \
--beta 0.1 \
--metric cosine
"""