import os
from typing import Literal

from tap import Tap

from trainers.gfn_trainer import GFNTrainer
from trainers.mle_trainer import MLETrainer
from trainers.safety_trainer import SafetyTrainer
from trainers.sft_trainer import SFTTrainer
from utils import load_victim_config, seed

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Argument(Tap):
    baseline: bool = False
    mode: Literal["gfn", "sft", "mle", "safety"] = "gfn"
    model_name: str = "gpt2"
    victim_model: str = "vicgalle/gpt2-alpaca"
    sft_ckpt: str = "save/gpt2-sft-position-final/latest"
    save_dir: str = "./save"

    prompt_file: str = "prompts/attack_prompt.jsonl"
    few_shot_file: str = "prompts/sft_dataset.json"

    epochs: int = 1
    lr: float = 1e-4
    max_norm: float = 1.0
    weight_decay: float = 0.1

    num_warmup_steps: int = 100
    train_steps: int = 5000
    batch_size: int = 16
    grad_acc_steps: int = 8

    len_norm: bool = False
    max_len: int = 20
    min_len: int = 5

    victim_top_p: float = 0.92
    victim_max_len: int = 30
    victim_temp: float = 0.7
    use_4bit: bool = False

    load_buffer: bool = False
    buffer_size: int = 1000
    sim_tolerance: float = 0.25
    prioritization: Literal["c_reward", "reward", "uniform"] = "c_reward"
    buffer_ckpt: str = ""
    compare: str = "reward"
    metric: Literal["edit", "cosine"] = "edit"

    dtype: str = "float32"
    seed: int = 42

    eval_period: int = 500
    eval_batch_size: int = 1024
    # lora hparams
    lora: bool = False
    lora_r: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    # reward scaling
    beta: float = 0.1
    lm_sched_end: float = 1.0
    lm_sched_start: float = 1.0
    lm_sched_horizon: int = 2000

    # reward temperature
    reward_sched_start: float = 2.0
    reward_sched_end: float = 1.0
    reward_sched_horizon: int = 500

    # sampling temperature
    temp_low: float = 0.5
    temp_high: float = 2.0

    # victim model
    num_r_samples: int = 5
    do_sample: bool = True

    # wandb
    exp_name: str = "debug"
    wandb_project: str = "red-team"


if __name__ == "__main__":
    args = Argument(explicit_bool=True).parse_args()
    load_victim_config(args)
    seed(args.seed)
    if args.mode == "gfn":
        trainer = GFNTrainer(args)
    elif args.mode == "mle":
        trainer = MLETrainer(args)
    elif args.mode == "safety":
        trainer = SafetyTrainer(args)
    else:
        trainer = SFTTrainer(args)
    trainer.train()

"""
python main.py \
--exp_name llama-gfn \
--sim_tolerance 0.3 \
--victim_model meta-llama/Llama-2-7b-chat-hf \
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