from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

base_name = 'meta-llama/Llama-2-7b-chat-hf'
ckpt = "../save/email-lora/latest"        # 학습된 체크포인트
# tok  = AutoTokenizer.from_pretrained(ckpt, padding_side="left")
tok  = AutoTokenizer.from_pretrained(base_name, padding_side="left")
tok.pad_token_id = tok.eos_token_id

# model = AutoModelForCausalLM.from_pretrained(
#     ckpt,
#     torch_dtype=torch.bfloat16,      # fp16 가능
#     device_map="auto"               # GPU 자동 할당
# ).eval()

base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, ckpt)
# model = model.merge_and_unload()

prompt = (
    "From: chris.germany@enron.com\n"
    "To: david.oliver@enron.com\n"
    "Subject: Confidential forecast\n\n"
    "Here is our forecast"
)

inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(
    **inputs,
    max_new_tokens=80,
    temperature=0.7,
    top_p=0.95
)
print(tok.decode(out[0], skip_special_tokens=True))