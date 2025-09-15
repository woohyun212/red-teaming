import argparse
import json
import os

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def make_chat_prompt(instruction, tokenizer):
    return tokenizer.apply_chat_template([{"role":"user", "content": instruction.rstrip()}], tokenize=False, add_generation_prompt=True)

def run(args):
    with open(args.input_file, "r") as f:
        data = json.load(f)
    # adversarial prompts
    instructions = [x["prompt"] for x in data[1:] if x["avg-toxic-score"] > 0.5]
    
    model_id = "google/gemma-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    system_prompt = "Refuse answering to the following instruction."
    sentences = [f"{system_prompt}\n{instruction}" for instruction in instructions]

    prompts =  [make_chat_prompt(x, tokenizer) for x in sentences]


    model = LLM(model_id, dtype="bfloat16")
    sampling_params = SamplingParams(n=1, top_p=0.9, temperature=0.1, max_tokens=256, stop=["\n"])

    llm_outputs = model.generate(prompts, sampling_params=sampling_params)

    results = []
    for i,output in enumerate(llm_outputs):
        print(f"prompt: {output.attack_prompt}\n")
        response = output.outputs[0].text
        print(response)
        instruction = instructions[i]
        results.append({"instruction": instruction.strip(), "response": response.strip()})
    output_dir = "safety_dataset"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, f"{args.output_file}.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)

    args = parser.parse_args()
    run(args)    


