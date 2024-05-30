from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import set_seed, GenerationConfig, LlamaForCausalLM
from datasets import load_dataset, ReadInstruction
from tqdm import tqdm
import argparse
import torch
import json
from peft import get_peft_model, TaskType, LoraConfig
from trl import AutoModelForCausalLMWithValueHead


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="evaluation/koala_eval_50.json")
    # parser.add_argument("--start_per", type=int, default=0)
    # parser.add_argument("--end_per", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="generations/test.json")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--decoding_temperature", type=int, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument('--checkpoint_dirs', action='append', default=None)
    parser.add_argument('--tokenizer', type=str, default="")
    parser.add_argument("--subset", type=int, default=50)
    parser.add_argument("--prompt", type=str, default="")
    return parser.parse_args()

def generate_prompt(personalized_instruction: str, instruction: str, input_ctxt: str = None) -> str:
    if input_ctxt!="":
        return f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n### Instruction:\n{instruction} {personalized_instruction}\n### Input:\n{input_ctxt}\n### Response:"
    else:
        return f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n### Instruction:\n{instruction} {personalized_instruction}\n### Response:"

def generate_prompt_tulu(personalized_instruction: str, instruction: str, input_ctxt: str = None) -> str:
    return f"<|user|>\n{instruction} {input_ctxt} {personalized_instruction}\n<|assistant|>\n"

def main(args):
    if args.checkpoint_dirs!=None:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            args.base_model,
            load_in_8bit=True,
            peft_config=lora_config,
        )
        ckpt_params_lst = []
        for checkpoint_dir in args.checkpoint_dirs:
            ckpt_params = torch.load(checkpoint_dir, map_location=f'cuda:0')
            ckpt_params_lst.append(ckpt_params)
        loading_cnt = 0
        for name, param in model.named_parameters():
            temp_name = name.replace('pretrained_model.', '')
            temp_name = temp_name.replace('.default', '')
            if temp_name in ckpt_params:
                param_sum = None
                for i in range(len(ckpt_params_lst)):
                    params = ckpt_params_lst[i][temp_name]
                    if param_sum == None:
                        param_sum = params
                    else:
                        param_sum += params
                final_param = param_sum / len(ckpt_params_lst)
                param.data = final_param
                loading_cnt+=1
        print(f'loaded {loading_cnt} LoRA weights!')
        model.to('cuda')
    elif args.checkpoint_dir!=None:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            args.base_model,
            load_in_8bit=True,
            peft_config=lora_config,
        )
        ckpt_params = torch.load(args.checkpoint_dir, map_location='cuda:0')
        loading_cnt = 0
        for name, param in model.named_parameters():
            temp_name = name.replace('pretrained_model.', '')
            temp_name = temp_name.replace('.default', '')
            if temp_name in ckpt_params:
                param.data = ckpt_params[temp_name]
                loading_cnt+=1
        print(f'loaded {loading_cnt} LoRA weights!')
        model.to('cuda')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="auto", load_in_4bit=True)
        
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    generation_config = GenerationConfig(
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_k = 40,
            top_p = 0.8,
            no_repeat_ngram_size=3,
            temperature=0.5,
            do_sample=True,
            num_return_sequences=1,
            repetition_penalty=1.15,
            max_new_tokens=512,
            # min_new_tokens=32
        )

    data_path = args.dataset_name
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=data_path, split=args.split)
    else:
        dataset = load_dataset(data_path, split=args.split)

    dataloader = torch.utils.data.DataLoader(dataset)

    entries = []
    indx = 0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            input_entries = []
            instructions = []
            inputs = []
            print(batch['prompt'])
            for j in range(len(batch['prompt'])):
                if indx >=args.subset:
                    break
                instruction = batch['prompt'][j]
                input_entry = generate_prompt(args.prompt, batch['prompt'][j], "")
                input_entries.append(input_entry)
        
                model_inputs = tokenizer(input_entry, return_tensors="pt").to("cuda:0")
                output = model.generate(generation_config = generation_config,
                    max_new_tokens = 512,
                    # return_dict_in_generate = True,
                    **model_inputs)
                
                output_text = tokenizer.decode(output[0], skip_special_tokens=True)
                output_text = output_text[len(input_entry)+1:]
                
                print(f'INPUT: {input_entry}')
                print(f'OUTPUT: ', output_text)
                entry = {
                    "id": indx,
                    "prompt": input_entry,
                    "output": output_text
                }
                entries.append(entry)
                indx+=1

    with open(args.output_dir, "w") as f:
        json.dump(entries, f, indent=4)
    

if __name__ == "__main__":
    args = get_args()
    assert args.base_model != "", "Please provide the llama model path"
    set_seed(args.seed)
    main(args)