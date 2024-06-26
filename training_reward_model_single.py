# Reference: https://github.com/joeljang/RLPHF
import os

import torch
import evaluate
import numpy as np
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    LlamaTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.utils import PaddingStrategy
import multiprocessing
import functools
from scipy import stats
import math
from torch.utils.data import Subset, ConcatDataset
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})
    resume_from_checkpoint: Optional[bool] = field(
        default=False,
        metadata={"help": "If you want to resume training where it left off."},
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    report_to: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to report training log to."
        },
    )

    max_seq_length: Optional[int] = field(default=512)
    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    learning_rate: Optional[float] = field(default=2e-5)
    weight_decay: Optional[int] = field(default=0.001)
    seed: Optional[int] = field(default=1103)
    max_length: Optional[int] = field(default=512)
    log_freq: Optional[int] = field(default=1)
    eval_freq: Optional[int] = field(default=400)
    save_freq: Optional[int] = field(default=400)
    save_total_limit: Optional[int] = field(default=3)
    lora_r: Optional[int] = field(default=8)
    lora_alpha: Optional[int] = field(default=32)
    lora_dropout: Optional[float] = field(default=0.1)
    model_name: Optional[str] = field(
        default="tulu-7b",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub or local."
        },
    )
    dataset_name1: Optional[str] = field(
        default="data/rm_training/P1A.json",
        metadata={"help": "The dataset name"},
    )
    dataset_name2: Optional[str] = field(
        default="data/rm_training/P1B.json",
        metadata={"help": "The dataset name"},
    )
    eval_dataset_name: Optional[List[str]] = field(
        default=None,
        metadata={"help": "The dataset name"},
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    train_subset1: Optional[int] = field(
        default=0,
        metadata={"help": "The size of the subset of the training data to use"},
    )
    train_subset2: Optional[int] = field(
        default=0,
        metadata={"help": "The size of the subset of the training data to use"},
    )
    eval_subset: Optional[int] = field(
        default=0,
        metadata={"help": "The size of the subset of the eval data to use"},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="adamw_hf",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear",
        metadata={"help": "The lr scheduler"},
    )
    output_dir: Optional[str] = field(default="./checkpoints/training_reward_modelP2AEpoch1/",
                                      metadata={"help": "n steps to save the model"})

# We need to define a special data collator that batches the data in our j vs k format.
@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_j = []
        features_k = []
        for feature in features:
            features_j.append(
                {
                    "input_ids": feature["input_ids_j"],
                    "attention_mask": feature["attention_mask_j"],
                }
            )
            features_k.append(
                {
                    "input_ids": feature["input_ids_k"],
                    "attention_mask": feature["attention_mask_k"],
                }
            )
        batch_j = self.tokenizer.pad(
            features_j,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_k = self.tokenizer.pad(
            features_k,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_j": batch_j["input_ids"],
            "attention_mask_j": batch_j["attention_mask"],
            "input_ids_k": batch_k["input_ids"],
            "attention_mask_k": batch_k["attention_mask"],
            "return_loss": True,
        }
        return batch


PREF_PROMPTS = [
    "Generate a response that can be easily understood by an elementary school student.",
    "Generate a response that only a PhD Student in that specific field could understand.",
    "Generate a response that is concise and to the point, without being verbose.",
    "Generate a response that is very informative, without missing any background information.",
    "Generate a response that is friendly, witty, funny, and humorous, like a close friend.",
    "Generate a response in an unfriendly manner.",
    "Generate a response in a sassy manner.",
    "Generate a response in a sarcastic manner."
]

# Turn the dataset into pairs of post + summaries, where text_j is the preferred question + answer and
# text_k is the other. Then tokenize the dataset.
def preprocess_function(examples, args, tokenizer):
    new_examples = {
        "input_ids_j": [],
        "attention_mask_j": [],
        "input_ids_k": [],
        "attention_mask_k": [],
    }
    for question, response_j, response_k in zip(examples["user_input"], examples["completion_a"],
                                                examples["completion_b"]):
        for pref_prompt in PREF_PROMPTS:
            if pref_prompt in question:
                question = question.replace(f'{pref_prompt}', '')
                break
        question = f"<|user|>\n{question} \n<|assistant|>\n"
        tokenized_j = tokenizer(question + response_j, truncation=True, max_length=args.max_seq_length)
        tokenized_k = tokenizer(question + response_k, truncation=True, max_length=args.max_seq_length)

        new_examples["input_ids_j"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_j"].append(tokenized_j["attention_mask"])
        new_examples["input_ids_k"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_k"].append(tokenized_k["attention_mask"])

    return new_examples


def compute_metrics(eval_pred):
    # Define the metric that we'll use for validation.
    accuracy = evaluate.load("accuracy")
    predictions, _ = eval_pred
    # Here, predictions is rewards_j and rewards_k.
    # We want to see how much of the time rewards_j > rewards_k.
    rewards_j_stats = stats.describe(predictions[0])
    rewards_k_stats = stats.describe(predictions[1])
    print("rewards_j_mean", rewards_j_stats.mean[0])
    print("rewards_k_mean", rewards_k_stats.mean[0])
    print("rewards_total_mean", (rewards_j_stats.mean[0] + rewards_k_stats.mean[0]) / 2)
    print("rewards_j_std", math.sqrt(rewards_j_stats.variance[0]))
    print("rewards_k_std", math.sqrt(rewards_k_stats.variance[0]))
    print("rewards_total_std", math.sqrt((rewards_j_stats.variance[0] + rewards_k_stats.variance[0]) / 2))
    
    predictions = np.argmax(predictions, axis=0)
    labels = np.zeros(predictions.shape)
    result = accuracy.compute(predictions=predictions, references=labels)
    print("accuracy:", result)
    return result


class RewardTrainer(Trainer):
    # Define how to compute the reward loss. We use the InstructGPT pairwise logloss: https://arxiv.org/abs/2203.02155
    def compute_loss(self, model, inputs, return_outputs=False):
        rewards_j = model(input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"])[0]
        rewards_k = model(input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])[0]
        #loss = (-nn.functional.logsigmoid(rewards_j - rewards_k) + (beta * l2)).mean()
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss

def main(script_args):
    import os
    os.environ['WANDB_DISABLED'] = 'true'
    # Loading Model
    if "decapoda" in script_args.model_name.lower():
        tokenizer = LlamaTokenizer.from_pretrained(script_args.model_name, use_fast=False)
        # required for llama
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
                "pad_token": DEFAULT_PAD_TOKEN,
            }
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, use_fast=False)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Load the dataset for tuning the reward model.
    # print("script_args.dataset_name",script_args.dataset_name)
    print("script_args.output_dir",script_args.output_dir)
    data_path1 = script_args.dataset_name1
    if data_path1.endswith(".json") or data_path1.endswith(".jsonl"):
        dataset1 = load_dataset("json", data_files=data_path1, split="train")
    else:
        dataset1 = load_dataset(data_path1, split="train")
    data_path2 = script_args.dataset_name2
    if data_path2.endswith(".json") or data_path2.endswith(".jsonl"):
        dataset2 = load_dataset("json", data_files=data_path2, split="train")
    else:
        dataset2 = load_dataset(data_path2, split="train")
    # 
    dataset1 = dataset1.shuffle(seed=script_args.seed)
    dataset1 = dataset1.train_test_split(test_size=0.1, seed=script_args.seed)
    train_dataset1 = dataset1["train"]
    eval_dataset1 = dataset1["test"]
    dataset2 = dataset2.shuffle(seed=script_args.seed)
    dataset2 = dataset2.train_test_split(test_size=0.1, seed=script_args.seed)
    train_dataset2 = dataset2["train"]
    eval_dataset2 = dataset2["test"]

    print("original train_dataset1",len(train_dataset1))
    print("original train_dataset2",len(train_dataset2))
    if script_args.train_subset1 > 0:
        train_dataset1 = train_dataset1.select(range(len(train_dataset1)//script_args.train_subset1))
        print("train_dataset1",len(train_dataset1))
              
    eval_dataset1 = eval_dataset1.select(range(250))
    
    if script_args.train_subset2 > 0:
        train_dataset2 = train_dataset2.select(range(len(train_dataset2)//script_args.train_subset2))
        print("train_dataset2",len(train_dataset2))
        
    eval_dataset2 = eval_dataset2.select(range(250))

    # Define the training args. Needs to be done before the model is loaded if you are using deepspeed.
    model_name_split = script_args.model_name.split("/")[-1]
    output_name = (
        f"experiment"
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(script_args.output_dir, output_name),
        learning_rate=script_args.learning_rate,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        weight_decay=script_args.weight_decay,
        # evaluation_strategy="steps",
        evaluation_strategy="epoch",
        # eval_steps=script_args.eval_freq,
        #save_strategy="steps",
        save_strategy="epoch",
        save_steps=script_args.save_freq,
        #save_total_limit=script_args.save_total_limit,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        deepspeed=script_args.deepspeed,
        local_rank=script_args.local_rank,
        remove_unused_columns=False,
        label_names=[],
        bf16=script_args.bf16,
        logging_strategy="steps",
        logging_steps=script_args.log_freq,
        optim=script_args.optim,
        lr_scheduler_type=script_args.lr_scheduler_type,
        report_to='none',
    )

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name, 
        num_labels=1, 
        load_in_8bit=True,
        torch_dtype=torch.bfloat16
    )

    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()
    model.config.use_cache = script_args.gradient_checkpointing
    print("Saving Location:", script_args.output_dir + "peft_last_checkpoint")
    print("Number of training epoch",script_args.num_train_epochs)
    #num_proc = 24  # Can adjust to be higher if you have more processors.
    num_proc = multiprocessing.cpu_count() # Setting the num of processors same as cpu count
    original_columns = train_dataset1.column_names
    # preprocess the dataset and filter out QAs that are longer than max_length
    train_dataset1 = train_dataset1.map(
        functools.partial(preprocess_function, args=script_args, tokenizer=tokenizer), batched=True, num_proc=num_proc, remove_columns=original_columns
    )
    train_dataset1 = train_dataset1.filter(
        lambda x: len(x["input_ids_j"]) <= script_args.max_length and len(x["input_ids_k"]) <= script_args.max_length)
    eval_dataset1 = eval_dataset1.map(
        functools.partial(preprocess_function, args=script_args, tokenizer=tokenizer), batched=True, num_proc=num_proc, remove_columns=original_columns
    )
    eval_dataset1 = eval_dataset1.filter(
        lambda x: len(x["input_ids_j"]) <= script_args.max_length and len(x["input_ids_k"]) <= script_args.max_length)
    
    train_dataset2 = train_dataset2.map(
        functools.partial(preprocess_function, args=script_args, tokenizer=tokenizer), batched=True, num_proc=num_proc, remove_columns=original_columns
    )
    train_dataset2 = train_dataset2.filter(
        lambda x: len(x["input_ids_j"]) <= script_args.max_length and len(x["input_ids_k"]) <= script_args.max_length)
    eval_dataset2 = eval_dataset2.map(
        functools.partial(preprocess_function, args=script_args, tokenizer=tokenizer), batched=True, num_proc=num_proc, remove_columns=original_columns
    )
    eval_dataset2 = eval_dataset2.filter(
        lambda x: len(x["input_ids_j"]) <= script_args.max_length and len(x["input_ids_k"]) <= script_args.max_length)
    train_dataset = ConcatDataset([train_dataset1, train_dataset2])
    # train_dataset = train_dataset.shuffle(seed=script_args.seed)
    eval_dataset = ConcatDataset([eval_dataset1, eval_dataset2])
    # eval_dataset = eval_dataset.shuffle(seed=script_args.seed)
    
    eval_datasets = {} # Having multiple evalsets
    # if custom_eval_datasets:
    #     indx = 1
    #     for custom_eval_dataset in custom_eval_datasets:
    #         custom_eval_dataset = custom_eval_dataset.map(
    #             functools.partial(preprocess_function, args=script_args, tokenizer=tokenizer), batched=True, num_proc=num_proc, remove_columns=original_columns
    #         )
    #         custom_eval_dataset = custom_eval_dataset.filter(
    #             lambda x: len(x["input_ids_j"]) <= script_args.max_length and len(x["input_ids_k"]) <= script_args.max_length)
    #         eval_datasets[f'custom_eval_{indx}'] = custom_eval_dataset
    #         indx+=1
    eval_datasets['original_eval'] = eval_dataset

    # Train the model
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # eval_dataset=train_dataset,
        compute_metrics=compute_metrics,
        data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=script_args.max_length)
    )

    trainer.train(script_args.resume_from_checkpoint)
    print("Evaluating the model on dataset1")
    trainer.evaluate(eval_dataset=eval_dataset1)
    print("Evaluating the model on dataset2")
    trainer.evaluate(eval_dataset=eval_dataset2)
    print("Saving last checkpoint of the model")
    
    model.save_pretrained(script_args.output_dir + "peft_last_checkpoint")

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    set_seed(script_args.seed)
    main(script_args)