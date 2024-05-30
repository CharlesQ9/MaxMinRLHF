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
import datasets
from torch.utils.data import Subset, ConcatDataset
import random
from tqdm import tqdm
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
        default= None,
        metadata={"help": "The dataset name"},
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[int] = field(
        default=3,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    train_subset: Optional[int] = field(
        default=3000,
        metadata={"help": "The size of the subset of the training data to use"},
    )
    cluster1_user: Optional[int] = field(
        default=30,
        metadata={"help": "The size of users for the biased data"},
    )
    cluster2_user: Optional[int] = field(
        default=30,
        metadata={"help": "The size of users for the biased data"},
    )
    eval_subset: Optional[int] = field(
        default=100,
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
    output_dir: Optional[str] = field(default="./checkpoints/training_reward_model/",
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
    return accuracy.compute(predictions=predictions, references=labels)


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

# new helper functions
def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

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
    # data_path = script_args.dataset_name
    
    # data_path1 = f"data/rm_training/P3A.json"
    # data_path2 = f"data/rm_training/P3B.json"
    # data_path3 = f"data/rm_training/P3C.json"
    # data_path4 = f"data/rm_training/P3D.json"
    data_path1 = script_args.dataset_name1
    data_path2 = script_args.dataset_name2
    # data_path1 = f"data/rm_training/P1A.json"
    # data_path2 = f"data/rm_training/P1B.json"
    
    # data_path_list = [data_path1, data_path2, data_path3, data_path4]
    data_path_list = [data_path1, data_path2]
    # data_path_list = [data_path3, data_path4]
    num_mixture = len(data_path_list)
    dataset_list = []
    for data_path in data_path_list:
        dataset_list.append(load_dataset("json", data_files=data_path, split="train"))
    for i in range(len(dataset_list)):
        dataset_list[i].shuffle(seed=script_args.seed)
        dataset_list[i]=dataset_list[i].select(range(script_args.train_subset))
    # data_path1 = f"data/rm_training/P1A.json"
    # data_path2 = f"data/rm_training/P1B.json"
    # data_path3 = f"data/rm_training/P2A.json"
    # data_path4 = f"data/rm_training/P2B.json"
    # dataset1 = load_dataset("json", data_files=data_path1, split="train")
    # dataset2 = load_dataset("json", data_files=data_path2, split="train")
    
    # dataset3 = load_dataset("json", data_files="data/rm_training/allcombo_16.json", split="train")
    # print(dataset3[0])
    # print(dataset1[0])
    # print(dataset1[1])
    # print(dataset2[0])
    
    # dataset1 = dataset1.train_test_split(test_size=0.1, seed=script_args.seed)
    # dataset2 = dataset2.train_test_split(test_size=0.1, seed=script_args.seed)
    

    # dataset = dataset.shuffle(seed=script_args.seed)
    # dataset = dataset.train_test_split(test_size=0.1, seed=script_args.seed)
    # dataset1 = dataset1.select(range(1000))
    # dataset2 = dataset2.select(range(1000))
    user_dataset_map = {}
    user_id = 0
    training_user_id = []
    evaluation_user_id = []
    for i in range(len(dataset_list)):
        original_columns = dataset_list[i].column_names
        num_proc = multiprocessing.cpu_count() # Setting the num of processors same as cpu count
        dataset_list[i] = dataset_list[i].map(
            functools.partial(preprocess_function, args=script_args, tokenizer=tokenizer), batched=True, num_proc=num_proc, remove_columns=original_columns
        )
        dataset_list[i] = dataset_list[i].filter(
            lambda x: len(x["input_ids_j"]) <= script_args.max_length and len(x["input_ids_k"]) <= script_args.max_length)
        dataset_length = len(dataset_list[i])
        subset_size = dataset_length // script_args.cluster1_user
        # Make sure the total size of all subsets equals the dataset length
        sizes = [subset_size] * script_args.cluster1_user
        if sum(sizes) < dataset_length:
            sizes[-1] += dataset_length - sum(sizes)
        subsets = torch.utils.data.random_split(dataset_list[i], sizes)
        counter = 0
        for subset in subsets:
            user_dataset_map[user_id] = subset
            if i == 0:
                num_train_user = (script_args.cluster1_user-20)
                
            if i==1:
                num_train_user = (script_args.cluster2_user-20)
                
            if counter < num_train_user:
                training_user_id.append(user_id)
            elif counter >= (script_args.cluster1_user-20):
                evaluation_user_id.append(user_id)
            counter += 1
            user_id += 1
    print("training_dataset", len(training_user_id))
    print(training_user_id)
    print("evaluation_user_id", len(evaluation_user_id))
    print(evaluation_user_id)

    del dataset_list
    torch.cuda.empty_cache()
    custom_eval_datasets = None

    # Define the training args. Needs to be done before the model is loaded if you are using deepspeed.
    model_name_split = script_args.model_name.split("/")[-1]
    output_name = (f"experiment")
    print("script_args.num_train_epochs",script_args.num_train_epochs)
    training_args = TrainingArguments(
        output_dir=os.path.join(script_args.output_dir, output_name),
        learning_rate=script_args.learning_rate,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        weight_decay=script_args.weight_decay,
        # evaluation_strategy="steps",
        evaluation_strategy="epoch",
        eval_steps=script_args.eval_freq,
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
    
    model_list =[]
    for i in range(num_mixture):
        model_list.append(AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name, 
        num_labels=1, 
        load_in_8bit=True,
        torch_dtype=torch.bfloat16
    ))
        model_list[i] = get_peft_model(model_list[i], peft_config)
        model_list[i].print_trainable_parameters()
        model_list[i].config.use_cache = script_args.gradient_checkpointing
    
    
    # print("#######################Model#######################")
    # print(model1)
    # print("###################################################")
    # model1 = get_peft_model(model1, peft_config)
    # model2 = get_peft_model(model2, peft_config)

    # model1.print_trainable_parameters()
    # model1.config.use_cache = script_args.gradient_checkpointing
    
    # model2.print_trainable_parameters()
    # model2.config.use_cache = script_args.gradient_checkpointing
    
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.empty_cache()
    
    num_EM_iterations = 10
    for j in range(num_EM_iterations):
        print("EM iteration:", j)
        training_dataset_list = {cluster_id: [] for cluster_id in range(num_mixture)}
        dataset_id_dict ={cluster_id: [] for cluster_id in range(num_mixture)}
        with torch.no_grad():
            for user_id in tqdm(training_user_id):
                current_dataset = user_dataset_map[user_id]
                if j==0:
                    # if user_id in [0,1,2,3,4,5,6,7,8,9,10]:
                    #     current_index = 0 
                    # elif user_id in [40,41,42,43,44,45,46,47,48,49]:
                    #     current_index = 1
                    # elif user_id in [80,81,82,83,84,85,86,87,88,89]:
                    #     current_index = 2 
                    # elif user_id in [120,121,122,123,124,125,126,127,128,129]:
                    #     current_index = 3 
                    # else:
                    #     continue
                    current_index = user_id%num_mixture
                    training_dataset_list[current_index].append(current_dataset)
                    dataset_id_dict[current_index].append(user_id)
                    continue
                current_dataloader = torch.utils.data.DataLoader(current_dataset, batch_size = 1, collate_fn=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=script_args.max_length))
                score_list = []
                for i in range(num_mixture):
                    reward_list_j = []
                    reward_list_k = []
                    for step, batch in enumerate(current_dataloader):
                        inputs = to_device(batch, device)
                        rewards_j = model_list[i](input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"])[0]
                        rewards_k = model_list[i](input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])[0]
                        reward_list_j.append(rewards_j)
                        reward_list_k.append(rewards_k)
                    reward_j = torch.exp(torch.stack(reward_list_j))
                    reward_k = torch.exp(torch.stack(reward_list_k))
                    score = torch.sum(torch.log(reward_j / (reward_j + reward_k)))
                    score_list.append(score)
                max_position = score_list.index(max(score_list))
                training_dataset_list[max_position].append(current_dataset)
                dataset_id_dict[max_position].append(user_id)
                
            print("training_dataset_id_dict", dataset_id_dict)
            cluster1_list = dataset_id_dict[0]
            cluster2_list = dataset_id_dict[1]
            cluster1_num = len([i for i in cluster1_list if i < script_args.cluster1_user])
            print("Group1 on Reward Model 1:", cluster1_num)
            print("Group2 on Reward Model 1:", len(cluster1_list)-cluster1_num)
            cluster2_num = len([i for i in cluster2_list if i < script_args.cluster1_user])
            print("Group1 on Reward Model 2:", cluster2_num)
            print("Group2 on Reward Model 2:", len(cluster2_list)-cluster2_num)
            
            evaluation_dataset_list = {cluster_id: [] for cluster_id in range(num_mixture)}
            eval_dataset_id_dict ={cluster_id: [] for cluster_id in range(num_mixture)}
            for user_id in tqdm(evaluation_user_id):
                current_dataset = user_dataset_map[user_id]
                if j==0:
                    current_index = user_id%num_mixture
                    evaluation_dataset_list[current_index].append(current_dataset)
                    eval_dataset_id_dict[current_index].append(user_id)
                    continue
                current_dataloader = torch.utils.data.DataLoader(current_dataset, batch_size = 1, collate_fn=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=script_args.max_length))
                score_list = []
                for i in range(num_mixture):
                    reward_list_j = []
                    reward_list_k = []
                    for step, batch in enumerate(current_dataloader):
                        inputs = to_device(batch, device)
                        rewards_j = model_list[i](input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"])[0]
                        rewards_k = model_list[i](input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])[0]
                        reward_list_j.append(rewards_j)
                        reward_list_k.append(rewards_k)
                    reward_j = torch.exp(torch.stack(reward_list_j))
                    reward_k = torch.exp(torch.stack(reward_list_k))
                    score = torch.sum(torch.log(reward_j / (reward_j + reward_k)))
                    score_list.append(score)
                max_position = score_list.index(max(score_list))
                evaluation_dataset_list[max_position].append(current_dataset)
                eval_dataset_id_dict[max_position].append(user_id)

            print("eval_dataset_id_dict", eval_dataset_id_dict)
            cluster1_list = eval_dataset_id_dict[0]
            cluster2_list = eval_dataset_id_dict[1]
            cluster1_num = len([i for i in cluster1_list if i < script_args.cluster1_user])
            print("Group1 on Reward Model 1:", cluster1_num)
            print("Group2 on Reward Model 1:", len(cluster1_list)-cluster1_num)
            cluster2_num = len([i for i in cluster2_list if i < script_args.cluster1_user])
            print("Group1 on Reward Model 2:", cluster2_num)
            print("Group2 on Reward Model 2:", len(cluster2_list)-cluster2_num)
            
            for i in range(num_mixture):
                if len(evaluation_dataset_list[i]) == 0:
                    continue
                else:
                    current_dataset = ConcatDataset(evaluation_dataset_list[i])
                    current_dataloader = torch.utils.data.DataLoader(current_dataset, batch_size = 1, collate_fn=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=script_args.max_length))
                prediction_total = 0
                correct_total = 0
                for step, batch in enumerate(current_dataloader):
                    inputs = to_device(batch, device)
                    rewards_j = model_list[i](input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"])[0]
                    rewards_k = model_list[i](input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])[0]
                    prediction_total+=1
                    if rewards_j > rewards_k:
                        correct_total+=1
                print(f"model cluster {i} accuracy:", correct_total/prediction_total)      
        
        for i in range(num_mixture):
            if len(training_dataset_list[i]) == 0:
                continue
            else:
                current_dataset = ConcatDataset(training_dataset_list[i])
                # Train the model
                trainer = RewardTrainer(
                    model=model_list[i],
                    args=training_args,
                    train_dataset=current_dataset,
                    #eval_dataset=eval_dataset,
                    eval_dataset=current_dataset,
                    compute_metrics=compute_metrics,
                    data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=script_args.max_length)
                )
                trainer.train(script_args.resume_from_checkpoint)
                print("Saving last checkpoint of the model")
                model_list[i].save_pretrained(script_args.output_dir + f"peft_{i}_last_checkpoint")
               
    # # Train the model
    # trainer = RewardTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     #eval_dataset=eval_dataset,
    #     eval_dataset=eval_datasets,
    #     compute_metrics=compute_metrics,
    #     data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=script_args.max_length)
    # )

    # trainer.train(script_args.resume_from_checkpoint)

    # print("Saving last checkpoint of the model")
    # model.save_pretrained(script_args.output_dir + "peft_last_checkpoint")

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    set_seed(script_args.seed)
    main(script_args)
