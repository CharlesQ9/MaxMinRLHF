#!/bin/bash
#SBATCH --job-name=xxxxxx    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=2        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G                 # total memory per node (4 GB per cpu-core is default)
##SBATCH --partition=pli
#SBATCH --constraint=gpu80
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:29:59          # total run time limit (HH:MM:SS)

module purge
module load anaconda3/2023.3
module load cudatoolkit/11.7
conda activate #TODO: change to your environment name

echo $CUDA_VISIBLE_DEVICES
export PATH_TO_LLAMA_CKPT="MaxMinRLHF/Llama-2-7b-hf" #TODO: change model path
export PATH_TO_TOKENIZER_CKPT="MaxMinRLHF/Llama-2-7b-hf" #TODO: change model path
export EVAL_DATASET_NAME='MaxMinRLHF/data/koala_eval_50.json'
export POLICY_MODEL_DIR='' #TODO: change path

export OUTPUT_DIR="./llama2_output_new/llama2_P1_maxminA.json" #TODO: change path
python3 simple_generate_llama2.py --output_dir $OUTPUT_DIR --base_model $PATH_TO_LLAMA_CKPT --dataset_name $EVAL_DATASET_NAME --tokenizer $PATH_TO_TOKENIZER_CKPT --subset 50 --checkpoint_dir $POLICY_MODEL_DIR --prompt "Generate a response that can be easily understood by an elementary school student."

export OUTPUT_DIR="./llama2_output_new/llama2_P1_maxminB.json" #TODO: change path
python3 simple_generate_llama2.py --output_dir $OUTPUT_DIR --base_model $PATH_TO_LLAMA_CKPT --dataset_name $EVAL_DATASET_NAME --tokenizer $PATH_TO_TOKENIZER_CKPT --subset 50 --checkpoint_dir $POLICY_MODEL_DIR --prompt "Generate a response that only a PhD Student in that specific field could understand."
