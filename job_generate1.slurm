#!/bin/bash
#SBATCH --job-name=step_190_generate_baseline    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=2        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G                 # total memory per node (4 GB per cpu-core is default)
##SBATCH --partition=mig
#SBATCH --constraint=gpu80
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:29:59          # total run time limit (HH:MM:SS)

module purge
module load anaconda3/2023.3
module load cudatoolkit/11.7

conda activate psoup2

echo $CUDA_VISIBLE_DEVICES
export PATH_TO_TULU_CKPT="tulu-2-7b"
export WANDB_PROJECT_NAME='robust_rlhf_baseline'
export WANDB_RUN_NAME='generate_answer_baseline'
export WANDB_MODE=offline
export POLICY_MODEL_DIR='' #TODO: change path
export RANK=1
export WORLD_SIZE=1
export LOCAL_RANK=1
export LOCAL_WORLD_SIZE=1
export EVAL_DATASET_NAME='MaxMinRLHF/data/koala_eval_50.json' #TODO: change path

export OUTPUT_DIR="./tulu2output/single_reward_P2A_10_1.json" #TODO: change path
torchrun --nnodes 1 --nproc_per_node 1 eval.py \
    --output_dir $OUTPUT_DIR --base_model $PATH_TO_TULU_CKPT \
    --prompt "Generate a response that is concise and to the point, without being verbose." \
    --batch_size 16 --start_per 0 --end_per 100 \
    --checkpoint_dir $POLICY_MODEL_DIR --dataset_name $EVAL_DATASET_NAME

export OUTPUT_DIR="./tulu2output/single_reward_P2B_10_1.json" #TODO: change path
torchrun --nnodes 1 --nproc_per_node 1 eval.py \
    --output_dir $OUTPUT_DIR --base_model $PATH_TO_TULU_CKPT \
    --prompt "Generate a response that is very informative, without missing any background information." \
    --batch_size 16 --start_per 0 --end_per 100 \
    --checkpoint_dir $POLICY_MODEL_DIR --dataset_name $EVAL_DATASET_NAME
