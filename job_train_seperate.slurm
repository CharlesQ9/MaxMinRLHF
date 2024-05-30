#!/bin/bash
#SBATCH --job-name=xxx    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=2        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G                 # total memory per node (4 GB per cpu-core is default)
##SBATCH --partition=mig
#SBATCH --constraint=gpu80
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)

module purge
module load anaconda3/2023.3
module load cudatoolkit/11.7
conda activate #TODO: your environment

# P1A,P1B,P2A,P2B,P3A,P3B
export OUTPUT_DIR="./checkpoints/tulu2_P1A_Epoch1" #TODO: change path
export PATH_TO_TULU_CKPT="tulu-2-7b" #TODO:can be changed to llama2 7B
export PATH_TO_RM_DATA='MaxMinRLHF/data/rm_training/P1A.json' #TODO: change path
export EVAL_DATASET_NAME='MaxMinRLHF/data/rm_training/P1A.json' #TODO: change path

python training_reward_model.py \
    --model_name $PATH_TO_TULU_CKPT \
    --dataset_name $PATH_TO_RM_DATA \
    --eval_dataset_name $EVAL_DATASET_NAME \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size 1 \
    --num_train_epochs 1 

