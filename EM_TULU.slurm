#!/bin/bash
#SBATCH --job-name=xxxxxx    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=2        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G                 # total memory per node (4 GB per cpu-core is default)
##SBATCH --partition=mig
#SBATCH --constraint=gpu80
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=23:59:59          # total run time limit (HH:MM:SS)

module purge
module load anaconda3/2023.3
module load cudatoolkit/11.7
conda activate #TODO: change to your environment name

export PATH_TO_TULU_CKPT="tulu-7b" #TODO: change model path when necessary
# export PATH_TO_TULU_CKPT="tulu-2-7b"

export datapath1="data/rm_training/P1A.json" #TODO: change path
export datapath2="data/rm_training/P1B.json" #TODO: change path
export OUTPUT_DIR="./checkpoints/training_reward_model_P1_tulu_seed_1104/" #TODO: change path
python test_data_multiple.py --dataset_name1 $datapath1 --dataset_name2 $datapath2 --output_dir $OUTPUT_DIR --train_subset 3000 --num_train_epochs 10 --cluster1_user 40 --cluster2_user 40 --seed 1104 --model_name $PATH_TO_TULU_CKPT


