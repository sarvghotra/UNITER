#!/bin/bash

#SBATCH --job-name=repro_vqa2
#SBATCH --output=job_output_repro_vqa2.txt
#SBATCH --error=job_error_repro_vqa2.txt
#SBATCH --time=11:59:00
#SBATCH --mem=64G
#SBATCH -c 16
#SBATCH --gres=gpu:a100:2
#SBATCH --reservation=DGXA100

# unset CUDA_VISIBLE_DEVICES
module load anaconda/3
conda activate trans_vqa2
cd /home/mila/s/sarvjeet-singh.ghotra/git/UNITER/scripts/

export WANDB_MODE=offline

# unset CUDA_VISIBLE_DEVICES
# CHANGE IT

bash debug_train_vqa.sh


# --evaluate \ ft_repro

echo "Done"

echo "====== Done Sbatch ====="


# module load miniconda/3
# conda activate blip_py37
# module load cuda/11.1
# #SBATCH --reservation=DGXA100
# #SBATCH --nodelist=cn-d[003-004]
#SBATCH --reservation=DGXA100