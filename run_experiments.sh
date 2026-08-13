#!/bin/bash
#SBATCH -J "base"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=azm0269@auburn.edu
#SBATCH -N1
#SBATCH --ntasks=1
#SBATCH -D /aiau010_scratch/azm0269/clover
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --time=1-20:00:00
#SBATCH --nodelist=aiau011
#SBATCH --gres=gpu:2
#SBATCH --partition=general

# module load python3
export TORCH_HOME=/aiau010_scratch/azm0269/clover/.cache/torch
echo $TORCH_HOME
export HF_HOME="/aiau010_scratch/azm0269/hub"
export env_dir=/aiau010_scratch/azm0269/
cd $env_dir
source /aiau010_scratch/azm0269/clover/.venv/bin/activate
export workdir=/aiau010_scratch/azm0269/clover
cd $workdir
source .env
export HF_HOME="/aiau010_scratch/azm0269/hub"
export TMPDIR="/aiau010_scratch/azm0269/tmp"
# CUDA_VISIBLE_DEVICES=6,7
# TOKENIZERS_PARALLELISM=true

# for baseline
nohup srun .venv/bin/python -m main > experiment.log 2>&1
wait

deactivate