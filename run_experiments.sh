#!/bin/bash
#SBATCH -J "base"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=azm0269@auburn.edu
#SBATCH -N1
#SBATCH --ntasks=1
#SBATCH -D /aiau010_scratch/azm0269/clover
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --time=0-00:15:00
#SBATCH --nodelist=aiau001
#SBATCH --gres=gpu:1
#SBATCH --partition=general

# module load python3

HF_HOME="/aiau010_scratch/azm0269/hub"
env_dir=/aiau010_scratch/azm0269/
cd $env_dir
source /aiau010_scratch/azm0269/clover/.venv/bin/activate
workdir=/aiau010_scratch/azm0269/clover
cd $workdir
source .env
HF_HOME="/aiau010_scratch/azm0269/hub"
HF_TOKEN="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"
HF_HOME="/aiau010_scratch/azm0269/hub"
TMPDIR="/aiau010_scratch/azm0269/tmp"
# CUDA_VISIBLE_DEVICES=6,7
# TOKENIZERS_PARALLELISM=true

# for baseline
nohup srun .venv/bin/python -m main > experiment.log 2>&1
wait

deactivate