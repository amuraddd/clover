#!/bin/bash
#SBATCH -J "base"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=azm0269@auburn.edu
#SBATCH -N1
#SBATCH --ntasks=1
#SBATCH -D /aiau010_scratch/azm0269/clover
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --time=3-24:00:00
#SBATCH --nodelist=aiau001
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
# TOKENIZERS_PARALLELISM=true

# Reduce allocator fragmentation during long rollout/update cycles.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# for baseline
srun --ntasks=1 .venv/bin/python -m main > experiment_1e_5_20.log 2>&1

deactivate
