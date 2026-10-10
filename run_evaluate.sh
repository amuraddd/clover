#!/bin/bash
#SBATCH -J "eval"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=azm0269@auburn.edu
#SBATCH -N1
#SBATCH --ntasks=1
#SBATCH -D /aiau010_scratch/azm0269/clover
#SBATCH --output=evaluate_output.txt
#SBATCH --error=evaluate_error.txt
#SBATCH --time=3-24:00:00
#SBATCH --nodelist=aiau001
#SBATCH --gres=gpu:1
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
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export HF_HOME="/aiau010_scratch/azm0269/hub"
export TMPDIR="/aiau010_scratch/azm0269/tmp"
# TOKENIZERS_PARALLELISM=true
# export CUDA_VISIBLE_DEVICES=1,2,3

# Reduce allocator fragmentation during long rollout/update cycles.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Missing CompBench runtime/weights are installed automatically before generation.
# Setup requires a CUDA 12.4 toolkit for the Detectron2 build.
# Pass --compbench-install-runtime to rerun setup for an existing installation.
# Usage: sbatch run_evaluate.sh --baselines sd15 emo_v5 --b2-run-name b2_eval
srun --ntasks=1 .venv/bin/python -m evaluate "$@" > evaluate.log 2>&1
evaluation_status=$?

deactivate
exit "$evaluation_status"
