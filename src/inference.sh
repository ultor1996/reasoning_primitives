#!/bin/bash -l
# =============================================================================
#  inference.sh  —  SLURM job script for running inference on Marvin
#
#  Submit with:
#    sbatch inference.sh \
#        --input  /path/to/data.json \
#        --model  allenai/Olmo-3-7B-Think \
#        --output /path/to/results.json
#
#  All extra CLI args after the script name are forwarded to inference.py.
# =============================================================================

#SBATCH --account=ag_bit_flek
#SBATCH --partition=mlgpu_devel
#SBATCH --job-name=inference
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --oversubscribe

#############################################
# Working Directory Setup
#############################################

PROJECT_DIR="/lustre/mlnvme/data/srawat_hpc-reasoning_primitivs/reasoning_primitives"
VENV_DIR="/lustre/mlnvme/data/srawat_hpc-reasoning_primitivs/.venv_amd"
HF_CACHE="/lustre/mlnvme/data/srawat_hpc-reasoning_primitivs/.cache/huggingface"

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/results"
ulimit -c 0

out="$PROJECT_DIR/logs/inference-out.$SLURM_JOB_ID"
err="$PROJECT_DIR/logs/inference-err.$SLURM_JOB_ID"

#############################################
# Environment Setup
#############################################

export MODULEPATH=/opt/software/easybuild-AMD/modules/all:/etc/modulefiles:/usr/share/modulefiles:/opt/software/modulefiles:/usr/share/modulefiles/Linux:/usr/share/modulefiles/Core:/usr/share/lmod/lmod/modulefiles/Core

module purge
module load CUDA/12.6.0 Python/3.12.3-GCCcore-13.3.0

# Create venv on first run — comment out once installed
# python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

#  Install dependencies on first run — comment out once installed
# pip3 install --upgrade pip --no-cache-dir
# pip3 install torch==2.8.0 --no-cache-dir
# pip3 install transformers accelerate --no-cache-dir
# pip3 install vllm --no-cache-dir
# pip3 install json-repair --no-cache-dir

#############################################
# Environment Variables
#############################################

export CUDA_VISIBLE_DEVICES="0"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export HF_HOME="$HF_CACHE"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"

#############################################
# Job Info
#############################################

echo "# [${SLURM_JOB_ID}] Job started at: $(date)" > "$out"
echo "# [${SLURM_JOB_ID}] Running on node: $(hostname)" >> "$out"
echo "# [${SLURM_JOB_ID}] GPU: $CUDA_VISIBLE_DEVICES" >> "$out"
echo "# Python: $(which python3)" >> "$out"

#############################################
# Run Inference
#############################################

cd "$PROJECT_DIR/src"

echo "" >> "$out"
echo "=== Starting inference ===" >> "$out"

python3 inference.py "$@" 1>>"$out" 2>>"$err"

echo "# [${SLURM_JOB_ID}] Job finished at: $(date)" >> "$out"