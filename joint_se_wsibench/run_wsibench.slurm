#!/bin/bash --login

#SBATCH --job-name=wsibench_joint_se
#SBATCH --partition=gpuL
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=125G
#SBATCH --time=1-00:00:00
#SBATCH --output=wsibench_joint_se_%j.out
#SBATCH --error=wsibench_joint_se_%j.err

set -euo pipefail

# --- Project paths -----------------------------------------------------------
readonly PROJECT_DIR="${HOME}/dissertation_2026/joint_se_wsibench"
readonly VENV_DIR="${HOME}/dissertation_2026/.venv"
readonly NOTEBOOK_NAME="WSIBench_GPT_LoRA_Joint_SE.ipynb"
readonly EXECUTED_NAME="WSIBench_GPT_LoRA_Joint_SE_executed_${SLURM_JOB_ID}.ipynb"

# --- CSF3 software environment ----------------------------------------------
module purge
module load libs/cuda/12.8.1

source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"

# --- Non-interactive notebook settings ---------------------------------------
export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export WSIBENCH_FEATURE_ROOT="${PROJECT_DIR}/data/WSI-Bench/features"
export WSIBENCH_ANNOTATION_DIR="${PROJECT_DIR}/data/WSI-Bench/annotations"
export WSIBENCH_TEST_MODE="${WSIBENCH_TEST_MODE:-both}"

export REUSE_TRAINED_CHECKPOINT="${REUSE_TRAINED_CHECKPOINT:-1}"
export JSE_GENERATION_BATCH_SIZE="${JSE_GENERATION_BATCH_SIZE:-5}"

# --- Job diagnostics ---------------------------------------------------------
echo "======================================================================"
echo "Started: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Virtual environment: ${VIRTUAL_ENV}"
echo "Python: $(which python)"
python --version
echo "Visible GPU IDs: ${CUDA_VISIBLE_DEVICES:-not-set}"
nvidia-smi
echo "======================================================================"

# --- Execute every notebook cell and save outputs to a new notebook ----------
jupyter nbconvert \
    --to notebook \
    --execute "${NOTEBOOK_NAME}" \
    --output "${EXECUTED_NAME}" \
    --ExecutePreprocessor.timeout=-1

# --- Completion summary ------------------------------------------------------
echo "======================================================================"
echo "Finished: $(date)"
echo "Executed notebook: ${PROJECT_DIR}/${EXECUTED_NAME}"
echo "======================================================================"
