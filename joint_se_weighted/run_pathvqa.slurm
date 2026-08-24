#!/bin/bash --login

#SBATCH --job-name=pathvqa_joint_se_weighted
#SBATCH --partition=gpuL
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=125G
#SBATCH --time=1-00:00:00
#SBATCH --output=pathvqa_joint_se_weighted_%j.out
#SBATCH --error=pathvqa_joint_se_weighted_%j.err

set -euo pipefail

# --- Project paths -----------------------------------------------------------
readonly PROJECT_DIR="${HOME}/dissertation_2026/joint_se_weighted"
readonly VENV_DIR="${HOME}/dissertation_2026/.venv"
readonly NOTEBOOK_NAME="PathVQA_GPT_LoRA_Joint_SE_Condition_Weights.ipynb"
readonly EXECUTED_NAME="PathVQA_GPT_LoRA_Joint_SE_Condition_Weights_executed_${SLURM_JOB_ID}.ipynb"

export UTILITY_EVAL_BATCH_SIZE="${UTILITY_EVAL_BATCH_SIZE:-32}"

# --- CSF3 software environment ----------------------------------------------
module purge
module load libs/cuda/12.8.1

source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"

# --- Non-interactive notebook settings ---------------------------------------
export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

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
