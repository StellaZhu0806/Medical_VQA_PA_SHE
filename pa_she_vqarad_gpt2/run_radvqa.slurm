#!/bin/bash --login

#SBATCH --job-name=pa_she_vqarad_gpt2
#SBATCH --partition=gpuL
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=125G
#SBATCH --time=1-00:00:00
#SBATCH --output=pa_she_vqarad_gpt2_%j.out
#SBATCH --error=pa_she_vqarad_gpt2_%j.err

set -euo pipefail

# --- Project paths -----------------------------------------------------------
readonly PROJECT_DIR="${HOME}/dissertation_2026/pa_she_vqarad_gpt2"
readonly VENV_DIR="${HOME}/dissertation_2026/.venv"
readonly NOTEBOOK_NAME="VQARAD_GPT_LoRA_PA_SHE.ipynb"
readonly EXECUTED_NAME="VQARAD_GPT_LoRA_PA_SHE_executed_${SLURM_JOB_ID}.ipynb"

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
