#!/bin/bash
#SBATCH --output=/dev/null
#SBATCH --job-name=generic_cellvit_inference_runner
#SBATCH -c 1                     # Request 1 CPU cores
#SBATCH -t 02:00:00              # 2 hours max runtime

# === Script name + datetime ===
SCRIPT_NAME=$(basename "$1" .py)
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_DIR="./output/runner_logs"
LOG_FILE="$LOG_DIR/${SCRIPT_NAME}_${TIMESTAMP}.out"

# === Create log dir if needed ===
mkdir -p "$LOG_DIR"

# === Redirect output manually ===
exec > "$LOG_FILE" 2>&1

# === Activate your environment ===
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate /appdata/users/P098864/CellVitPlusPlus

# === Run script with all args ===
echo "Running: python $@"
python -u "$@"
