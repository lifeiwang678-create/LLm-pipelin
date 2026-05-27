#!/bin/bash
#SBATCH -p a100
#SBATCH -J prep_hhar_dreamt
#SBATCH -o /home/users/grad/2025/25t9801/logs/preprocess_hhar_dreamt_%j.out
#SBATCH -e /home/users/grad/2025/25t9801/logs/preprocess_hhar_dreamt_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=4

set -euo pipefail

BASE=${BASE:-/home/users/grad/2025/25t9801/projects/LLm-pipelin/experiment_4x3x2}
LOGROOT=${LOGROOT:-/home/users/grad/2025/25t9801/logs/preprocess_hhar_dreamt_$(date +%Y%m%d%H%M%S)}
PROCESSED_DIR=${PROCESSED_DIR:-Processed}

mkdir -p "$LOGROOT"
cd "$BASE"

export MPLCONFIGDIR="${LOGROOT}/mplconfig"
mkdir -p "$MPLCONFIGDIR"

echo "Job started at $(date)"
echo "Base: $BASE"
echo "Log dir: $LOGROOT"
echo "Processed dir: $PROCESSED_DIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

source .venv/bin/activate

echo "=================================================="
echo "Preprocessing HHAR dataset windows"
.venv/bin/python preprocess_datasets.py \
  -dataset HHAR \
  --processed-dir "$PROCESSED_DIR" \
  --overwrite \
  2>&1 | tee "$LOGROOT/hhar_dataset.log"

echo "=================================================="
echo "Precomputing HHAR input caches"
.venv/bin/python preprocess_inputs.py \
  -dataset HHAR \
  -Input all \
  --processed-dir "$PROCESSED_DIR" \
  --input-cache-dir "$PROCESSED_DIR" \
  --overwrite \
  2>&1 | tee "$LOGROOT/hhar_inputs.log"

echo "=================================================="
echo "Discovering DREAMT subjects"
mapfile -t DREAMT_SUBJECTS < <(
  find Dataset/DREAMT/data_64Hz -maxdepth 1 -type f -name '*_whole_df.csv' \
    | sed -E 's#.*/(S[0-9]+)_.*#\1#' \
    | sort
)
echo "DREAMT subjects: ${#DREAMT_SUBJECTS[@]}"
printf '%s\n' "${DREAMT_SUBJECTS[@]}" > "$LOGROOT/dreamt_subjects.txt"

echo "=================================================="
echo "Preprocessing DREAMT dataset windows by subject shard"
.venv/bin/python preprocess_datasets.py \
  -dataset DREAMT \
  --processed-dir "$PROCESSED_DIR" \
  --shard-by-subject \
  --subjects "${DREAMT_SUBJECTS[@]}" \
  --overwrite \
  2>&1 | tee "$LOGROOT/dreamt_dataset.log"

echo "=================================================="
echo "Precomputing DREAMT input caches"
.venv/bin/python preprocess_inputs.py \
  -dataset DREAMT \
  -Input all \
  --processed-dir "$PROCESSED_DIR" \
  --input-cache-dir "$PROCESSED_DIR" \
  --overwrite \
  2>&1 | tee "$LOGROOT/dreamt_inputs.log"

echo "=================================================="
echo "Generated caches:"
find "$PROCESSED_DIR" -maxdepth 1 -type f \
  \( -name 'HHAR_*' -o -name 'DREAMT_*' \) \
  -printf '%p\t%s bytes\n' \
  | sort \
  | tee "$LOGROOT/generated_caches.txt"

echo "Preprocessing finished at $(date)"
