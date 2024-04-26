#!/bin/bash
#SBATCH --array=0-127
#SBATCH --job-name=DFKD
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem-per-cpu=2G
#SBATCH --ntasks=1 --cpus-per-task=8
#SBATCH --time=12:00:00
# Train 2 settings x 64 shadow models per setting = 128 models

SEED=0

CANARY_TYPE_ALL=("clean" "label_noise")
NUM_CANARIES=500
NUM_SHADOW=64
NUM_POISON=0
POISON_TYPE="canary_duplicates"

TASK_IDX=$SLURM_ARRAY_TASK_ID
SHADOW_MODEL_IDX=$((TASK_IDX % NUM_SHADOW))
TASK_IDX=$((TASK_IDX / NUM_SHADOW))
CANARY_TYPE_IDX=$TASK_IDX
CANARY_TYPE="${CANARY_TYPE_ALL[$CANARY_TYPE_IDX]}"

EXPERIMENT="${CANARY_TYPE}"

EXPERIMENT_DIR= # TODO: path to base dir for this experiment
DATA_DIR= # TODO: path to data root
REPO_DIR= # TODO: path to repository (WITHOUT src/)

echo "Running task ${SLURM_ARRAY_TASK_ID} experiment ${EXPERIMENT}, shadow model ${SHADOW_MODEL_IDX}"
echo "Canaries: ${NUM_CANARIES} (${CANARY_TYPE}), poisons: ${NUM_POISON} (${POISON_TYPE})"

# Prepare environment
export PYTHONPATH="${PYTHONPATH}:${REPO_DIR}/src"

python -u -m experiments.dfkd --experiment-dir "${EXPERIMENT_DIR}" --experiment "${EXPERIMENT}" --data-dir "${DATA_DIR}" \
    --seed "${SEED}" --num-shadow "${NUM_SHADOW}" --teacher_dir "${TEACHER_DIR}" \
    --num-canaries "${NUM_CANARIES}" --canary-type "${CANARY_TYPE}" \
    --num-poison "${NUM_POISON}" --poison-type "${POISON_TYPE}" \
    train  --shadow-model-idx "${SHADOW_MODEL_IDX}"
