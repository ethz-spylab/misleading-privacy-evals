#!/bin/bash
#SBATCH --array=0-255
#SBATCH --job-name=selena-distillation
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem-per-cpu=2G
#SBATCH --ntasks=1 --cpus-per-task=8
#SBATCH --time=01:00:00
# Takes ~35min per teacher, but request 1h to be safe
# Train 64 shadow models x 4 settings = 256 models

SEED=0

NUM_TEACHER=25
NUM_QUERY=10

EXPERIMENT_ALL=("clean" "label_noise" "duplicates" "both")
NUM_POISON_ALL=(0 0 500 500)
POISON_TYPE="canary_duplicates"
NUM_CANARIES=500
CANARY_TYPE_ALL=("clean" "label_noise" "clean" "label_noise")
NUM_SHADOW=64

TASK_IDX=$SLURM_ARRAY_TASK_ID
SHADOW_MODEL_IDX=$((TASK_IDX % NUM_SHADOW))
TASK_IDX=$((TASK_IDX / NUM_SHADOW))
EXPERIMENT_IDX=$TASK_IDX

EXPERIMENT="${EXPERIMENT_ALL[$EXPERIMENT_IDX]}"
NUM_POISON="${NUM_POISON_ALL[$EXPERIMENT_IDX]}"
CANARY_TYPE="${CANARY_TYPE_ALL[$EXPERIMENT_IDX]}"

EXPERIMENT_DIR= # TODO: path to base dir for this experiment
DATA_DIR= # TODO: path to data root
REPO_DIR= # TODO: path to repository (WITHOUT src/)

echo "Running task ${SLURM_ARRAY_TASK_ID} experiment ${EXPERIMENT}, shadow model ${SHADOW_MODEL_IDX}"
echo "Canaries: ${NUM_CANARIES} (${CANARY_TYPE}), poisons: ${NUM_POISON} (${POISON_TYPE})"

# Prepare environment
export PYTHONPATH="${PYTHONPATH}:${REPO_DIR}/src"

python -u -m experiments.selena --experiment-dir "${EXPERIMENT_DIR}" --experiment "${EXPERIMENT}" --data-dir "${DATA_DIR}" \
 --seed "${SEED}" --num-shadow "${NUM_SHADOW}" \
 --num-teachers "${NUM_TEACHER}" --num-query "${NUM_QUERY}" \
 --num-canaries "${NUM_CANARIES}" --canary-type "${CANARY_TYPE}" \
 --num-poison "${NUM_POISON}" --poison-type "${POISON_TYPE}" \
 train --shadow-model-idx "${SHADOW_MODEL_IDX}" \
 distillation
