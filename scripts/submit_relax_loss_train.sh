#!/bin/bash
#SBATCH --array=0-767
#SBATCH --job-name=relax-loss-train
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem-per-cpu=2G
#SBATCH --ntasks=1 --cpus-per-task=8
#SBATCH --time=4:00:00
# Training takes ~2h per model, just 3h in worst case
# Train 4 settings x 3 HPs x 64 shadow models per setting = 768 models

SEED=0

TARGET_LOSS_ALL=("0.5946" "1.2496" "0.3406")
CANARY_TYPE_ALL=("clean" "label_noise")
NUM_POISON_ALL=(0 5000)
NUM_SHADOW=64
# no data augmentation (as in original paper)

NUM_CANARIES=500
POISON_TYPE="random_images"

TASK_IDX=$SLURM_ARRAY_TASK_ID
SHADOW_MODEL_IDX=$((TASK_IDX % NUM_SHADOW))
TASK_IDX=$((TASK_IDX / NUM_SHADOW))
NUM_POISON_IDX=$((TASK_IDX % ${#NUM_POISON_ALL[@]}))
NUM_POISON="${NUM_POISON_ALL[$NUM_POISON_IDX]}"
TASK_IDX=$((TASK_IDX / ${#NUM_POISON_ALL[@]}))
CANARY_TYPE_IDX=$((TASK_IDX % ${#CANARY_TYPE_ALL[@]}))
CANARY_TYPE="${CANARY_TYPE_ALL[$CANARY_TYPE_IDX]}"
TASK_IDX=$((TASK_IDX / ${#CANARY_TYPE_ALL[@]}))
TARGET_LOSS_IDX=$TASK_IDX
TARGET_LOSS="${TARGET_LOSS_ALL[$TARGET_LOSS_IDX]}"

EXPERIMENT="${CANARY_TYPE}_${NUM_POISON}"
RUN="target_loss_${TARGET_LOSS_IDX}"

EXPERIMENT_DIR= # TODO: path to base dir for this experiment
DATA_DIR= # TODO: path to data root
REPO_DIR= # TODO: path to repository (WITHOUT src/)

echo "Running task ${SLURM_ARRAY_TASK_ID} experiment ${EXPERIMENT}, run ${RUN}, shadow model ${SHADOW_MODEL_IDX}"
echo "Target loss: ${TARGET_LOSS}"
echo "Canaries: ${NUM_CANARIES} (${CANARY_TYPE}), poisons: ${NUM_POISON} (${POISON_TYPE})"

# Prepare environment
export PYTHONPATH="${PYTHONPATH}:${REPO_DIR}/src"

python -u -m experiments.relax_loss --experiment-dir "${EXPERIMENT_DIR}" --experiment "${EXPERIMENT}" --data-dir "${DATA_DIR}" \
 --seed "${SEED}" --num-shadow "${NUM_SHADOW}" \
 --run-suffix "${RUN}" \
 --num-canaries "${NUM_CANARIES}" --canary-type "${CANARY_TYPE}" \
 --num-poison "${NUM_POISON}" --poison-type "${POISON_TYPE}" \
 train --shadow-model-idx "${SHADOW_MODEL_IDX}" \
 --target-loss "${TARGET_LOSS}"
