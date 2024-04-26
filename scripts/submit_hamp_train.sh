#!/bin/bash
#SBATCH --array=0-639
#SBATCH --job-name=hamp-train
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem-per-cpu=2G
#SBATCH --ntasks=1 --cpus-per-task=8
#SBATCH --time=3:30:00
# Training takes a bit more than 1h per model; 3.5h should be very safe
# Train 2 settings x 5 HPs x 64 shadow models = 640 models

SEED=0

NUM_POISON=0
POISON_TYPE="random_images"
NUM_CANARIES=500
CANARY_TYPE_ALL=("clean" "label_noise")
ENTROPY_THRESHOLD_ALL=("0.95" "0.99" "0.99" "0.99" "0.9996837722339832")
REGULARIZATION_STRENGTH_ALL=("0.001" "0.5" "0.05" "1.0" "0.005")
NUM_HPS=${#ENTROPY_THRESHOLD_ALL[@]}
NUM_SHADOW=64

TASK_IDX=$SLURM_ARRAY_TASK_ID
SHADOW_MODEL_IDX=$((TASK_IDX % NUM_SHADOW))
TASK_IDX=$((TASK_IDX / NUM_SHADOW))
HP_IDX=$((TASK_IDX % NUM_HPS))
ENTROPY_THRESHOLD="${ENTROPY_THRESHOLD_ALL[$HP_IDX]}"
REGULARIZATION_STRENGTH="${REGULARIZATION_STRENGTH_ALL[$HP_IDX]}"
TASK_IDX=$((TASK_IDX / NUM_HPS))
CANARY_IDX=$TASK_IDX
CANARY_TYPE="${CANARY_TYPE_ALL[$CANARY_IDX]}"

EXPERIMENT="${CANARY_TYPE}"
RUN="train_${HP_IDX}"

EXPERIMENT_DIR= # TODO: path to base dir for this experiment
DATA_DIR= # TODO: path to data root
REPO_DIR= # TODO: path to repository (WITHOUT src/)

echo "Running task ${SLURM_ARRAY_TASK_ID} experiment ${EXPERIMENT}, run ${RUN}, shadow model ${SHADOW_MODEL_IDX}"
echo "Entropy threshold: ${ENTROPY_THRESHOLD}, regularization strength: ${REGULARIZATION_STRENGTH}"
echo "Canaries: ${NUM_CANARIES} (${CANARY_TYPE}), poisons: ${NUM_POISON} (${POISON_TYPE})"

# Prepare environment
export PYTHONPATH="${PYTHONPATH}:${REPO_DIR}/src"

python -u -m experiments.hamp --experiment-dir "${EXPERIMENT_DIR}" --experiment "${EXPERIMENT}" --data-dir "${DATA_DIR}" \
 --seed "${SEED}" --num-shadow "${NUM_SHADOW}" \
 --run-suffix "${RUN}" \
 --num-canaries "${NUM_CANARIES}" --canary-type "${CANARY_TYPE}" \
 --num-poison "${NUM_POISON}" --poison-type "${POISON_TYPE}" \
 train --shadow-model-idx "${SHADOW_MODEL_IDX}" \
 --entropy-threshold "${ENTROPY_THRESHOLD}" --regularization-strength "${REGULARIZATION_STRENGTH}"
