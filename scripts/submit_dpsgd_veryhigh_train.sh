#!/bin/bash
#SBATCH --array=0-255
#SBATCH --job-name=dpsgd-veryhigh-train
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem-per-cpu=2G
#SBATCH --ntasks=1 --cpus-per-task=8
#SBATCH --time=4:00:00
# Runs typically take at most ~2h
# Train on 4 canary types x 64 shadow models per setting = 256 models

SEED=0
NUM_SHADOW=64

# Shared HPs
AUGMULT_FACTOR="8"
LEARNING_RATE="4.0"
MAX_GRAD_NORM="1.0"

# Specific HPs
BATCH_SIZE="64"
NUM_EPOCHS="25"
NOISE_MULTIPLIER="0.003125"

NUM_CANARIES=500
NUM_POISON=0
POISON_TYPE="canary_duplicates_noisy"

CANARY_TYPE_ALL=("clean" "label_noise" "blank_images" "ood")
CANARY_TYPE_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SHADOW))
SHADOW_MODEL_IDX=$((SLURM_ARRAY_TASK_ID % NUM_SHADOW))
CANARY_TYPE="${CANARY_TYPE_ALL[$CANARY_TYPE_IDX]}"

EXPERIMENT="veryhigh_${CANARY_TYPE}"

EXPERIMENT_DIR= # TODO: path to base dir for this experiment
DATA_DIR= # TODO: path to data root
REPO_DIR= # TODO: path to repository (WITHOUT src/)

echo "Running task ID ${SLURM_ARRAY_TASK_ID} for experiment ${EXPERIMENT}, shadow model ${SHADOW_MODEL_IDX}"
echo "Canaries: ${NUM_CANARIES} (${CANARY_TYPE}), poisons: ${NUM_POISON} (${POISON_TYPE})"

# Prepare environment
export PYTHONPATH="${PYTHONPATH}:${REPO_DIR}/src"

python -u -m experiments.dpsgd --experiment-dir "${EXPERIMENT_DIR}" --experiment "${EXPERIMENT}" --data-dir "${DATA_DIR}" \
 --seed "${SEED}" --num-shadow "${NUM_SHADOW}" \
 --num-canaries "${NUM_CANARIES}" --canary-type "${CANARY_TYPE}" \
 --num-poison "${NUM_POISON}" --poison-type "${POISON_TYPE}" \
 train --shadow-model-idx "${SHADOW_MODEL_IDX}" \
 --augmult-factor "${AUGMULT_FACTOR}" --learning-rate "${LEARNING_RATE}" --max-grad-norm "${MAX_GRAD_NORM}" \
 --batch-size "${BATCH_SIZE}" --noise-multiplier "${NOISE_MULTIPLIER}" --num-epochs "${NUM_EPOCHS}"
