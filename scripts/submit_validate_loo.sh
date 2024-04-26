#!/bin/bash
#SBATCH --array=0-319
#SBATCH --job-name=validate-loo-train
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem-per-cpu=2G
#SBATCH --tmp=4G
#SBATCH --ntasks=1 --cpus-per-task=8
#SBATCH --time=1:30:00
# Training should take <<1h
# Train 5 settings x 64 shadow models per setting = 320 models

NUM_SHADOW=64
NUM_POISON=0

EXPERIMENT_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SHADOW))
SHADOW_MODEL_IDX=$((SLURM_ARRAY_TASK_ID % NUM_SHADOW))

EXPERIMENTS=("fixed_halves_true" "fixed_halves_false" "fixed_halves_true_500" "fixed_halves_false_500" "fixed_halves_false_500_label_noise")
EXPERIMENT="${EXPERIMENTS[$EXPERIMENT_IDX]}"

FIXED_HALVES_ALL=("true" "false" "true" "false" "false")
FIXED_HALVES="${FIXED_HALVES_ALL[$EXPERIMENT_IDX]}"
NUM_CANARIES_ALL=(250 250 500 500 500)
NUM_CANARIES="${NUM_CANARIES_ALL[$EXPERIMENT_IDX]}"
CANARY_TYPE_ALL=("clean" "clean" "clean" "clean" "label_noise")
CANARY_TYPE="${CANARY_TYPE_ALL[$EXPERIMENT_IDX]}"

EXPERIMENT_DIR= # TODO: path to base dir for this experiment
DATA_DIR= # TODO: path to data root
REPO_DIR= # TODO: path to repository (WITHOUT src/)

echo "Running task ID ${SLURM_ARRAY_TASK_ID} for experiment ${EXPERIMENT}, shadow model ${SHADOW_MODEL_IDX}"
echo "Fixed halves ${FIXED_HALVES}, canary type ${CANARY_TYPE}, num canaries ${NUM_CANARIES}"

# Prepare environment
export PYTHONPATH="${PYTHONPATH}:${REPO_DIR}/src"

python -u -m experiments.validate_loo --experiment-dir "${EXPERIMENT_DIR}" --experiment "${EXPERIMENT}" --data-dir "${DATA_DIR}" \
 --seed 0 --num-shadow "${NUM_SHADOW}" \
 --num-canaries "${NUM_CANARIES}" --canary-type "${CANARY_TYPE}" --num-poison "${NUM_POISON}" \
 --fixed-halves "${FIXED_HALVES}" \
 train --shadow-model-idx "${SHADOW_MODEL_IDX}"
