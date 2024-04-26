#!/bin/bash

EXPERIMENT_DIR= # TODO: path to base dir for this experiment
DATA_DIR= # TODO: path to data root
REPO_DIR= # TODO: path to repository (WITHOUT src/)

SEED=0

NUM_SHADOW=64
NUM_POISON=0
POISON_TYPE="canary_duplicates"

EXPERIMENTS=("fixed_halves_true" "fixed_halves_false" "fixed_halves_true_500" "fixed_halves_false_500" "fixed_halves_false_500_label_noise")

FIXED_HALVES_ALL=("true" "false" "true" "false" "false")
NUM_CANARIES_ALL=(250 250 500 500 500)
CANARY_TYPE_ALL=("clean" "clean" "clean" "clean" "label_noise")

# Prepare environment
export PYTHONPATH="${PYTHONPATH}:${REPO_DIR}/src"

for experiment_idx in "${!EXPERIMENTS[@]}"; do
  EXPERIMENT="${EXPERIMENTS[$experiment_idx]}"
  FIXED_HALVES="${FIXED_HALVES_ALL[$experiment_idx]}"
  NUM_CANARIES="${NUM_CANARIES_ALL[$experiment_idx]}"
  CANARY_TYPE="${CANARY_TYPE_ALL[$experiment_idx]}"

  echo "Attacking experiment ${EXPERIMENT}"
  echo "Canaries: ${NUM_CANARIES} (${CANARY_TYPE}), poisons: ${NUM_POISON} (${POISON_TYPE})"

  python -u -m experiments.validate_loo --experiment-dir "${EXPERIMENT_DIR}" --experiment "${EXPERIMENT}" --data-dir "${DATA_DIR}" \
   --seed "${SEED}" --num-shadow "${NUM_SHADOW}" \
   --num-canaries "${NUM_CANARIES}" --canary-type "${CANARY_TYPE}" \
   --num-poison "${NUM_POISON}" --poison-type "${POISON_TYPE}" \
   --fixed-halves "${FIXED_HALVES}" \
    attack

done
