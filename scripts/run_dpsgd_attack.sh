#!/bin/bash

EXPERIMENT_DIR= # TODO: path to base dir for this experiment
DATA_DIR= # TODO: path to data root
REPO_DIR= # TODO: path to repository (WITHOUT src/)

SEED=0
NUM_SHADOW=64

NUM_CANARIES=500
NUM_POISON=0
POISON_TYPE="canary_duplicates_noisy"

BASELINE_TYPE_ALL=("medium" "high" "veryhigh")
CANARY_TYPE_ALL=("clean" "label_noise" "blank_images" "ood")

# Prepare environment
export PYTHONPATH="${PYTHONPATH}:${REPO_DIR}/src"

for baseline_idx in "${!BASELINE_TYPE_ALL[@]}"; do
  for canary_type_idx in "${!CANARY_TYPE_ALL[@]}"; do
    BASELINE_TYPE="${BASELINE_TYPE_ALL[$baseline_idx]}"
    CANARY_TYPE="${CANARY_TYPE_ALL[$canary_type_idx]}"
    EXPERIMENT="${BASELINE_TYPE}_${CANARY_TYPE}"

    echo "Attacking experiment ${EXPERIMENT}"

    python -u -m experiments.dpsgd --experiment-dir "${EXPERIMENT_DIR}" --experiment "${EXPERIMENT}" --data-dir "${DATA_DIR}" \
      --seed "${SEED}" --num-shadow "${NUM_SHADOW}" \
      --num-canaries "${NUM_CANARIES}" --canary-type "${CANARY_TYPE}" \
      --num-poison "${NUM_POISON}" --poison-type "${POISON_TYPE}" \
      attack
  done
done
