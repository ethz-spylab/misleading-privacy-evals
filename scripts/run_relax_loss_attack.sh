#!/bin/bash

EXPERIMENT_DIR= # TODO: path to base dir for this experiment
DATA_DIR= # TODO: path to data root
REPO_DIR= # TODO: path to repository (WITHOUT src/)

SEED=0
NUM_HPS=3
CANARY_TYPE_ALL=("clean" "label_noise")
NUM_POISON_ALL=(0 5000)
NUM_SHADOW=64
NUM_CANARIES=500
POISON_TYPE="random_images"

# Prepare environment
export PYTHONPATH="${PYTHONPATH}:${REPO_DIR}/src"

for canary_type_idx in "${!CANARY_TYPE_ALL[@]}"; do
  CANARY_TYPE="${CANARY_TYPE_ALL[$canary_type_idx]}"
  for num_poison_idx in "${!NUM_POISON_ALL[@]}"; do
    NUM_POISON="${NUM_POISON_ALL[$num_poison_idx]}"
    EXPERIMENT="${CANARY_TYPE}_${NUM_POISON}"

    for hp_idx in $(seq 0 $((NUM_HPS-1))); do
      TARGET_LOSS_IDX="${hp_idx}"
      RUN="target_loss_${TARGET_LOSS_IDX}"

      echo "Attacking experiment ${EXPERIMENT}, run ${RUN}"
      echo "Canaries: ${NUM_CANARIES} (${CANARY_TYPE}), poisons: ${NUM_POISON} (${POISON_TYPE})"

      python -u -m experiments.relax_loss --experiment-dir "${EXPERIMENT_DIR}" --experiment "${EXPERIMENT}" --data-dir "${DATA_DIR}" \
        --seed "${SEED}" --num-shadow "${NUM_SHADOW}" \
        --run-suffix "${RUN}" \
        --num-canaries "${NUM_CANARIES}" --canary-type "${CANARY_TYPE}" \
        --num-poison "${NUM_POISON}" --poison-type "${POISON_TYPE}" \
        attack

    done
  done
done
