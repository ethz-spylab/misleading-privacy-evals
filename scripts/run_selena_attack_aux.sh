#!/bin/bash

EXPERIMENT_DIR= # TODO: path to base dir for this experiment
DATA_DIR= # TODO: path to data root
REPO_DIR= # TODO: path to repository (WITHOUT src/)
EMBEDDINGS_FILE= # TODO: path to embeddings.pt file, generated using notebooks/generate_selena_similarities.ipynb

SEED=0

NUM_TEACHER=25
NUM_QUERY=10

EXPERIMENT_ALL=("duplicates_mislabel_half" "duplicates_mislabel_full")
NUM_POISON=0
POISON_TYPE="canary_duplicates"
NUM_CANARIES=500
CANARY_TYPE_ALL=("duplicates_mislabel_half" "duplicates_mislabel_full")
NUM_SHADOW=64

# Prepare environment
export PYTHONPATH="${PYTHONPATH}:${REPO_DIR}/src"

for experiment_idx in "${!EXPERIMENT_ALL[@]}"; do
  EXPERIMENT="${EXPERIMENT_ALL[$experiment_idx]}"
  CANARY_TYPE="${CANARY_TYPE_ALL[$experiment_idx]}"

  echo "Attacking experiment ${EXPERIMENT}"
  echo "Canaries: ${NUM_CANARIES} (${CANARY_TYPE}), poisons: ${NUM_POISON} (${POISON_TYPE})"

  python -u -m experiments.selena --experiment-dir "${EXPERIMENT_DIR}" --experiment "${EXPERIMENT}" --data-dir "${DATA_DIR}" \
    --seed "${SEED}" --num-shadow "${NUM_SHADOW}" \
    --num-teachers "${NUM_TEACHER}" --num-query "${NUM_QUERY}" \
    --num-canaries "${NUM_CANARIES}" --canary-type "${CANARY_TYPE}" \
    --num-poison "${NUM_POISON}" --poison-type "${POISON_TYPE}" \
    attack --embeddings-file "${EMBEDDINGS_FILE}"

done
