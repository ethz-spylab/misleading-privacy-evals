#!/bin/bash

EXPERIMENT_DIR= # TODO: path to base dir for this experiment
DATA_DIR= # TODO: path to data root
REPO_DIR= # TODO: path to repository (WITHOUT src/)

SEED=0

CANARY_TYPE_ALL=("clean" "label_noise" "ood")
NUM_CANARIES=500
NUM_SHADOW=64
NUM_POISON=0
POISON_TYPE="canary_duplicates"
METHOD="moco"

# Prepare environment
export PYTHONPATH="${PYTHONPATH}:${REPO_DIR}/src"

for canary_type_idx in "${!CANARY_TYPE_ALL[@]}"; do
    CANARY_TYPE="${CANARY_TYPE_ALL[$canary_type_idx]}"
    EXPERIMENT="${CANARY_TYPE}"

    echo "Attacking experiment ${EXPERIMENT}"
    echo "Canaries: ${NUM_CANARIES} (${CANARY_TYPE}), poisons: ${NUM_POISON} (${POISON_TYPE})"

    # white-box attack
    python -u -m experiments.self_supervised --experiment-dir "${EXPERIMENT_DIR}" --experiment "${EXPERIMENT}" --data-dir "${DATA_DIR}" \
        --seed "${SEED}" --num-shadow "${NUM_SHADOW}" \
        --num-canaries "${NUM_CANARIES}" --canary-type "${CANARY_TYPE}" \
        --finetune 0 --score_type "backbone" --method "${METHOD}" \
        --num-poison "${NUM_POISON}" --poison-type "${POISON_TYPE}" \
        attack
        
    # black-box attack, adaptive attack
    python -u -m experiments.self_supervised --experiment-dir "${EXPERIMENT_DIR}" --experiment "${EXPERIMENT}" --data-dir "${DATA_DIR}" \
        --seed "${SEED}" --num-shadow "${NUM_SHADOW}" \
        --num-canaries "${NUM_CANARIES}" --canary-type "${CANARY_TYPE}" \
        --finetune 1 --score_type "similarity" --method "${METHOD}" \
        --num-poison "${NUM_POISON}" --poison-type "${POISON_TYPE}" \
        attack

    # black-box attack, non-adaptive attack
    python -u -m experiments.self_supervised --experiment-dir "${EXPERIMENT_DIR}" --experiment "${EXPERIMENT}" --data-dir "${DATA_DIR}" \
        --seed "${SEED}" --num-shadow "${NUM_SHADOW}" \
        --num-canaries "${NUM_CANARIES}" --canary-type "${CANARY_TYPE}" \
        --finetune 1 --score_type "confidence" --method "${METHOD}" \
        --num-poison "${NUM_POISON}" --poison-type "${POISON_TYPE}" \
        attack
done
