#!/bin/bash

EXPERIMENT_DIR= # TODO: path to base dir for this experiment
DATA_DIR= # TODO: path to data root
REPO_DIR= # TODO: path to repository (WITHOUT src/)

export PYTHONPATH="${PYTHONPATH}:${REPO_DIR}/src"

SEED=0
NUM_POISON=0
POISON_TYPE="random_images"
NUM_CANARIES=500
CANARY_TYPE_ALL=("clean" "label_noise")
NUM_HPS=5
NUM_SHADOW=64
declare -A LOGREG_CS
LOGREG_CS[0,0]="0.31622776601683794"
LOGREG_CS[0,1]="1.0"
LOGREG_CS[0,2]="0.31622776601683794"
LOGREG_CS[0,3]="0.1"
LOGREG_CS[0,4]="1.0"
LOGREG_CS[1,0]="100.0"
LOGREG_CS[1,1]="1.0"
LOGREG_CS[1,2]="31.622776601683793"
LOGREG_CS[1,3]="3.1622776601683795"
LOGREG_CS[1,4]="10.0"

for (( canary_type_idx=0; canary_type_idx<${#CANARY_TYPE_ALL[@]}; canary_type_idx++ )); do
  CANARY_TYPE="${CANARY_TYPE_ALL[$canary_type_idx]}"
  EXPERIMENT="${CANARY_TYPE}"
  for (( hp_idx=0; hp_idx<${NUM_HPS}; hp_idx++ )); do
    LOGREG_C="${LOGREG_CS[$canary_type_idx,$hp_idx]}"
    RUN="train_${hp_idx}"

    echo "Attacking experiment ${EXPERIMENT}, run ${RUN}, with C=${LOGREG_C}"
    echo "Canaries: ${NUM_CANARIES} (${CANARY_TYPE}), poisons: ${NUM_POISON} (${POISON_TYPE})"

    python -u -m experiments.hamp --experiment-dir "${EXPERIMENT_DIR}" --experiment "${EXPERIMENT}" --data-dir "${DATA_DIR}" \
      --seed "${SEED}" --num-shadow "${NUM_SHADOW}" \
      --run-suffix "${RUN}" \
      --num-canaries "${NUM_CANARIES}" --canary-type "${CANARY_TYPE}" \
      --num-poison "${NUM_POISON}" --poison-type "${POISON_TYPE}" \
      attack --logreg-c "${LOGREG_C}"
  done

done
