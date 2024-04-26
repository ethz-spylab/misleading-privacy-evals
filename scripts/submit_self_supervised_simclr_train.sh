#!/bin/bash
#SBATCH --array=0-191
#SBATCH --job-name=SSL
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem-per-cpu=2G
#SBATCH --ntasks=1 --cpus-per-task=8
# Train 3 settings x 64 shadow models per setting = 192 models

SEED=0

CANARY_TYPE_ALL=("clean" "label_noise" "ood")
NUM_CANARIES=500
NUM_SHADOW=64
NUM_POISON=0
POISON_TYPE="canary_duplicates"
METHOD="simclr"

TASK_IDX=$SLURM_ARRAY_TASK_ID
SHADOW_MODEL_IDX=$((TASK_IDX % NUM_SHADOW))
TASK_IDX=$((TASK_IDX / NUM_SHADOW))
CANARY_TYPE_IDX=$TASK_IDX
CANARY_TYPE="${CANARY_TYPE_ALL[$CANARY_TYPE_IDX]}"

EXPERIMENT="${CANARY_TYPE}"

EXPERIMENT_DIR= # TODO: path to base dir for this experiment
DATA_DIR= # TODO: path to data root
REPO_DIR= # TODO: path to repository (WITHOUT src/)

echo "Running task ${SLURM_ARRAY_TASK_ID} experiment ${EXPERIMENT}, shadow model ${SHADOW_MODEL_IDX}"
echo "Canaries: ${NUM_CANARIES} (${CANARY_TYPE}), poisons: ${NUM_POISON} (${POISON_TYPE})"

# Prepare environment
export PYTHONPATH="${PYTHONPATH}:${REPO_DIR}/src"

# pretrain the shadow models, adaptive attack
python -u -m experiments.self_supervised --experiment-dir "${EXPERIMENT_DIR}" --experiment "${EXPERIMENT}" --data-dir "${DATA_DIR}" \
    --seed "${SEED}" --num-shadow "${NUM_SHADOW}" \
    --num-canaries "${NUM_CANARIES}" --canary-type "${CANARY_TYPE}" \
    --finetune 0 --score_type "backbone" --method "${METHOD}" \
    --num-poison "${NUM_POISON}" --poison-type "${POISON_TYPE}" \
    train --shadow-model-idx "${SHADOW_MODEL_IDX}"

# finetune the shadow models, adaptive attack 
python -u -m experiments.self_supervised --experiment-dir "${EXPERIMENT_DIR}" --experiment "${EXPERIMENT}" --data-dir "${DATA_DIR}" \
    --seed "${SEED}" --num-shadow "${NUM_SHADOW}" \
    --num-canaries "${NUM_CANARIES}" --canary-type "${CANARY_TYPE}" \
    --finetune 1 --score_type "similarity" --method "${METHOD}" \
    --num-poison "${NUM_POISON}" --poison-type "${POISON_TYPE}" \
    train --shadow-model-idx "${SHADOW_MODEL_IDX}"

# non-adaptive attack, eval_only
python -u -m experiments.self_supervised --experiment-dir "${EXPERIMENT_DIR}" --experiment "${EXPERIMENT}" --data-dir "${DATA_DIR}" \
    --seed "${SEED}" --num-shadow "${NUM_SHADOW}" \
    --num-canaries "${NUM_CANARIES}" --canary-type "${CANARY_TYPE}" \
    --finetune 1 --score_type "confidence" --method "${METHOD}" \
    --num-poison "${NUM_POISON}" --poison-type "${POISON_TYPE}" --eval_only \
    train --shadow-model-idx "${SHADOW_MODEL_IDX}"
