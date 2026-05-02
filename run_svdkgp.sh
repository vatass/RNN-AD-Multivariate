

#!/bin/bash

PYTHON_SCRIPT="svdkgp.py"

FILE="subjectsamples_longclean_dl_muse_allstudies.csv"
FOLDER=2
HIDDEN_DIM=64
POINTS=3
EPOCHS=30
LAMBDA_VAL=0.0
HELDOUT=-1

INCREASING_INDICES=(0 1 14 15 16 17)

is_increasing() {
    local idx=$1
    for inc_idx in "${INCREASING_INDICES[@]}"; do
        if [[ "$idx" -eq "$inc_idx" ]]; then
            return 0
        fi
    done
    return 1
}

for INDEX in {0..1}; do

    if is_increasing "$INDEX"; then
        MODE=2
    else
        MODE=1
    fi

    echo "Running INDEX=$INDEX with MODE=$MODE"

    python "$PYTHON_SCRIPT" \
        --experimentID "$EXPERIMENT_ID" \
        --file "$FILE" \
        --folder "$FOLDER" \
        --mode "$MODE" \
        --hidden_dim "$HIDDEN_DIM" \
        --points "$POINTS" \
        --epochs "$EPOCHS" \
        --lambda_val "$LAMBDA_VAL" \
        --heldout "$HELDOUT" \
        --index "$INDEX"

done