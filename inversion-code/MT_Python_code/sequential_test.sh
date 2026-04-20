#!/usr/bin/env bash
# Production synthetic run: all 3 synthetic datasets at 1000 steps x
# 10 chains (6 cold T=1 + 4 hot T=2,4,8,16 for parallel tempering).
# Runs each dataset fully before starting the next.
#
# Usage (from MT_Python_code/):
#     bash sequential_test.sh            # 1000 steps (default)
#     bash sequential_test.sh 200        # custom steps, e.g. quick check

set -e

NSTEPS=${1:-1000}
NSAMPLES=1000
TEMPERATURES="1 1 1 1 1 1 2 4 8 16"   # 6 cold + 4 hot

mkdir -p logs results

for NAME in default_3layer craton mobile_belt; do
    RESULTS="results/test_${NAME}_${NSTEPS}steps"
    DATA="data/synthetic/${NAME}/MT_data_Z.dat"
    TRUE="data/synthetic/${NAME}/true_model.json"
    LOG="logs/test_${NAME}_${NSTEPS}steps.log"

    TRUE_MODEL_ARGS=""
    if [ -f "$TRUE" ]; then
        TRUE_MODEL_ARGS="--true_model $TRUE"
    fi

    echo ""
    echo "============================================================"
    echo "  [$(date '+%H:%M:%S')] START: $NAME  (nsteps=$NSTEPS)"
    echo "  Temperatures: $TEMPERATURES"
    echo "============================================================"

    python run_inversion.py \
        --data "$DATA" $TRUE_MODEL_ARGS \
        --nsteps "$NSTEPS" --nsamples "$NSAMPLES" \
        --temperatures $TEMPERATURES \
        --parallel \
        --output "$RESULTS" 2>&1 | tee "$LOG"

    echo ""
    echo "  [$(date '+%H:%M:%S')] POSTPROCESS: $NAME"
    python postprocess/chain_convergence.py --folder "$RESULTS" 2>&1 | tee -a "$LOG"
    python postprocess/process_chains.py    --folder "$RESULTS" 2>&1 | tee -a "$LOG"
    python postprocess/plot_posterior.py    --folder "$RESULTS" 2>&1 | tee -a "$LOG"
    python postprocess/plot_noise.py        --folder "$RESULTS" 2>&1 | tee -a "$LOG"

    echo ""
    echo "  [$(date '+%H:%M:%S')] VALIDATE: $NAME"
    python postprocess/validate_results.py --folder "$RESULTS" --data "$DATA" 2>&1 | tee -a "$LOG"

    echo ""
    echo "  [$(date '+%H:%M:%S')] DONE: $NAME"
done

echo ""
echo "============================================================"
echo "  SEQUENTIAL TEST COMPLETE — all 3 datasets, ${NSTEPS} steps each"
echo "============================================================"
