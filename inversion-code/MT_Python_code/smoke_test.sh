#!/usr/bin/env bash
# Smoke test: run all 3 synthetic datasets end-to-end with only 5 steps
# (inversion -> chain_convergence -> process_chains -> plot_posterior ->
#  plot_noise -> validate_results).  Purely a pipeline test, not a science
# run — use sequential_test.sh (200 steps) once this works.
#
# Usage (from MT_Python_code/):
#     bash smoke_test.sh

set -e   # stop on first error

mkdir -p logs results

for NAME in default_3layer craton mobile_belt; do
    RESULTS="results/smoke_${NAME}_5steps"
    DATA="data/synthetic/${NAME}/MT_data_Z.dat"
    TRUE="data/synthetic/${NAME}/true_model.json"
    LOG="logs/smoke_${NAME}_5steps.log"

    TRUE_MODEL_ARGS=""
    if [ -f "$TRUE" ]; then
        TRUE_MODEL_ARGS="--true_model $TRUE"
    fi

    echo ""
    echo "============================================================"
    echo "  [$(date '+%H:%M:%S')] START: $NAME"
    echo "============================================================"

    python run_inversion.py --data "$DATA" $TRUE_MODEL_ARGS --nsteps 5 --nsamples 1000 --nchains 5 --parallel --output "$RESULTS" 2>&1 | tee "$LOG"

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
echo "  SMOKE TEST COMPLETE — all 3 datasets, 5 steps each"
echo "============================================================"
