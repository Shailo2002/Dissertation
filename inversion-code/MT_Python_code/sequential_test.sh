#!/usr/bin/env bash
# Sequential science-quality run: all 3 synthetic datasets at 200 steps x
# 5 chains, end-to-end (inversion -> chain_convergence -> process_chains ->
# plot_posterior -> plot_noise -> validate_results).  Runs each dataset
# fully before starting the next, so only 5 chain workers are active at
# any time (safe on 8-core PCs).
#
# Usage (from MT_Python_code/):
#     bash sequential_test.sh

set -e

mkdir -p logs results

for NAME in default_3layer craton mobile_belt; do
    RESULTS="results/test_${NAME}_200steps"
    DATA="data/synthetic/${NAME}/MT_data_Z.dat"
    TRUE="data/synthetic/${NAME}/true_model.json"
    LOG="logs/test_${NAME}_200steps.log"

    # Only pass --true_model if the file exists (older synthetic datasets
    # did not generate it).
    TRUE_MODEL_ARGS=""
    if [ -f "$TRUE" ]; then
        TRUE_MODEL_ARGS="--true_model $TRUE"
    fi

    echo ""
    echo "============================================================"
    echo "  [$(date '+%H:%M:%S')] START: $NAME"
    echo "============================================================"

    python run_inversion.py --data "$DATA" $TRUE_MODEL_ARGS --nsteps 200 --nsamples 1000 --nchains 5 --parallel --output "$RESULTS" 2>&1 | tee "$LOG"

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
echo "  SEQUENTIAL TEST COMPLETE — all 3 datasets, 200 steps each"
echo "============================================================"
