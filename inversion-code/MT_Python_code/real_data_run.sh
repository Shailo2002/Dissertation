#!/usr/bin/env bash
# Real SAMTEX data inversion — all stations, sequential.
# Each station runs fully (inversion -> postprocess -> validate)
# before the next one starts.  10 chains (6 cold + 4 hot PT).
#
# Usage (from MT_Python_code/):
#     bash real_data_run.sh              # 1000 steps (default)
#     bash real_data_run.sh 50           # custom steps, e.g. quick test

set -e

NSTEPS=${1:-1000}
NSAMPLES=1000
TEMPERATURES="1 1 1 1 1 1 2 4 8 16"   # 6 cold + 4 hot

mkdir -p logs results

for DATAFILE in data/real/SAMTEX.*.dat; do
    STATION=$(basename "$DATAFILE" .dat)
    RESULTS="results/real_${STATION}_${NSTEPS}steps"
    LOG="logs/real_${STATION}_${NSTEPS}steps.log"

    echo ""
    echo "============================================================"
    echo "  [$(date '+%H:%M:%S')] START: $STATION  (nsteps=$NSTEPS)"
    echo "  Temperatures: $TEMPERATURES"
    echo "============================================================"

    python run_inversion.py \
        --data "$DATAFILE" \
        --nsteps "$NSTEPS" --nsamples "$NSAMPLES" \
        --temperatures $TEMPERATURES \
        --parallel \
        --output "$RESULTS" 2>&1 | tee "$LOG"

    echo ""
    echo "  [$(date '+%H:%M:%S')] POSTPROCESS: $STATION"
    python postprocess/chain_convergence.py --folder "$RESULTS" 2>&1 | tee -a "$LOG"
    python postprocess/process_chains.py    --folder "$RESULTS" 2>&1 | tee -a "$LOG"
    python postprocess/plot_posterior.py    --folder "$RESULTS" 2>&1 | tee -a "$LOG"
    python postprocess/plot_noise.py        --folder "$RESULTS" 2>&1 | tee -a "$LOG"

    echo ""
    echo "  [$(date '+%H:%M:%S')] VALIDATE: $STATION"
    python postprocess/validate_results.py --folder "$RESULTS" --data "$DATAFILE" 2>&1 | tee -a "$LOG"

    echo ""
    echo "  [$(date '+%H:%M:%S')] DONE: $STATION"
done

echo ""
echo "============================================================"
echo "  ALL STATIONS COMPLETE  (nsteps=$NSTEPS, 10 chains with PT)"
echo "============================================================"
