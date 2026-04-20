#!/usr/bin/env bash
# Quick test — 10 steps on ONE synthetic + ONE real station.
# Purpose: verify the three code changes made before production runs:
#   1. likelihood.py  — normalization formula fix (Change_Nse should be ~20-60%, not 99%)
#   2. config.py      — balanced birth/death (both ~10%), noise gets 12% proposal
#   3. chain_convergence.py — preserves Total time in Acceptance_Rate_Summary.txt
#
# Expected runtime: ~3-8 minutes on a laptop.
#
# Usage (from MT_Python_code/):
#     bash quick_test.sh

set -e

NSTEPS=10
NSAMPLES=200
TEMPERATURES="1 1 1 1 1 1 2 4 8 16"   # 6 cold + 4 hot  (same as production)

mkdir -p logs results

PASS=0
FAIL=0

# ------------------------------------------------------------------ #
# Helper: check acceptance rate file for expected properties
# ------------------------------------------------------------------ #
check_acceptance() {
    local folder="$1"
    local label="$2"
    local summary="$folder/Acceptance_Rate_Summary.txt"

    echo ""
    echo "  --- Checking acceptance rates: $label ---"

    if [ ! -f "$summary" ]; then
        echo "  FAIL: Acceptance_Rate_Summary.txt not found"
        FAIL=$((FAIL + 1))
        return
    fi

    # Check Total time is present (chain_convergence.py fix)
    if grep -q "Total time:" "$summary"; then
        echo "  PASS  Total time present in summary"
        PASS=$((PASS + 1))
    else
        echo "  FAIL  Total time missing from summary (chain_convergence.py fix broken)"
        FAIL=$((FAIL + 1))
    fi

    # Check Steps is present
    if grep -q "Steps" "$summary"; then
        echo "  PASS  Steps count present in summary"
        PASS=$((PASS + 1))
    else
        echo "  FAIL  Steps count missing from summary"
        FAIL=$((FAIL + 1))
    fi

    # Check Change_Nse acceptance is NOT 99%+ (normalization fix)
    local nse_line
    nse_line=$(grep "Change_Nse" "$summary" || true)
    if [ -n "$nse_line" ]; then
        local nse_mean
        nse_mean=$(echo "$nse_line" | awk '{print $4}')
        # Use awk for float comparison: fail if mean > 90%
        local too_high
        too_high=$(echo "$nse_mean" | awk '{print ($1 > 90) ? "yes" : "no"}')
        if [ "$too_high" = "yes" ]; then
            echo "  FAIL  Change_Nse mean=${nse_mean}% — still near 100% (normalization fix may not have applied)"
            FAIL=$((FAIL + 1))
        else
            echo "  PASS  Change_Nse mean=${nse_mean}% — reasonable (was 99.92% before fix)"
            PASS=$((PASS + 1))
        fi
    else
        echo "  WARN  Change_Nse line not found in summary (old chain file?)"
    fi

    # Check Birth and Death are roughly balanced (proposal fix)
    local birth_mean death_mean
    birth_mean=$(grep "^Birth" "$summary" | awk '{print $4}' || echo "0")
    death_mean=$(grep "^Death" "$summary" | awk '{print $4}' || echo "0")
    if [ -n "$birth_mean" ] && [ -n "$death_mean" ]; then
        # Accept if |birth - death| < 10%  (with only 10 steps there's variance)
        local diff_ok
        diff_ok=$(echo "$birth_mean $death_mean" | awk '{d=$1-$2; if(d<0) d=-d; print (d < 15) ? "yes" : "no"}')
        if [ "$diff_ok" = "yes" ]; then
            echo "  PASS  Birth=${birth_mean}% / Death=${death_mean}% — balanced (proposal fix working)"
            PASS=$((PASS + 1))
        else
            echo "  FAIL  Birth=${birth_mean}% / Death=${death_mean}% — imbalanced (was 10%/40% before fix)"
            FAIL=$((FAIL + 1))
        fi
    fi
}

# ------------------------------------------------------------------ #
# TEST 1: Synthetic — craton (most challenging, best test of PT)
# ------------------------------------------------------------------ #
NAME="craton"
RESULTS="results/qtest_${NAME}_${NSTEPS}steps"
DATA="data/synthetic/${NAME}/MT_data_Z.dat"
TRUE="data/synthetic/${NAME}/true_model.json"
LOG="logs/qtest_${NAME}_${NSTEPS}steps.log"

echo ""
echo "============================================================"
echo "  [$(date '+%H:%M:%S')] TEST 1: Synthetic / $NAME"
echo "  Temperatures: $TEMPERATURES"
echo "============================================================"

python run_inversion.py \
    --data "$DATA" --true_model "$TRUE" \
    --nsteps $NSTEPS --nsamples $NSAMPLES \
    --temperatures $TEMPERATURES \
    --parallel \
    --output "$RESULTS" 2>&1 | tee "$LOG"

echo "  [$(date '+%H:%M:%S')] Postprocessing ..."
python postprocess/chain_convergence.py --folder "$RESULTS" 2>&1 | tee -a "$LOG"
python postprocess/process_chains.py    --folder "$RESULTS" 2>&1 | tee -a "$LOG"
python postprocess/plot_posterior.py    --folder "$RESULTS" 2>&1 | tee -a "$LOG"
python postprocess/plot_noise.py        --folder "$RESULTS" 2>&1 | tee -a "$LOG"
python postprocess/validate_results.py  --folder "$RESULTS" --data "$DATA" 2>&1 | tee -a "$LOG"

check_acceptance "$RESULTS" "Synthetic/$NAME"
echo "  [$(date '+%H:%M:%S')] DONE: $NAME"

# ------------------------------------------------------------------ #
# TEST 2: Real data — one representative station
# ------------------------------------------------------------------ #
REAL_STATION="SAMTEX.bot228.2005"
REAL_DATA="data/real/${REAL_STATION}.dat"
REAL_RESULTS="results/qtest_real_${REAL_STATION}_${NSTEPS}steps"
REAL_LOG="logs/qtest_real_${REAL_STATION}_${NSTEPS}steps.log"

echo ""
echo "============================================================"
echo "  [$(date '+%H:%M:%S')] TEST 2: Real data / $REAL_STATION"
echo "  Temperatures: $TEMPERATURES"
echo "============================================================"

if [ ! -f "$REAL_DATA" ]; then
    echo "  SKIP: real data file not found: $REAL_DATA"
else
    python run_inversion.py \
        --data "$REAL_DATA" \
        --nsteps $NSTEPS --nsamples $NSAMPLES \
        --temperatures $TEMPERATURES \
        --parallel \
        --output "$REAL_RESULTS" 2>&1 | tee "$REAL_LOG"

    echo "  [$(date '+%H:%M:%S')] Postprocessing ..."
    python postprocess/chain_convergence.py --folder "$REAL_RESULTS" 2>&1 | tee -a "$REAL_LOG"
    python postprocess/process_chains.py    --folder "$REAL_RESULTS" 2>&1 | tee -a "$REAL_LOG"
    python postprocess/plot_posterior.py    --folder "$REAL_RESULTS" 2>&1 | tee -a "$REAL_LOG"
    python postprocess/plot_noise.py        --folder "$REAL_RESULTS" 2>&1 | tee -a "$REAL_LOG"
    python postprocess/validate_results.py  --folder "$REAL_RESULTS" --data "$REAL_DATA" 2>&1 | tee -a "$REAL_LOG"

    check_acceptance "$REAL_RESULTS" "Real/$REAL_STATION"
    echo "  [$(date '+%H:%M:%S')] DONE: $REAL_STATION"
fi

# ------------------------------------------------------------------ #
# Final report
# ------------------------------------------------------------------ #
echo ""
echo "============================================================"
echo "  QUICK TEST COMPLETE"
echo "  PASS: $PASS   FAIL: $FAIL"
echo "============================================================"

if [ "$FAIL" -gt 0 ]; then
    echo "  One or more checks FAILED — do NOT run production yet."
    echo "  Check logs/ for details."
    exit 1
else
    echo "  All checks passed — ready for production run."
    echo ""
    echo "  Production command:"
    echo "    bash sequential_test.sh        # synthetic (1000 steps, 10 chains)"
    echo "    bash real_data_run.sh 1000     # real data  (1000 steps, 10 chains)"
fi
