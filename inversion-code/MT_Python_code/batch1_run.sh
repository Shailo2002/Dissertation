#!/usr/bin/env bash
# Batch 1 of 3 — 7 stations: CPV027 ELZ291A NEN003 bot416 dmb017 k2g006 kal017
# Run from MT_Python_code/:
#     bash batch1_run.sh          # 1000 steps (default)
#     bash batch1_run.sh 500      # custom steps

set -e

NSTEPS=${1:-1000}
NSAMPLES=1000
TEMPERATURES="1 1 1 1 1 1 2 4 8 16"   # 6 cold + 4 hot

PROF_MODEL="../../Jiff_ModEM_AP3DMT/Fine_model_10km_HS.dat"
STATION_CSV="../../1d_data_checkup/output/csv/generated_csv/selected_24_stations.csv"

# Force Python to flush output immediately (fixes silent tee)
export PYTHONUNBUFFERED=1

mkdir -p logs results

DATAFILES=(
    "data/real/SAMTEX.CPV027.2006.dat"
    "data/real/SAMTEX.ELZ291A.2008.dat"
    "data/real/SAMTEX.NEN003.2006.dat"
    "data/real/SAMTEX.bot416.2005.dat"
    "data/real/SAMTEX.dmb017.2006.dat"
    "data/real/SAMTEX.k2g006.2006.dat"
    "data/real/SAMTEX.kal017.2008.dat"
)

for DATAFILE in "${DATAFILES[@]}"; do
    STATION=$(basename "$DATAFILE" .dat)
    CODE=$(echo "$STATION" | cut -d'.' -f2)
    RESULTS="results/real_${CODE}_${NSTEPS}steps"
    LOG="logs/real_${CODE}_${NSTEPS}steps.log"

    echo ""
    echo "============================================================"
    echo "  [$(date '+%H:%M:%S')] START: $STATION  (nsteps=$NSTEPS)"
    echo "  Temperatures: $TEMPERATURES"
    echo "============================================================"

    python -u run_inversion.py \
        --data "$DATAFILE" \
        --nsteps "$NSTEPS" --nsamples "$NSAMPLES" \
        --temperatures $TEMPERATURES \
        --parallel \
        --output "$RESULTS" 2>&1 | tee "$LOG"

    echo ""
    echo "  [$(date '+%H:%M:%S')] POSTPROCESS: $STATION"
    python -u postprocess/chain_convergence.py --folder "$RESULTS" 2>&1 | tee -a "$LOG"
    python -u postprocess/process_chains.py    --folder "$RESULTS" 2>&1 | tee -a "$LOG"
    python -u postprocess/plot_posterior.py    --folder "$RESULTS" 2>&1 | tee -a "$LOG"
    python -u postprocess/plot_noise.py        --folder "$RESULTS" 2>&1 | tee -a "$LOG"

    echo ""
    echo "  [$(date '+%H:%M:%S')] VALIDATE: $STATION"

    LATLON=""
    if [ -f "$STATION_CSV" ]; then
        LATLON=$(python -u -c "
import csv, sys
station = '${STATION}'
try:
    with open('${STATION_CSV}') as f:
        for row in csv.DictReader(f):
            if row.get('Site','').strip() == station:
                print(row['Lat'], row['Lon'])
                break
except Exception:
    pass
" 2>/dev/null)
    fi

    if [ -n "$LATLON" ] && [ -f "$PROF_MODEL" ]; then
        LAT=$(echo "$LATLON" | awk '{print $1}')
        LON=$(echo "$LATLON" | awk '{print $2}')
        echo "  Station $CODE: lat=$LAT lon=$LON → using professor's 3D model"
        python -u postprocess/validate_results.py \
            --folder "$RESULTS" --data "$DATAFILE" \
            --lat "$LAT" --lon "$LON" --prof_model "$PROF_MODEL" \
            2>&1 | tee -a "$LOG"
    else
        echo "  Station $CODE: lat/lon not found — skipping 3D comparison"
        python -u postprocess/validate_results.py --folder "$RESULTS" --data "$DATAFILE" \
            2>&1 | tee -a "$LOG"
    fi

    echo ""
    echo "  [$(date '+%H:%M:%S')] DONE: $STATION"
done

echo ""
echo "============================================================"
echo "  BATCH 1 COMPLETE  (nsteps=$NSTEPS)"
echo "  Stations: CPV027 ELZ291A NEN003 bot416 dmb017 k2g006 kal017"
echo "============================================================"
