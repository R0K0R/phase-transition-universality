#!/usr/bin/env bash
# Overnight run script for phase-transition universality project.
# Runs Ising → MD (2D then 3D, N=1000/2000/4000) → analysis.
# Skips steps whose output file already exists (safe to resume).
set -euo pipefail

PYTHON="${PYTHON:-python3}"
export PYTHONUNBUFFERED=1
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# ── helpers ───────────────────────────────────────────────────────────────────

RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'; NC='\033[0m'
ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo -e "[$(ts)]  $*"; }
hr()  { echo "────────────────────────────────────────────────────────"; }

elapsed_since() {
    local s=$(( $(date +%s) - $1 ))
    printf "%dh %02dm %02ds" $(( s/3600 )) $(( (s%3600)/60 )) $(( s%60 ))
}

run_step() {
    local tag="$1"; shift          # human-readable name
    local out="$1"; shift          # output .npy file to check
    local t0=$(date +%s)

    if [[ -f "$out" ]]; then
        log "${YLW}SKIP${NC}  $tag  ($out exists)"
        return 0
    fi

    hr
    log "${YLW}START${NC} $tag"
    log "  cmd: $PYTHON $*"
    log "  log: $LOG_DIR/${tag}.log"

    if "$PYTHON" "$@" 2>&1 | tee "$LOG_DIR/${tag}.log"; then
        log "${GRN}DONE${NC}  $tag  ($(elapsed_since $t0))"
    else
        log "${RED}FAIL${NC}  $tag — check $LOG_DIR/${tag}.log"
        exit 1
    fi
}

# ── main ──────────────────────────────────────────────────────────────────────

T_GLOBAL=$(date +%s)
hr
log "Phase-transition universality — overnight run"
log "Python: $($PYTHON --version 2>&1)"
log "GPU:    $($PYTHON -c 'import cupy; print(cupy.cuda.runtime.getDeviceProperties(0)[\"name\"].decode())' 2>/dev/null || echo 'none / CuPy not found')"
hr

# 1. Ising MC (GPU Metropolis L=32-256 + CPU Wolff L=512)
run_step "ising" \
         "ising_data.npy" \
         ising.py

# 2. MD — sequential (each owns the full GPU)
# 2a. 2D coexistence curve (skip if exists — already done)
run_step "md2d_N1000" \
         "md2d_N1000.npy" \
         md_lj.py 2 1000

# 2b. 2D near-T_c run for beta extraction (N=2000, T=0.43-0.46)
run_step "md2d_N2000_nearTc" \
         "md2d_N2000_nearTc.npy" \
         md_lj.py 2 2000 nearTc

# 2c. 3D with corrected rho_c=0.48 (N=800)
run_step "md3d_N800" \
         "md3d_N800.npy" \
         md_lj.py 3 800

# 3. Analysis + plots
hr
log "${YLW}START${NC} analysis"
"$PYTHON" analysis.py 2>&1 | tee "$LOG_DIR/analysis.log"
log "${GRN}DONE${NC}  analysis"

# ── summary ───────────────────────────────────────────────────────────────────
hr
log "${GRN}ALL DONE${NC}  total time: $(elapsed_since $T_GLOBAL)"
log "Output files:"
for f in ising_data.npy md2d_N1000.npy md2d_N2000_nearTc.npy md3d_N800.npy; do
    [[ -f "$f" ]] && log "  ✓ $f  ($(du -h "$f" | cut -f1))" \
                  || log "  ${RED}✗ $f  MISSING${NC}"
done
log "Plots: $(ls *.png 2>/dev/null | tr '\n' ' ')"
hr
