#!/bin/bash
# Regenerates every product in results/ and the tables and figures of the paper in output/, in the order in
# which they depend on each other.
#     PY=python3 NPROC=12 ./run_all.sh                    # everything except the XMM-Newton atlas
#     XMM_PRODUCTS=/path/to/reduced/xmm ./run_all.sh     # also rebuilds the atlas from the reduced XMM data
# Run `python code/baixar_dados.py` first to download the Fermi/GBM and MAXI data.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/code"
R="$ROOT/results"
PY="${PY:-python3}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# 1. XMM-Newton timing atlas
if [ -n "${XMM_PRODUCTS:-}" ]; then
    "$PY" frequencia_esperada.py                     # dados/nu_esperado.json (expected spin frequencies)
    "$PY" rodar_atlas.py "$XMM_PRODUCTS" "$R/atlas" --proc "${NPROC:-8}"
fi
"$PY" atlas_correcoes.py "$R"                        # detections, upper limits, pulsed fractions
"$PY" atlas_tentativas_mc.py "$R"                    # effective number of trials of the pulsation search

# 2. Spin-up points and transition compilation
"$PY" torque.py "$R"
"$PY" sintese.py "$R"

# 3. Torque fits
"$PY" torque_ajuste.py "$R"
"$PY" torque_ajuste.py "$R" --indep
"$PY" torque_ajuste.py "$R" --I=1.43e45
"$PY" reversao.py "$R" --boot="${NBOOT_REV:-2000}"

# 4. Field-discrepancy index and hierarchical model of the transitions
"$PY" indice.py "$R"
"$PY" h1_hierarquico.py "$R"
"$PY" h1_calibracao.py "$R" "${NBOOT_H1:-10000}"
"$PY" h1_poder.py "$R" 4000
"$PY" h1_lrt_perfil.py "$R" 4000
"$PY" h1_robustez.py "$R"

# 5. Tables and figures of the paper
mkdir -p "$ROOT/output/tables" "$ROOT/output/figures"
"$PY" tabelas_paper.py "$R" "$ROOT/output/tables"
"$PY" figuras_paper.py "$R" "$ROOT/output/figures"
echo "done: tables and figures in $ROOT/output"
