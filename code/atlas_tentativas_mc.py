"""Número efetivo de tentativas da busca de pulsação do atlas XMM (bexrb/atlas_obs.py), por simulação de
ruído puro: eventos uniformes no tempo, a mesma busca (Z^2_4 binado numa grade de passo 1/(8T) e refinamento
fino em volta do pico) e o H-test na frequência escolhida. Para cada limiar x de p_H, a fração f de
simulações com p_H < x dá n_ef = ln(1 - f) / ln(1 - x), comparado com Delta_nu T (a largura da busca em
frequências independentes de Fourier). A correção usada no atlas e no paper é n = 4 Delta_nu T.

Uso: python atlas_tentativas_mc.py <resultados> [n_sim]   (escreve <resultados>/atlas/tentativas_mc.json)
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from bexrb.atlas_obs import _busca_frequencia, _htest  # noqa: E402

RES = sys.argv[1]
N_SIM = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
N_EV = 1000           # eventos por simulação
T = 1.0e4             # duração (s)
NU0 = 0.1             # centro da busca (Hz)
LARGURAS = (10, 30, 100)   # Delta_nu T
LIMIARES = (0.01, 0.003, 0.001)


def uma(args):
    largura, semente = args
    rng = np.random.default_rng(semente)
    t = np.sort(rng.uniform(0.0, T, N_EV))
    nu, _, _, _ = _busca_frequencia(t, NU0, 0.5 * largura / T, T, rng)
    return _htest(np.mod(t * nu, 1.0))[2]


if __name__ == "__main__":
    out = {"n_sim": N_SIM, "n_eventos": N_EV, "larguras": {}}
    ss = np.random.SeedSequence(27)
    with Pool(int(os.environ.get("NPROC", os.cpu_count()))) as pool:
        for w in LARGURAS:
            p = np.array(pool.map(uma, [(w, s) for s in ss.spawn(N_SIM)], chunksize=20))
            r = {}
            for x in LIMIARES:
                f = float(np.mean(p < x))
                n_ef = float(np.log1p(-f) / np.log1p(-x)) if 0 < f < 1 else float("nan")
                r[str(x)] = {"fracao": f, "n_ef": n_ef, "n_ef_sobre_DnuT": n_ef / w,
                             "erro_fracao": float(np.sqrt(f * (1 - f) / N_SIM))}
                print(f"Delta_nu T = {w:4d}  p_H < {x:<6g} em {f:.4f}  n_ef = {n_ef:7.1f}  n_ef/(Delta_nu T) = {n_ef / w:.2f}",
                      flush=True)
            out["larguras"][str(w)] = r
    os.makedirs(os.path.join(RES, "atlas"), exist_ok=True)
    json.dump(out, open(os.path.join(RES, "atlas", "tentativas_mc.json"), "w"), indent=1)
