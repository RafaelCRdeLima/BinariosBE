"""Poder do teste sigma_fonte = 0 com o limiar calibrado.

h1_robustez.py estima a taxa de falso positivo e o poder com ajustes em grade e o p da mistura assintótica
de chi^2. Aqui o mesmo desenho
(fontes, indicadores, trabalhos e erros reais; demais componentes nas medianas da normalização de Becker
com B_cyc) é simulado com a máxima verossimilhança contínua de h1_calibracao.py. O limiar calibrado é o
quantil de 95% do LRT sob a nula, e o poder é a fração das simulações com sigma_fonte > 0 acima dele,
comparada com o poder pelo p assintótico.

Uso: python h1_poder.py <resultados> [n_sim]   (escreve sintese/h1_poder.json)
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import h1_hierarquico as H  # noqa: E402

RES = sys.argv[1]
H.RES = RES
N_SIM = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
TETO = H.TETO

tr = H.carregar(H.d_indice(RES), "limpa")
Cm = H.cov_medida(tr, "becker")       # a distância é comum às transições de uma fonte
LM = np.linalg.cholesky(Cm)
K = [H.incid([t[k] for t in tr]) for k in ("fonte", "tipo", "ref")]
rot = {k: np.unique([t[k] for t in tr], return_inverse=True)[1] for k in ("fonte", "tipo", "ref")}
obs = json.load(open(os.path.join(RES, "sintese", "h1_hierarquico.json")))["limpa"]["becker"]
S_IND, S_TRAB, S_RES = (obs[k]["med"] for k in ("sigma_indicador", "sigma_trabalho", "sigma_residual"))


def max_ll(y, com_fonte, rng, partidas=4):
    melhor = -np.inf
    for _ in range(partidas):
        x0 = rng.uniform(0.05, 0.8, 4 if com_fonte else 3)

        def f(x):
            s = np.r_[x] if com_fonte else np.r_[0.0, x]
            return -float(H.log_ver(y, Cm, K, np.atleast_2d(s))[0])
        r = minimize(f, x0, method="L-BFGS-B", bounds=[(0.0, TETO)] * len(x0))
        melhor = max(melhor, -r.fun)
    return melhor


def uma(args):
    s_f, semente = args
    rng = np.random.default_rng(semente)
    y = (rng.normal(0, s_f, rot["fonte"].max() + 1)[rot["fonte"]]
         + rng.normal(0, S_IND, rot["tipo"].max() + 1)[rot["tipo"]]
         + rng.normal(0, S_TRAB, rot["ref"].max() + 1)[rot["ref"]]
         + rng.normal(0, S_RES, Cm.shape[0]) + LM @ rng.standard_normal(Cm.shape[0]))
    return max(0.0, 2 * (max_ll(y, True, rng) - max_ll(y, False, rng)))


if __name__ == "__main__":
    out = {"n_sim": N_SIM, "componentes": {"indicador": S_IND, "trabalho": S_TRAB, "residual": S_RES}}
    ss = np.random.SeedSequence(28)
    lrt = {}
    with Pool(int(os.environ.get("NPROC", os.cpu_count()))) as pool:
        for s_f, sem in zip((0.0, 0.3, 0.5), ss.spawn(3)):
            lrt[s_f] = np.array(pool.map(uma, [(s_f, s) for s in sem.spawn(N_SIM)], chunksize=50))
    lim_ass = chi2.isf(0.10, 1)              # p = 0,5 chi2_1(LRT) = 0,05
    lim_cal = float(np.quantile(lrt[0.0], 0.95))
    out["LRT_limiar_assintotico"] = float(lim_ass)
    out["LRT_limiar_calibrado"] = lim_cal
    for s_f, v in lrt.items():
        a, c = float(np.mean(v > lim_ass)), float(np.mean(v > lim_cal))
        out[f"sigma_fonte={s_f}"] = {"frac_rejeita_assintotico": a, "frac_rejeita_calibrado": c,
                                      "erro": float(np.sqrt(max(a, c) * (1 - min(a, c)) / N_SIM)),
                                      "frac_LRT_zero": float(np.mean(v <= 1e-9))}
        print(f"sigma_fonte={s_f}: rejeita {a:.3f} (assintótico, LRT>{lim_ass:.2f})  {c:.3f} (calibrado, LRT>{lim_cal:.2f})"
              f"  LRT=0 em {np.mean(v <= 1e-9):.2f}", flush=True)
    json.dump(out, open(os.path.join(RES, "sintese", "h1_poder.json"), "w"), indent=1)
