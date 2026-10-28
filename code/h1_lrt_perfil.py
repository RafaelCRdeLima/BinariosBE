"""Teste sigma_fonte = 0 com alfa maximizado (verossimilhança perfilada) em vez de integrado.

h1_hierarquico.py e h1_calibracao.py usam a verossimilhança com alfa integrado (prior plano), como na máxima
verossimilhança restrita (REML), que não subestima os componentes de variância. Aqui o mesmo teste é feito com
alfa maximizado: LRT, p assintótico (mistura 50:50 de chi^2_0 e chi^2_1) e bootstrap paramétrico sob a nula
ajustada também por máxima verossimilhança perfilada.

Uso: python h1_lrt_perfil.py <resultados> [B]   (escreve sintese/h1_lrt_perfil.json)
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
B = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
TETO = H.TETO

tr = H.carregar(H.d_indice(RES), "limpa")
K = [H.incid([t[k] for t in tr]) for k in ("fonte", "tipo", "ref")]
rot = {k: np.unique([t[k] for t in tr], return_inverse=True)[1] for k in ("fonte", "tipo", "ref")}
UM = np.ones(len(tr))
# covariância de medida por normalização (a distância é comum às transições de uma fonte)
CM = {n: H.cov_medida(tr, n) for n in ("L", "becker", "mushtukov", "dipolo")}


def ll_perfil(s, y, Cm):
    """ln L com alfa no máximo (sem a constante), para as sigmas s = (fonte, indicador, trabalho, residual)."""
    S = s[0] ** 2 * K[0] + s[1] ** 2 * K[1] + s[2] ** 2 * K[2] + s[3] ** 2 * np.eye(UM.size) + Cm
    a11 = UM @ np.linalg.solve(S, UM)
    return float(H.log_ver(y, Cm, K, np.atleast_2d(s))[0] + 0.5 * np.log(a11))


def max_ll(y, Cm, com_fonte, rng, partidas=4):
    melhor = (-np.inf, None)
    for _ in range(partidas):
        x0 = rng.uniform(0.05, 0.8, 4 if com_fonte else 3)

        def f(x):
            return -ll_perfil(np.r_[x] if com_fonte else np.r_[0.0, x], y, Cm)
        r = minimize(f, x0, method="L-BFGS-B", bounds=[(0.0, TETO)] * len(x0))
        if -r.fun > melhor[0]:
            melhor = (-r.fun, np.r_[r.x] if com_fonte else np.r_[0.0, r.x])
    return melhor


def lrt(y, Cm, rng):
    l1, _ = max_ll(y, Cm, True, rng)
    l0, s0 = max_ll(y, Cm, False, rng)
    return max(0.0, 2 * (l1 - l0)), s0


def uma(args):
    s0, nome, semente = args
    rng = np.random.default_rng(semente)
    Cm = CM[nome]
    y = (rng.normal(0, s0[1], rot["tipo"].max() + 1)[rot["tipo"]]
         + rng.normal(0, s0[2], rot["ref"].max() + 1)[rot["ref"]]
         + rng.normal(0, s0[3], Cm.shape[0]) + np.linalg.cholesky(Cm) @ rng.standard_normal(Cm.shape[0]))
    return lrt(y, Cm, rng)[0]


if __name__ == "__main__":
    out = {"B": B}
    rng = np.random.default_rng(2026)
    sementes = np.random.SeedSequence(2027).spawn(B)
    with Pool(int(os.environ.get("NPROC", os.cpu_count()))) as pool:
        for nome in ("L", "becker", "mushtukov", "dipolo"):
            y = np.array([t[nome] for t in tr])
            L0, s0 = lrt(y, CM[nome], rng)
            boot = np.array(pool.map(uma, [(s0, nome, s) for s in sementes], chunksize=50))
            p_boot = (1 + int(np.sum(boot >= L0 - 1e-9))) / (B + 1)
            p_ass = 1.0 if L0 <= 1e-9 else float(0.5 * chi2.sf(L0, 1))
            out[nome] = {"LRT": L0, "p_assintotico": p_ass, "p_bootstrap": p_boot,
                         "p_bootstrap_erro_mc": float(np.sqrt(p_boot * (1 - p_boot) / B)), "sigmas_nulo": s0.tolist()}
            print(f"{nome:10s} LRT={L0:6.2f} p_ass={p_ass:.3f} p_boot={p_boot:.3f}±{out[nome]['p_bootstrap_erro_mc']:.3f}",
                  flush=True)
    json.dump(out, open(os.path.join(RES, "sintese", "h1_lrt_perfil.json"), "w"), indent=1)
