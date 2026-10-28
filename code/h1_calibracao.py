"""Calibração e comparação direta do H1 (amostra limpa):

1. p do teste sigma_fonte = 0 por bootstrap paramétrico sob a hipótese nula, com máxima verossimilhança
   contínua (L-BFGS-B, várias partidas). 10^4 simulações, em paralelo.
2. Comparação direta entre normalizações. Cada normalização só soma a cada fonte uma constante conhecida
   (-log L_crit), então "ell universal com B_cyc", "ell universal com B_dip" e "L universal" são modelos
   para os mesmos dados (log L), com os mesmos parâmetros livres (alfa, sigma_indicador, sigma_trabalho,
   sigma_residual). As verossimilhanças máximas e as evidências (prior uniforme 0-2 dex em cada sigma;
   o prior plano de alfa é o mesmo em todos e se cancela) são diretamente comparáveis.
3. Extrapolação da curva de Mushtukov+2015 acima de 69 keV (só a Swift J0243, 10 das 15 transições):
   inclinações 1,0 e 2,0 além da adotada (1,45).
4. Previsão deixando uma fonte de fora: cada fonte é retirada, o modelo é ajustado às outras e as
   transições dela são previstas. ln p(y_fora | y_resto) = ln Z(todas) - ln Z(resto) (as sigmas
   integradas com a posterior do resto; alfa plano). Com o modelo "ell universal" (sigma_fonte = 0) e com
   sigma_fonte livre. A soma sobre as fontes é comparada entre normalizações.
5. Sensibilidade do dipolo ao D (a mesma grade de h1_robustez.py): ln BF e previsão contra Becker.

Uso: python h1_calibracao.py <resultados> [B] ['<json com D por fonte>']   (escreve sintese/h1_calibracao.json)
O D de cada fonte vem de sintese/indice.json (as mesmas escolhas de h1_robustez.py), a menos que seja dado.
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import chi2

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import h1_hierarquico as H  # noqa: E402

RES = sys.argv[1]
H.RES = RES
B = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
if len(sys.argv) > 3:
    D = json.loads(sys.argv[3])
else:
    D = H.d_indice(RES)
TETO = H.TETO
SEMENTE = 2026


def ll(sig, y, e, K):
    return float(H.log_ver(y, e, K, np.atleast_2d(sig))[0])


def max_ll(y, e, K, com_fonte, rng, partidas=4):
    """Máximo de ln L (alfa integrado) em [0, TETO]^4, com sigma_fonte = 0 se com_fonte é falso."""
    melhor = (-np.inf, None)
    for _ in range(partidas):
        x0 = rng.uniform(0.05, 0.8, 4)
        if not com_fonte:
            x0 = x0[1:]

        def f(x):
            s = np.r_[x] if com_fonte else np.r_[0.0, x]
            return -ll(s, y, e, K)
        r = minimize(f, x0, method="L-BFGS-B", bounds=[(0.0, TETO)] * len(x0))
        if -r.fun > melhor[0]:
            melhor = (-r.fun, np.r_[r.x] if com_fonte else np.r_[0.0, r.x])
    return melhor


def lrt(y, e, K, rng):
    l1, _ = max_ll(y, e, K, True, rng)
    l0, s0 = max_ll(y, e, K, False, rng)
    return max(0.0, 2 * (l1 - l0)), s0


def simular_nulo(s0, Cm, rot, rng):
    """Conjunto sob a nula; o erro de medida tem a covariância Cm (distância comum às transições da fonte)."""
    return (rng.normal(0, s0[1], rot["tipo"].max() + 1)[rot["tipo"]]
            + rng.normal(0, s0[2], rot["ref"].max() + 1)[rot["ref"]]
            + rng.normal(0, s0[3], Cm.shape[0]) + np.linalg.cholesky(Cm) @ rng.standard_normal(Cm.shape[0]))


def uma_simulacao(args):
    """LRT de um conjunto simulado sob a nula (usa K e rot do processo principal, herdados no fork)."""
    s0, Cm, semente = args
    rng = np.random.default_rng(semente)
    return lrt(simular_nulo(s0, Cm, rot, rng), Cm, K, rng)[0]


def grade(com_fonte, passo):
    """Grade e ln dos pesos da regra do trapézio (meio peso nas bordas 0 e TETO); a integral converge com
    passo 0,05 e 0,025 e concorda com uma integração de Sobol."""
    g = np.arange(0.0, TETO + 1e-9, passo)
    w = np.where((g == 0) | np.isclose(g, TETO), 0.5, 1.0)
    sa, wa = (g, w) if com_fonte else (np.array([0.0]), np.array([1.0]))
    G = np.array(np.meshgrid(sa, g, g, g, indexing="ij")).reshape(4, -1).T
    W = np.einsum("i,j,k,l->ijkl", wa, w, w, w).ravel()
    return G, np.log(W / W.sum())


G0 = grade(False, 0.05)     # ell universal: sigma_fonte = 0
G1 = grade(True, 0.1)       # sigma_fonte livre


def ln_z(tr, y, g, y_nome="L"):
    """ln da evidência (prior uniforme nas sigmas, alfa plano), com a constante (2 pi)^-(n-1)/2."""
    G, lw = g
    K = [H.incid([t[k] for t in tr]) for k in ("fonte", "tipo", "ref")]
    lv = H.log_ver(y, H.cov_medida(tr, y_nome), K, G)
    return float(logsumexp(lv + lw) - 0.5 * (y.size - 1) * np.log(2 * np.pi))


def prever_fora(tr, y, g, y_nome="L"):
    """ln p(transições de cada fonte | as outras fontes), por fonte e somado."""
    fontes = sorted(set(t["fonte"] for t in tr))
    z_todas = ln_z(tr, y, g, y_nome)
    out = {}
    for f in fontes:
        m = np.array([t["fonte"] != f for t in tr])
        resto = [t for t, k in zip(tr, m) if k]
        lp = z_todas - ln_z(resto, y[m], g, y_nome)
        out[f] = {"n": int((~m).sum()), "ln_p": lp, "ln_p_por_transicao": lp / (~m).sum(),
                  "desvio_medio_dex": float(y[~m].mean() - y[m].mean())}
    out["soma"] = float(sum(v["ln_p"] for v in out.values()))
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(SEMENTE)
    tr = H.carregar(D, "limpa")
    K = [H.incid([t[k] for t in tr]) for k in ("fonte", "tipo", "ref")]
    rot = {k: np.unique([t[k] for t in tr], return_inverse=True)[1] for k in ("fonte", "tipo", "ref")}
    swift = np.array([t["fonte"] == "Swift J0243.6+6124" for t in tr])

    # Mushtukov acima de 69 keV: L_crit(146) = L(69) (146/69)^s; ell = L / L_crit
    def mush(s):
        y = np.array([t["mushtukov"] for t in tr])
        return y - swift * (s - 1.4461) * np.log10(146.0 / 69.0)

    Y = {"L": np.array([t["L"] for t in tr]), "becker": np.array([t["becker"] for t in tr]),
         "mushtukov": mush(1.4461), "mushtukov_incl1.0": mush(1.0), "mushtukov_incl2.0": mush(2.0),
         "dipolo": np.array([t["dipolo"] for t in tr])}
    out = {"N": len(tr), "B_bootstrap": B, "testes": {}, "comparacao_direta": {}, "previsao_fonte_fora": {}}
    sementes = np.random.SeedSequence(SEMENTE).spawn(B)
    with Pool(int(os.environ.get("NPROC", os.cpu_count()))) as pool:
        for nome, y in Y.items():
            yn = "dipolo" if nome == "dipolo" else "L"
            Cm = H.cov_medida(tr, yn)
            L0, s0 = lrt(y, Cm, K, rng)
            boot = np.array(pool.map(uma_simulacao, [(s0, Cm, s) for s in sementes], chunksize=50))
            n_ge = int(np.sum(boot >= L0 - 1e-9))
            p_boot = (1 + n_ge) / (B + 1)
            p_ass = 1.0 if L0 <= 1e-9 else float(0.5 * chi2.sf(L0, 1))
            out["testes"][nome] = {"LRT": L0, "p_assintotico": p_ass, "p_bootstrap": p_boot,
                                   "p_bootstrap_erro_mc": float(np.sqrt(p_boot * (1 - p_boot) / B)),
                                   "sigmas_nulo": s0.tolist(), "frac_boot_LRT_zero": float(np.mean(boot <= 1e-9))}
            lmax0 = max_ll(y, Cm, K, False, rng)[0]
            out["comparacao_direta"][nome] = {"lnL_max_nulo": lmax0, "ln_evidencia_nula": ln_z(tr, y, G0, yn)}
            out["previsao_fonte_fora"][nome] = {"ell_universal": prever_fora(tr, y, G0, yn),
                                                "sigma_fonte_livre": prever_fora(tr, y, G1, yn)}
            print(f"{nome:18s} LRT={L0:6.2f} p_ass={p_ass:.3f} p_boot={p_boot:.4f}±{out['testes'][nome]['p_bootstrap_erro_mc']:.4f} "
                  f"(LRT*=0 em {np.mean(boot <= 1e-9):.2f})  lnL0={lmax0:.2f} "
                  f"lnZ0={out['comparacao_direta'][nome]['ln_evidencia_nula']:.2f} "
                  f"prev={out['previsao_fonte_fora'][nome]['ell_universal']['soma']:.2f}", flush=True)
    ref = out["comparacao_direta"]["becker"]
    pref = out["previsao_fonte_fora"]["becker"]
    for nome, v in out["comparacao_direta"].items():
        v["delta_lnL_vs_becker"] = v["lnL_max_nulo"] - ref["lnL_max_nulo"]
        v["ln_fator_bayes_vs_becker"] = v["ln_evidencia_nula"] - ref["ln_evidencia_nula"]
        p = out["previsao_fonte_fora"][nome]
        for m in ("ell_universal", "sigma_fonte_livre"):
            p[m]["soma_vs_becker"] = p[m]["soma"] - pref[m]["soma"]
            for f in p[m]:
                if isinstance(p[m][f], dict):
                    p[m][f]["vs_becker"] = p[m][f]["ln_p"] - pref[m][f]["ln_p"]
        print(f"  {nome:18s} delta lnL = {v['delta_lnL_vs_becker']:+.2f}  ln BF = {v['ln_fator_bayes_vs_becker']:+.2f}  "
              f"previsão (ell univ.) = {p['ell_universal']['soma_vs_becker']:+.2f}  "
              f"(sigma_fonte livre) = {p['sigma_fonte_livre']['soma_vs_becker']:+.2f}  por fonte: "
              + "  ".join(f"{f[:6]} {p['ell_universal'][f]['vs_becker']:+.2f}" for f in p["ell_universal"]
                          if isinstance(p["ell_universal"][f], dict)))

    # 5. dipolo: a mesma grade de D de h1_robustez.py (cenários A e B da seção 6.1)
    out["dipolo_sensibilidade"] = []
    DS, DA = H.grade_d(RES)
    for ds in DS:
        for da in DA:
            t2 = H.carregar(dict(D, **{"Swift J0243.6+6124": ds, "1A 0535+262": da}), "limpa")
            y = np.array([t["dipolo"] for t in t2])
            r = {"D_Swift": ds, "D_1A0535": da,
                 "lnBF_vs_becker": ln_z(t2, y, G0, "dipolo") - ref["ln_evidencia_nula"],
                 "previsao_vs_becker": prever_fora(t2, y, G0, "dipolo")["soma"] - pref["ell_universal"]["soma"],
                 "previsao_sigma_fonte_livre_vs_becker": prever_fora(t2, y, G1, "dipolo")["soma"] - pref["sigma_fonte_livre"]["soma"]}
            out["dipolo_sensibilidade"].append(r)
            print(f"dipolo D_Swift={ds:5.2f} D_1A={da:4.2f} ln BF={r['lnBF_vs_becker']:+.2f} "
                  f"previsão={r['previsao_vs_becker']:+.2f} ({r['previsao_sigma_fonte_livre_vs_becker']:+.2f})", flush=True)
    json.dump(out, open(os.path.join(RES, "sintese", "h1_calibracao.json"), "w"), indent=1)
