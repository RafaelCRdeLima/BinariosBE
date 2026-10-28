"""Pontos de spin-up (nu-ponto do Fermi/GBM contra a luminosidade do MAXI) e uma estimativa ponto a ponto do
campo dipolar pelo torque. O paper usa só os pontos; os ajustes estão em torque_ajuste.py e reversao.py.

nu-ponto: diferenças finitas entre detecções consecutivas do Fermi/GBM (frequência já no
referencial do pulsar) separadas por <= 8 d. L: MAXI 2-20 keV, média diária no mesmo intervalo,
com pesos 1/sigma^2 (erro da média ponderada), convertida em fluxo pela taxa da Nebulosa do Caranguejo no próprio MAXI (2-20 keV) e
F_Crab(2-20) = 3,0e-8 erg/cm2/s (Toor & Seward 1974), vezes K_bol = F(0,5-100)/F(2-20) = 2
(cutoffpl Gamma=0,5, E_fold=15 keV; ~1,3-4,8 para Gamma 0-1 e E_fold 8-25 keV).

Torque de Ghosh & Lamb (1979):
    nu-ponto = n(w) Mdot sqrt(G M r_m) / (2 pi I),  r_m = xi (mu^4 / (2 G M Mdot^2))^(1/7),
    n(w) = 1,39 {1 - w [4,03 (1-w)^0,173 - 0,878]} / (1 - w),  w = (r_m / r_co)^(3/2),
com Mdot = L R / (G M), M = 1,4 Msol, R = 12 km, I = 1e45 g cm2, xi = 0,5. Ajustamos só mu
(mínimos quadrados em nu-ponto). Como mu ~ nu-ponto^(7/2) L^(-3), mu ~ d^(-6): a incerteza da
distância domina e é propagada por amostragem.

Uso: python torque.py [<resultados>]  (lê dados/gbm, dados/maxi, dados/parametros_fontes.csv; escreve
<resultados>/torque/pontos_*.csv e campos_torque.csv; sem argumento, resultados/ ao lado do código)
"""
import csv
import json
import os
import sys

import numpy as np
from astropy.io import fits
from scipy.optimize import minimize_scalar

AQUI = os.path.dirname(os.path.abspath(__file__))
SAIDA = os.path.join(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--")
                     else os.path.join(AQUI, "resultados"), "torque")
G, MSOL, C = 6.674e-8, 1.989e33, 2.998e10
M, R, I, XI = 1.4 * MSOL, 1.2e6, 1e45, 0.5
KPC = 3.0857e21
F_CRAB_2_20 = 3.0e-8
K_BOL = 2.0
GBM = {"4U 0115+63": ("4u0115", "J0118+637"), "V 0332+53": ("v0332", "J0334+531"),
       "1A 0535+262": ("a0535", "J0538+263"), "Swift J0243.6+6124": ("swiftj0243", "J0243+614")}


def n_gl(w):
    w = np.clip(w, 0, 0.999)
    return 1.39 * (1 - w * (4.03 * (1 - w) ** 0.173 - 0.878)) / (1 - w)


def nudot_modelo(mu, L, nu):
    mdot = L * R / (G * M)
    rm = XI * (mu ** 4 / (2 * G * M * mdot ** 2)) ** (1 / 7)
    rco = (G * M / (2 * np.pi * nu) ** 2) ** (1 / 3)
    w = (rm / rco) ** 1.5
    return n_gl(w) * mdot * np.sqrt(G * M * rm) / (2 * np.pi * I)


# corte dos dados do GBM e do MAXI (2026-10-17): as últimas detecções do GBM usadas são de MJD 61316,5, e os
# últimos dias do MAXI são provisórios e revisados depois; com o corte, arquivos baixados mais tarde dão os
# mesmos pontos
MJD_MAX = 61330.5
# taxa mediana do Crab no MAXI/GSC, 2-20 keV (contagens/s/cm^2), em MJD 55051,5-61338,5, que normaliza todos os
# fluxos do paper; fixa porque a mediana de um arquivo mais novo muda nas últimas casas
CRAB_MAXI = 3.4930655


def ler_maxi(nome):
    """Curva diária do MAXI/GSC até MJD_MAX (colunas: MJD, taxa 2-20 keV, erro, e as bandas 2-4, 4-10, 10-20)."""
    d = np.loadtxt(os.path.join(AQUI, "dados", "maxi", nome + "_1day.dat"))
    return d[d[:, 0] <= MJD_MAX]


def maxi(nome):
    d = ler_maxi(nome)
    return d[:, 0], d[:, 1], d[:, 2]


def pontos(fonte, d_kpc):
    gbm, mx = GBM[fonte]
    h = fits.getdata(os.path.join(AQUI, "dados", "gbm", gbm + ".fits.gz"), "PSR_HIST")
    ok = (h["DETECTED"] == 1) & (h["BARYTIME"] <= MJD_MAX)
    t, nu, sg = h["BARYTIME"][ok], h["FREQUENCY"][ok], h["FREQUENCY_ERR"][ok]
    o = np.argsort(t)
    t, nu, sg = t[o], nu[o], sg[o]
    crab = CRAB_MAXI
    tm, rm, em = maxi(mx)
    out = []
    for i in range(t.size - 1):
        dt = t[i + 1] - t[i]
        if dt <= 0 or dt > 8:
            continue
        s = (tm >= t[i]) & (tm <= t[i + 1])
        if s.sum() < 1:
            continue
        w = 1 / np.maximum(em[s], 1e-3) ** 2
        taxa = np.average(rm[s], weights=w)
        if taxa <= 0:
            continue
        L = K_BOL * F_CRAB_2_20 * taxa / crab * 4 * np.pi * (d_kpc * KPC) ** 2
        out.append({"mjd": 0.5 * (t[i] + t[i + 1]), "nudot": (nu[i + 1] - nu[i]) / (dt * 86400),
                    "sig": np.hypot(sg[i], sg[i + 1]) / (dt * 86400), "nu": 0.5 * (nu[i] + nu[i + 1]),
                    # erro da média ponderada com os mesmos pesos
                    "taxa_maxi": taxa, "L": L, "sig_taxa": np.sqrt(np.sum(w ** 2 * em[s] ** 2)) / w.sum(),
                    # para a covariância: pares vizinhos (i, i+1) e (i+1, i+2) dividem a frequência i+1
                    "i": i, "dt_s": dt * 86400, "sig_nu0": sg[i], "sig_nu1": sg[i + 1]})
    return out, crab


GRADE_LMU = np.linspace(25.0, 34.0, 1801)


def raizes_mu(nd, L, nu):
    """As duas soluções de Ghosh-Lamb para um ponto: nu-ponto(mu) sobe com mu (rotador lento,
    w << 1), passa por um máximo e cai a zero em w = w_c (rotador rápido, perto do equilíbrio).
    Devolve (log mu lento, log mu rápido); None se o nu-ponto observado passa do máximo do modelo."""
    m = nudot_modelo(10 ** GRADE_LMU, L, nu)
    # a fórmula de n(w) só vale até w_c (n = 0); depois dela o ramo não tem sentido aqui
    zero = np.where(m <= 0)[0]
    fim = int(zero[0]) if zero.size else m.size
    m, g = m[:fim], GRADE_LMU[:fim]
    k = int(np.argmax(m))
    if m[k] < nd:
        return None
    lento = np.interp(nd, m[:k + 1], g[:k + 1])
    rapido = np.interp(-nd, -m[k:], g[k:]) if fim < GRADE_LMU.size else np.nan
    return float(lento), float(rapido)


def ajustar(p, frac_pico=0.3):
    """Fase de alta L (L >= frac_pico * L_max, nu-ponto > 0): as duas soluções de cada ponto por
    inversão direta; por solução, a mediana, a dispersão (16-84%) e o erro da mediana. Ajusta também
    a inclinação de log nu-ponto contra log L (6/7 no ramo lento)."""
    L = np.array([x["L"] for x in p])
    nd = np.array([x["nudot"] for x in p])
    nu = np.array([x["nu"] for x in p])
    s = (L >= frac_pico * L.max()) & (nd > 0)
    if s.sum() < 5:
        return None
    rz = [raizes_mu(a, b, c) for a, b, c in zip(nd[s], L[s], nu[s])]
    ok = [r for r in rz if r]
    out = {"n": int(s.sum()), "n_sem_solucao": int(len(rz) - len(ok)), "L_max": float(L.max()),
           "inclinacao_loglog": float(np.polyfit(np.log10(L[s]), np.log10(nd[s]), 1)[0])}
    for i, nome in ((0, "lento"), (1, "rapido")):
        v = np.array([r[i] for r in ok])
        q16, q50, q84 = np.percentile(v, [16, 50, 84])
        e = 1.2533 * (q84 - q16) / 2 / np.sqrt(v.size)
        out[nome] = {"log_mu": float(q50), "err": float(e), "disp16": float(q16), "disp84": float(q84)}
    return out


def b_polo(mu):
    return 2 * mu / R ** 3


if __name__ == "__main__":
    os.makedirs(SAIDA, exist_ok=True)
    par = {x["fonte"]: x for x in csv.DictReader(open(os.path.join(AQUI, "dados", "parametros_fontes.csv")))}
    tab = []
    for fonte in GBM:
        d, sd = float(par[fonte]["d_kpc"]), float(par[fonte]["d_err_kpc"])
        p, crab = pontos(fonte, d)
        with open(os.path.join(SAIDA, f"pontos_{GBM[fonte][0]}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, list(p[0].keys()))
            w.writeheader()
            w.writerows(p)
        a = ajustar(p)
        if not a:
            continue
        ecyc = float(par[fonte]["E_cyc_keV"])
        um_z = (1 - 2 * G * M / (R * C ** 2)) ** -0.5
        b_cyc = um_z * ecyc / 11.57 * 1e12
        linha = {"fonte": fonte, "d_kpc": d, "d_err_kpc": sd, "n_pontos_alto_L": a["n"],
                 "n_sem_solucao": a["n_sem_solucao"], "L_max": a["L_max"],
                 "inclinacao_loglog": a["inclinacao_loglog"], "E_cyc_keV": ecyc, "B_cyc_G": b_cyc}
        for ramo in ("lento", "rapido"):
            r = a[ramo]
            # distância: no ramo lento mu ~ d^-6; usamos a mesma propagação nos dois (aproximação)
            err = np.hypot(r["err"], 6 * sd / d / np.log(10))
            bp = b_polo(10 ** r["log_mu"])
            linha.update({f"log_mu_{ramo}": r["log_mu"], f"err_log_mu_{ramo}": err,
                          f"disp16_{ramo}": r["disp16"], f"disp84_{ramo}": r["disp84"],
                          f"B_dip_{ramo}_G": bp, f"M_{ramo}": b_cyc / bp,
                          f"M_{ramo}_lo": b_cyc / b_polo(10 ** (r["log_mu"] + err)),
                          f"M_{ramo}_hi": b_cyc / b_polo(10 ** (r["log_mu"] - err))})
        linha["d_para_M1_lento_kpc"] = d * (linha["B_dip_lento_G"] / b_cyc) ** (1 / 6)
        linha["crab_maxi"] = crab
        tab.append(linha)
        print(json.dumps(linha, ensure_ascii=False))
    # M relativo: I, xi, K_bol e a normalização do modelo são comuns e se cancelam nas razões
    ref = next(x for x in tab if x["fonte"] == "1A 0535+262")
    for x in tab:
        x["M_lento_relativo_a_1A0535"] = x["M_lento"] / ref["M_lento"]
    with open(os.path.join(SAIDA, "campos_torque.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(tab[0].keys()))
        w.writeheader()
        w.writerows(tab)
