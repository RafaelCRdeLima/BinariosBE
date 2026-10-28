"""H1: as luminosidades de transição da literatura em unidades de L_crit(B_cyc), com as
distâncias do Gaia; e o ell de cada observação XMM. Uso: python sintese.py <resultados>

Entradas: dados/literatura/luminosidades.csv (compilação da literatura, com a
referência de cada valor), dados/parametros_fontes.csv (distância e E_cyc adotados),
<resultados>/atlas/catalogo_obs.csv, dados/maxi.
Saídas: <resultados>/sintese/{transicoes_h1.csv, atlas_ell.csv}, <resultados>/figuras/h1_transicoes.png

L_crit de Becker et al. (2012, eq. 32, Lambda = 0,1, w = 1, M = 1,4, R = 10 km):
    L_crit = 1,49e37 B12^(16/15) erg/s, com B12 = (1+z) E_cyc / 11,57 keV, (1+z) = 1,24 (R = 12 km).
A L da literatura é reescalada por (d_Gaia / d_usada)^2 e não é convertida de banda: a banda
fica registrada, porque a correção bolométrica de uma banda estreita pode chegar a ~2-4x.
"""
import csv
import json
import os
import re
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from torque import CRAB_MAXI, ler_maxi  # noqa: E402

RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(AQUI, "resultados")
LIT = os.path.join(AQUI, "dados", "literatura")
os.makedirs(os.path.join(RES, "sintese"), exist_ok=True)
os.makedirs(os.path.join(RES, "figuras"), exist_ok=True)
UM_Z = 1.24
KPC = 3.0857e21
ORDEM = ["4U 0115+63", "V 0332+53", "1A 0535+262", "Swift J0243.6+6124", "X Per"]
COR = dict(zip(ORDEM, ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]))
MARC = dict(zip(ORDEM, ["o", "s", "^", "D", "v"]))
TIPOS = ["transicao_perfil", "minimo_PF", "troca_ramo_HID", "inversao_Ecyc_L", "L_crit_inferida"]


def num(x):
    """Número da tabela; faixa 'a-b' vira a média geométrica (a faixa fica na coluna original)."""
    t = str(x).replace("~", "").replace("≈", "").replace(" ", "").strip()
    m = re.fullmatch(r"([\d.]+(?:e[+-]?\d+)?)-([\d.]+(?:e[+-]?\d+)?)", t)
    if m:
        return float(np.sqrt(float(m.group(1)) * float(m.group(2))))
    try:
        return float(t)
    except (TypeError, ValueError):
        return np.nan


def l_crit_becker(e_cyc):
    return 1.49e37 * (UM_Z * e_cyc / 11.57) ** (16 / 15)


# Mushtukov et al. (2015), fig. 6, painel W (a coluna enche o funil), curva vermelha (modo X); M = 1,4,
# R = 10 km, T_e = 1 keV (Lambda e l0/l da legenda são do caso D). Eixo x: E_cyc observado.
# Digitalizada a 600 dpi pelo centro da curva. Acima de 69 keV (Swift J0243) extrapolamos a inclinação de
# 40-69 keV, 1,45.
_MUSH_E = np.array([10, 11, 12.5, 15, 20, 25, 28, 30, 40, 46, 50, 60, 65, 69])
_MUSH_L = np.array([0.32, 0.30, 0.29, 0.31, 0.41, 0.56, 0.65, 0.72, 1.10, 1.36, 1.54, 1.99, 2.23, 2.42]) * 1e37
_MUSH_INCL = float(np.log(_MUSH_L[-1] / _MUSH_L[8]) / np.log(_MUSH_E[-1] / _MUSH_E[8]))


def l_crit_mushtukov(e_cyc):
    if e_cyc <= _MUSH_E[-1]:
        return float(10 ** np.interp(np.log10(e_cyc), np.log10(_MUSH_E), np.log10(_MUSH_L)))
    return float(_MUSH_L[-1] * (e_cyc / _MUSH_E[-1]) ** _MUSH_INCL)


par = {x["fonte"]: x for x in csv.DictReader(open(os.path.join(AQUI, "dados", "parametros_fontes.csv")))}
QUAL = json.load(open(os.path.join(AQUI, "dados", "qualidade.json")))
lum = list(csv.DictReader(open(os.path.join(LIT, "luminosidades.csv"))))

linhas = []
for x in lum:
    f = x["fonte"]
    if f not in par or x["tipo"] not in TIPOS:
        continue
    L, du = num(x["L_erg_s"]), num(x["d_usada_kpc"])
    d = num(par[f]["d_kpc"])
    if not (L > 0):
        continue
    esc = (d / du) ** 2 if du > 0 else np.nan
    lc = l_crit_becker(num(par[f]["E_cyc_keV"]))
    linhas.append({**{k: x[k] for k in ("fonte", "tipo", "L_erg_s", "banda_L", "d_usada_kpc", "ref")},
                   "d_gaia_kpc": d,
                   # valor calculado por um modelo no paper, não uma transição observada
                   "teorico": ("teóric" in (x["erro_ou_faixa"] + x["banda_L"]).lower()
                               or x["ref"].startswith(("Becker+2012", "Tsygankov+2006")) and x["tipo"] == "L_crit_inferida"),
                   "L_gaia": L * esc if esc == esc else np.nan, "L_crit_becker": lc,
                   "ell_becker": L * esc / lc if esc == esc else np.nan,
                   "L_crit_mushtukov": l_crit_mushtukov(num(par[f]["E_cyc_keV"])),
                   "ell_mushtukov": L * esc / l_crit_mushtukov(num(par[f]["E_cyc_keV"])) if esc == esc else np.nan,
                   "obs": x.get("obs", ""), "h1": x.get("h1", "")})
with open(os.path.join(RES, "sintese", "transicoes_h1.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, list(linhas[0].keys()))
    w.writeheader()
    w.writerows(linhas)

fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for ax, chave, rot in ((axs[0], "ell_becker", "Becker+2012"), (axs[1], "ell_mushtukov", "Mushtukov+2015 (modo X)")):
    for i, t in enumerate(TIPOS):
        for x in linhas:
            if x["tipo"] != t or not (x[chave] > 0) or x["teorico"]:
                continue
            ax.plot(x[chave], i + 0.12 * (ORDEM.index(x["fonte"]) - 2), marker=MARC[x["fonte"]],
                    color=COR[x["fonte"]], ls="none", ms=6)
    ax.axvline(1, color="0.3", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(f"ℓ = L(d Gaia) / L_crit, {rot}")
for f in ORDEM:
    axs[1].plot([], [], marker=MARC[f], color=COR[f], ls="none", label=f)
axs[0].set_yticks(range(len(TIPOS)))
axs[0].set_yticklabels(["perfil muda", "mínimo do PF", "troca de ramo (HID)", "E_cyc–L inverte", "L_crit medida no paper"])
axs[1].legend(fontsize=6.5, loc="lower right")
fig.suptitle("H1: transições observadas (literatura, L reescalada ao Gaia) em unidades de L_crit(B_cyc); valores teóricos dos papers excluídos", fontsize=8.5)
fig.tight_layout()
fig.savefig(os.path.join(RES, "figuras", "h1_transicoes.png"), dpi=160)

# H2 pelo propeller: L_lim = 4e37 k^(7/2) B12^2 P^(-7/3) M1.4^(-2/3) R6^5 (Tsygankov+2016, eq. 2),
# com a L_lim bolométrica da literatura reescalada para o Gaia; k = 0,5 e 0,7 (faixa do paper).
# R6 = 1,2, o mesmo R = 12 km do torque e do redshift (com R6 = 1 o B sairia 1,2^2,5 = 1,58x maior).
R6 = 1.2
P_SPIN = {"4U 0115+63": 3.614, "V 0332+53": 4.376, "1A 0535+262": 103.3, "Swift J0243.6+6124": 9.79, "X Per": 837.7}
camp = []
for x in lum:
    f = x["fonte"]
    if x["tipo"] != "L_propeller" or f not in par or not x["banda_L"].strip().startswith("bolom"):
        continue
    L, du = num(x["L_erg_s"].lstrip("<=")), num(x["d_usada_kpc"])
    if not (L > 0 and du > 0):
        continue
    Lg = L * (num(par[f]["d_kpc"]) / du) ** 2
    bcyc = UM_Z * num(par[f]["E_cyc_keV"]) / 11.57 * 1e12
    linha = {"fonte": f, "L_lim_lit": L, "d_usada_kpc": du, "L_lim_gaia": Lg, "ref": x["ref"],
             "limite_superior": x["L_erg_s"].strip().startswith("<"), "B_cyc_G": bcyc}
    for k in (0.5, 0.7):
        b = np.sqrt(Lg / (4e37 * k ** 3.5 * P_SPIN[f] ** (-7 / 3) * R6 ** 5)) * 1e12
        linha[f"B_prop_k{k}_G"] = b
        linha[f"M_prop_k{k}"] = bcyc / b
    camp.append(linha)
with open(os.path.join(RES, "sintese", "campos_propeller.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, list(camp[0].keys()))
    w.writeheader()
    w.writerows(camp)

# ell de cada observação XMM: L_2-10 do fluxo desdobrado, e L_bol do MAXI no mesmo dia (K_bol = 2)
crab_taxa = CRAB_MAXI
MAXI = {"4U 0115+63": "J0118+637", "V 0332+53": "J0334+531", "1A 0535+262": "J0538+263",
        "Swift J0243.6+6124": "J0243+614", "X Per": "J0355+310"}
saida = []
for x in csv.DictReader(open(os.path.join(RES, "atlas", "catalogo_obs.csv"))):
    f = x["fonte"]
    if f not in MAXI or x["estado"] != "ok":
        continue
    d = num(par[f]["d_kpc"])
    L210 = 4 * np.pi * (d * KPC) ** 2 * num(x["fluxo_2-10"])
    m = ler_maxi(MAXI[f])
    mjd = num(x["mjd_meio"])
    s = np.abs(m[:, 0] - mjd) <= 1.0
    lbol = np.nan
    snr = np.nan
    if s.any():
        taxa = np.mean(m[s, 1])
        snr = taxa / np.sqrt(np.sum(m[s, 2] ** 2)) * s.sum()
        if snr > 3:
            lbol = 2.0 * 3.0e-8 * taxa / crab_taxa * 4 * np.pi * (d * KPC) ** 2
    lc = l_crit_becker(num(par[f]["E_cyc_keV"]))
    if x["obsid"] in QUAL:
        L210 = np.nan
    saida.append({"fonte": f, "obsid": x["obsid"], "mjd": mjd, "L_2_10_xmm": L210,
                  "qualidade": QUAL.get(x["obsid"], ""),
                  "L_bol_maxi": lbol, "snr_maxi": snr, "L_crit_becker": lc,
                  "ell_xmm_Kbol4": 4 * L210 / lc if L210 > 0 else np.nan,
                  "ell_maxi": lbol / lc,
                  "L_crit_mushtukov": l_crit_mushtukov(num(par[f]["E_cyc_keV"])),
                  "ell_maxi_mushtukov": lbol / l_crit_mushtukov(num(par[f]["E_cyc_keV"]))})
with open(os.path.join(RES, "sintese", "atlas_ell.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, list(saida[0].keys()))
    w.writeheader()
    w.writerows(saida)
print(len(linhas), "transições;", len(saida), "observações com ell")
