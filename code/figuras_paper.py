"""Figuras do paper, em PDF. Uso: python figuras_paper.py <resultados> <saida>

Lê <resultados>/{atlas,torque,sintese} e escreve em <saida>: fig_torque.pdf, fig_D.pdf, fig_ell.pdf,
fig_profiles.pdf, fig_harmonics.pdf. Imprime os números derivados que o texto cita.
"""
import csv
import glob
import json
import os
import sys

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from torque import GBM, XI, G, R, M as M_NS, n_gl, nudot_modelo  # noqa: E402

RES, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
ORDEM = ["4U 0115+63", "V 0332+53", "1A 0535+262", "Swift J0243.6+6124", "X Per"]
COR = dict(zip(ORDEM, ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]))
MARC = dict(zip(ORDEM, ["o", "s", "^", "D", "v"]))
KPC = 3.0857e21
UM_Z = 1.24
COL, LARG = 3.31, 6.85     # largura de coluna e de página do MNRAS, em polegadas
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight", "pdf.fonttype": 42})


def num(x):
    try:
        return float(str(x).replace("~", ""))
    except (TypeError, ValueError):
        return np.nan


_W = np.linspace(0.0, 0.9, 9001)
W_C = float(_W[np.argmax(n_gl(_W) <= 0)])     # fastness crítica: n(w_c) = 0


def curva(mu, L, nu):
    """nu-ponto de Ghosh-Lamb só onde a fórmula de n(w) vale (w < w_c); fora dela, nan."""
    mdot = L * R / (G * M_NS)
    rm = XI * (mu ** 4 / (2 * G * M_NS * mdot ** 2)) ** (1 / 7)
    w = (rm / (G * M_NS / (2 * np.pi * nu) ** 2) ** (1 / 3)) ** 1.5
    m = nudot_modelo(mu, L, nu)
    m[(w >= W_C) | (m <= 0)] = np.nan
    return m


def l_crit_becker(e_cyc):
    return 1.49e37 * (UM_Z * e_cyc / 11.57) ** (16 / 15)


par = {x["fonte"]: x for x in csv.DictReader(open(os.path.join(AQUI, "dados", "parametros_fontes.csv")))}
prop = list(csv.DictReader(open(os.path.join(RES, "sintese", "campos_propeller.csv"))))
trans = list(csv.DictReader(open(os.path.join(RES, "sintese", "transicoes_h1.csv"))))
ell_obs = {x["obsid"]: x for x in csv.DictReader(open(os.path.join(RES, "sintese", "atlas_ell.csv")))}
cat = list(csv.DictReader(open(os.path.join(RES, "atlas", "catalogo_obs.csv"))))
js = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(os.path.join(RES, "atlas", "obs", "*.json"))}
QUAL = json.load(open(os.path.join(AQUI, "dados", "qualidade.json")))
COR_ATLAS = json.load(open(os.path.join(RES, "atlas", "correcoes.json")))["obs"]
CURTO = {"4U 0115+63": "4U 0115+63", "V 0332+53": "V 0332+53", "1A 0535+262": "1A 0535+262",
         "Swift J0243.6+6124": "Swift J0243.6+6124", "X Per": "X Per"}

# ---------------- 1. torque ----------------
from torque_ajuste import dados, modelo as modelo_aj, n_gl as n_gl_aj  # noqa: E402
aj = list(csv.DictReader(open(os.path.join(RES, "torque", "ajuste_global.csv"))))
rev = {x["fonte"]: x for x in csv.DictReader(open(os.path.join(RES, "torque", "reversao.csv")))}
IND = json.load(open(os.path.join(RES, "sintese", "indice.json")))
fs = ORDEM[:4]
fig, ax = plt.subplots(1, len(fs), figsize=(LARG, 2.0))
for a, f in zip(ax, fs):
    d = dados(f, num(par[f]["d_kpc"]), 0.0, 3.0)
    L, nd = d["L"], d["nd"]
    alto = L >= 0.3 * L.max()
    a.plot(L[~alto], nd[~alto], marker=MARC[f], mfc="none", mec=COR[f], ls="none", ms=2.5, mew=0.6, alpha=0.7)
    a.plot(L[alto], nd[alto], marker=MARC[f], color=COR[f], ls="none", ms=2.8)
    nu = np.median(d["nu"])
    Lg = np.logspace(np.log10(L.min()), np.log10(L.max()), 300)
    for ajuste, est in (("A", "-"), ("B", "--")):
        cand = [x for x in aj if x["fonte"] == f and x["torque"] == "GL79" and x["ajuste"] == ajuste]
        cand = [x for x in cand if x["na_borda"] == "False"] or cand
        x = min(cand, key=lambda x: num(x["m2lnL"]))
        LL = Lg[Lg >= 0.3 * L.max()] if ajuste == "A" else Lg
        a.plot(LL, modelo_aj(num(x["log_mu"]), LL, nu, n_gl_aj, num(x["lam"])), color="0.15", ls=est, lw=0.9)
    t2 = IND["fontes"][f].get("T2_GL79")
    if t2:
        a.axvspan(*t2["L_eq_faixa"], color="0.75", lw=0, zorder=0, alpha=0.45 if t2["marginal"] else 1.0,
                  hatch="////" if t2["marginal"] else None)
    a.axhline(0, color="0.6", lw=0.5)
    a.set_xscale("log")
    a.set_yscale("symlog", linthresh=1e-12, linscale=0.6)
    a.set_title(CURTO[f], fontsize=7.5)
    a.set_xlabel("$L$ (erg s$^{-1}$)")
    a.set_ylim(-1e-11, nd.max() * 3)
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax[0].set_ylabel("$\\dot\\nu$ (Hz s$^{-1}$)")
ax[0].plot([], [], color="0.15", ls="-", lw=0.9, label="fit A ($L\\geq0.3L_{\\max}$, $K_{\\rm bol}=2$)")
ax[0].plot([], [], color="0.15", ls="--", lw=0.9, label="fit B (all, $\\lambda$ free)")
ax[0].fill_between([], [], color="0.75", label="$L_{\\rm eq}$ (torque reversal)")
ax[0].plot([], [], marker="o", color="0.4", ls="none", ms=3, label="$L\\geq0.3L_{\\max}$")
ax[0].plot([], [], marker="o", mfc="none", mec="0.4", ls="none", ms=3, label="$L<0.3L_{\\max}$")
fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=5, frameon=False, fontsize=6,
           bbox_to_anchor=(0.5, 1.07))
fig.tight_layout(w_pad=0.5)
fig.savefig(os.path.join(OUT, "fig_torque.pdf"))
plt.close(fig)

# ---------------- 2. índice D por estimador ----------------
LINHAS = [("P_Tsygankov2016", "propeller onset"), ("P_Tsygankov2019", "accretion at 5e34 (P)"),
          ("P_Tsygankov2018", "no propeller (P)"), ("P_Doroshenko2020", "no propeller (P)"),
          ("T2_GL79", "torque reversal, GL79"), ("T2_lin", "torque reversal, linear"),
          ("T1_GL79", "spin-up, GL79"), ("T1_lin", "spin-up, linear")]
ROT_EST = {"P_Tsygankov2016": "P: onset (T16)", "P_Tsygankov2019": "P: accretes (T19)",
           "P_Tsygankov2018": "P: no onset (T18)", "P_Doroshenko2020": "P: no onset (D20)",
           "T2_GL79": "T2: reversal, GL", "T2_lin": "T2: reversal, lin.", "T1_GL79": "T1: spin-up, GL",
           "T1_lin": "T1: spin-up, lin."}
fig, a = plt.subplots(figsize=(COL, 3.6))
y, ticks, labs = 0, [], []
for f in ORDEM[:4]:
    o = IND["fontes"][f]
    ks = [k for k, _ in LINHAS if k in o]
    a.text(0.022, y + 0.75, CURTO[f], ha="left", va="center", fontsize=6.5, color=COR[f], fontweight="bold")
    for k in ks:
        v = o[k]["D"]
        lim = o[k].get("limite", False)
        if lim:
            # limite inferior de 95 por cento (q05), como no texto e na tabela
            a.annotate("", xy=(v["q05"] * 3, y), xytext=(v["q05"], y),
                       arrowprops=dict(arrowstyle="-|>", color=COR[f], lw=1.0, mutation_scale=6))
            a.plot(v["q05"], y, marker="|", color=COR[f], ms=6)
        else:
            a.errorbar(v["med"], y, xerr=[[v["med"] - v["q16"]], [v["q84"] - v["med"]]], marker=MARC[f],
                       color=COR[f], ms=3.5, capsize=0, lw=1.0, mfc=COR[f] if k.startswith(("P", "T2")) else "white",
                       alpha=0.5 if o[k].get("marginal") else 1.0)
        ticks.append(y)
        labs.append(ROT_EST[k] + (" (marg.)" if o[k].get("marginal") else "")
                    + (f" ({100 * o[k]['frac_interior']:.0f}%)" if "frac_interior" in o[k] else ""))
        y -= 1
    y -= 1.2
a.axvline(1, color="0.5", lw=0.7, ls="--")
a.set_xscale("log")
a.set_xlim(0.02, 200)
a.set_yticks(ticks)
a.set_yticklabels(labs, fontsize=5.8)
a.tick_params(axis="y", length=0)
a.set_xlabel("$\\mathcal{D} = B_{\\rm cyc}/B_{\\rm dip}$")
a.spines["left"].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_D.pdf"))
plt.close(fig)

# ---------------- 3. ell das transições ----------------
M_ADOT = {"4U 0115+63": IND["fontes"]["4U 0115+63"]["P_Tsygankov2016"]["D"]["med"],
          "V 0332+53": IND["fontes"]["V 0332+53"]["P_Tsygankov2016"]["D"]["med"],
          "1A 0535+262": IND["fontes"]["1A 0535+262"]["T2_GL79"]["D"]["med"],
          "Swift J0243.6+6124": IND["fontes"]["Swift J0243.6+6124"]["T2_GL79"]["D"]["med"]}
TIPOS = ["transicao_perfil", "minimo_PF", "troca_ramo_HID", "inversao_Ecyc_L"]
ROT = ["pulse profile changes", "pulsed-fraction minimum", "HID branch change", "$E_{\\rm cyc}$–$L$ slope reverses"]
usar = [x for x in trans if x["tipo"] in TIPOS and x["teorico"] == "False" and num(x["ell_becker"]) > 0]
fig, ax = plt.subplots(1, 3, figsize=(LARG, 1.9), sharey=True)
faixas = {}
for k, (a, rot) in enumerate(zip(ax, ["$L_{\\rm crit}(B_{\\rm cyc})$, Becker et al. (2012)",
                                       "$L_{\\rm crit}(B_{\\rm cyc})$, Mushtukov et al. (2015)",
                                       "$L_{\\rm crit}(B_{\\rm dip})$, Becker et al. (2012)"])):
    for x in usar:
        f = x["fonte"]
        if k == 0:
            v = num(x["ell_becker"])
        elif k == 1:
            v = num(x["ell_mushtukov"])
        else:
            v = num(x["ell_becker"]) * M_ADOT[f] ** (16 / 15)
        faixas.setdefault((k, x["tipo"]), []).append(v)
        i = TIPOS.index(x["tipo"])
        limpa = x.get("h1", "sim") == "sim"
        a.plot(v, i + 0.13 * (ORDEM.index(f) - 1.5), marker=MARC[f], color=COR[f], ls="none", ms=3.6,
               mfc=COR[f] if limpa else "none", alpha=1.0 if limpa else 0.6)
    a.axvline(1, color="0.5", lw=0.7, ls="--")
    a.set_xscale("log")
    a.set_xlim(0.03, 40)
    a.set_xlabel("$\\ell = L/L_{\\rm crit}$")
    a.set_title(rot, fontsize=7)
ax[0].set_yticks(range(len(TIPOS)))
ax[0].set_yticklabels(ROT, fontsize=6.5)
for f in ORDEM[:4]:
    ax[2].plot([], [], marker=MARC[f], color=COR[f], ls="none", ms=3.6, label=CURTO[f])
fig.legend(*ax[2].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False, fontsize=6,
           bbox_to_anchor=(0.6, 1.08))
fig.tight_layout(w_pad=0.5)
fig.savefig(os.path.join(OUT, "fig_ell.pdf"))
plt.close(fig)

# ---------------- 4. perfis ----------------
pul = [x for x in cat if x["estado"] == "ok" and x["fonte"] in ORDEM
       and (js[x["obsid"]].get("tempo") or {}).get("detectado") and x["obsid"] not in QUAL]
for x in pul:
    e = ell_obs.get(x["obsid"], {})
    x["ell"] = num(e.get("ell_xmm_Kbol4"))
    x["L210"] = num(e.get("L_2_10_xmm"))
pul.sort(key=lambda x: (ORDEM.index(x["fonte"]), np.nan_to_num(x["L210"], nan=0)))
ncol = 4
nlin = int(np.ceil(len(pul) / ncol))
fig, ax = plt.subplots(nlin, ncol, figsize=(LARG, 1.35 * nlin), sharex=True)
for a in ax.flat:
    a.set_visible(False)
for a, x in zip(ax.flat, pul):
    a.set_visible(True)
    z = np.load(os.path.join(RES, "atlas", "obs", f"{x['obsid']}_perfis.npz"))
    for banda, cor, rot in (("0.3-2", "#0072B2", "0.3–2 keV"), ("2-10", "#D55E00", "2–10 keV")):
        if f"{banda}__contagens" not in z:
            continue
        c, fe = z[f"{banda}__contagens"], z[f"{banda}__fracao_exposicao"]
        taxa = c / fe - float(z[f"{banda}__fundo_total"])
        if taxa.mean() <= 0:
            continue
        p, e = taxa / taxa.mean(), np.sqrt(c) / fe / taxa.mean()
        fi = (np.arange(c.size) + 0.5) / c.size
        a.step(np.r_[fi, fi + 1], np.r_[p, p], where="mid", color=cor, lw=0.8, label=rot)
        a.fill_between(np.r_[fi, fi + 1], np.r_[p - e, p - e], np.r_[p + e, p + e], step="mid", color=cor,
                       alpha=0.18, lw=0)
    a.axhline(1, color="0.6", lw=0.5)
    tit = f"{CURTO[x['fonte']]} {x['obsid']}"
    if x["L210"] == x["L210"]:
        tit += f"\n$L_{{2-10}}$ = {x['L210']:.1e}".replace("e+", "e")
    a.set_title(tit, fontsize=6)
    a.tick_params(labelsize=6)
for a in ax[-1]:
    a.set_xlabel("pulse phase")
for lin in ax:
    lin[0].set_ylabel("norm. rate", fontsize=6.5)
ax.flat[0].legend(fontsize=5, loc="upper right", frameon=False)
fig.tight_layout(h_pad=0.3, w_pad=0.4)
fig.savefig(os.path.join(OUT, "fig_profiles.pdf"))
plt.close(fig)

# ---------------- 5. PF e r2/r1 contra ell ----------------
fig, ax = plt.subplots(2, 1, figsize=(COL, 3.3), sharex=True)
for f in ORDEM[:4]:
    o = [x for x in pul if x["fonte"] == f and x["ell"] == x["ell"]]
    if not o:
        continue
    ell = np.array([x["ell"] for x in o])
    # PF_rms com o viés de ruído descontado da soma (atlas_correcoes.py)
    pf = np.array([COR_ATLAS.get(x["obsid"], {}).get("pf_rms", num(x["pf_rms_0.3-10"])) for x in o])
    spf = np.array([num(x["sig_pf_rms_0.3-10"]) for x in o])
    r = np.array([num(x["r2"]) / num(x["r1"]) for x in o])
    lab = CURTO[f]
    ax[0].errorbar(ell, pf, spf, marker=MARC[f], color=COR[f], ls="none", ms=4, capsize=0, lw=0.8, label=lab)
    ax[1].plot(ell, r, marker=MARC[f], color=COR[f], ls="none", ms=4)
for x in cat:
    j = js[x["obsid"]]
    ul = COR_ATLAS.get(x["obsid"], {}).get("pf_rms_lim_sup_3sigma")
    ul = ul if ul and ul < 1 else None
    e = num(ell_obs.get(x["obsid"], {}).get("ell_xmm_Kbol4"))
    if ul and x["fonte"] in ORDEM[:4] and e == e:
        ax[0].errorbar([e], [ul], yerr=[[0.06]], uplims=[True], color=COR[x["fonte"]], marker=MARC[x["fonte"]],
                       mfc="none", ls="none", ms=4, lw=0.8)
ax[1].axvline(1, color="0.5", lw=0.7, ls="--")
ax[0].axvline(1, color="0.5", lw=0.7, ls="--")
ax[0].set_ylabel("PF$_{\\rm rms}$ (0.3–10 keV)")
ax[1].set_ylabel("$r_2/r_1$ (0.3–10 keV)")
ax[1].set_xlabel("$\\ell = 4L_{2-10}/L_{\\rm crit}(B_{\\rm cyc})$")
ax[1].set_xscale("log")
ax[1].set_ylim(0, 1.1)
ax[0].legend(fontsize=5.5, frameon=False, loc="upper right")
fig.tight_layout(h_pad=0.4)
fig.savefig(os.path.join(OUT, "fig_harmonics.pdf"))
plt.close(fig)

# ---------------- números citados no texto ----------------
print("D usado no H1 com B_dip (propeller na 4U 0115 e V 0332; reversão GL79 na 1A 0535 e Swift):",
      {k: round(v, 2) for k, v in M_ADOT.items()})
for k, nome in ((0, "Becker B_cyc"), (1, "Mushtukov B_cyc"), (2, "Becker B_dip")):
    for t in TIPOS:
        v = faixas.get((k, t), [])
        if v:
            print(f"{nome:16s} {t:18s} n={len(v):2d} min={min(v):.3g} max={max(v):.3g} razao={max(v) / min(v):.1f}")
    todos = [u for t in TIPOS for u in faixas.get((k, t), [])]
    print(f"{nome:16s} {'TODOS':18s} n={len(todos):2d} dex={np.log10(max(todos) / min(todos)):.2f} "
          f"sd(log)={np.std(np.log10(todos)):.2f}")
for x in pul:
    print(x["fonte"], x["obsid"], "ell=%.3g PF=%.3f r2/r1=%.2f" % (x["ell"], num(x["pf_rms_0.3-10"]), num(x["r2"]) / num(x["r1"])))
