"""Tabelas LaTeX do paper. Uso: python tabelas_paper.py <resultados> <saida>

Escreve em <saida>: tab_torque.tex (ajustes de verossimilhança do torque), tab_index.tex (índice D por
estimador), tab_h1.tex (componentes da variância das transições), tab_xmm.tex (as observações XMM das cinco
acretoras), tab_orb.tex (efemérides orbitais), tab_leq.tex (robustez da luminosidade de reversão) e
tab_loso.tex (teste deixando uma fonte de fora). Os números vêm de <resultados>/{torque,sintese,atlas}.
"""
import csv
import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RES, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
ORDEM = ["4U 0115+63", "V 0332+53", "1A 0535+262", "Swift J0243.6+6124", "X Per"]
QUAL = json.load(open(os.path.join(AQUI, "dados", "qualidade.json")))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def sci(x, d=1):
    """1.7e12 -> $1.7\\times10^{12}$"""
    if not np.isfinite(x):
        return "--"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    if round(m, d) >= 10:
        m, e = m / 10, e + 1
    return f"${m:.{d}f}\\times10^{{{e}}}$"


def fx(x, d=2):
    return "--" if not np.isfinite(x) else f"{x:.{d}f}"


KPC = 3.0857e21
INVALIDO = {"0811212801"}   # fluxo e PF não representativos (fonte fora da janela de timing)
par = {x["fonte"]: x for x in csv.DictReader(open(os.path.join(AQUI, "dados", "parametros_fontes.csv")))}
sys.path.insert(0, AQUI)
from indice import E_FAIXA, K_ALTO, K_BAIXO  # noqa: E402


def l_crit_becker(e_cyc):
    return 1.49e37 * (1.24 * e_cyc / 11.57) ** (16 / 15)


def faixa(v, d=2, sci_=False):
    f = (lambda x: sci(x, 1)) if sci_ else (lambda x: f"{x:.{d}f}")
    return f"{f(v['med'])} ({f(v['q16'])}--{f(v['q84'])})" if not sci_ else f"{f(v['med'])}"


def dfmt(x):
    """D com 2 algarismos significativos."""
    return f"{x:#.2g}".rstrip(".") if x < 10 else f"{x:.0f}"


# ---------------- ajustes de torque ----------------
aj = list(csv.DictReader(open(os.path.join(RES, "torque", "ajuste_global.csv"))))
NOME_N = {"GL79": "GL79", "lin": "linear"}
T = [r"\begin{table*}",
     r"\centering",
     r"\small",
     r"\caption{Global likelihood fits of the spin-up rate (equation~\ref{eq:like}). Fit A uses the points with "
     r"$L\geq0.3L_{\max}$ and the nominal luminosity scale ($\Kbol=2$, Gaia distance, $\lambda=1$); fit B uses all "
     r"points with a MAXI detection at $\geq3\sigma$ and leaves $\lambda$ free. For each fit both branches are "
     r"given, split at the $\mu$ that maximizes $\nudot$ at the median luminosity; $\Delta_{\rm high}$ is $-2\ln\mathcal{L}$ "
     r"of the high-$\mu$ minus that of the low-$\mu$ branch (negative: high $\mu$ preferred). Fields assume "
     r"$\xi=0.5$, $M=1.4\,{\rm M}_\odot$, $R=12$\,km and $I=10^{45}$\,g\,cm$^2$. "
     r"$^{\rm e}$Minimum at the boundary of its branch or of the grid in $\mu$ or $\lambda$ (no interior "
     r"solution). The linear torque spins the star down for $\omega>1$.}",
     r"\label{tab:torque}",
     r"\begin{tabular}{llccccccccc}",
     r"\hline",
     r" & & \multicolumn{4}{c}{Fit A ($L\geq0.3L_{\max}$, $\lambda=1$)} & \multicolumn{5}{c}{Fit B (all points, $\lambda$ free)} \\",
     r"Source & $n(\omega)$ & $N$ & $\Bdip$ low $\mu$ (G) & $\Bdip$ high $\mu$ (G) & $\Delta_{\rm high}$ & "
     r"$N$ & $\Bdip$ low $\mu$ (G) & $\Bdip$ high $\mu$ (G) & $\lambda$ (pref.) & $\Delta_{\rm high}$ \\",
     r"\hline"]


def celula(x):
    if x is None:
        return "--"
    return sci(num(x["B_dip_G"])) + (r"$^{\rm e}$" if x["na_borda"] == "True" else "")


for f in ORDEM[:4]:
    for k, n in enumerate(("GL79", "lin")):
        lin = [f if k == 0 else "", NOME_N[n]]
        for ajuste in ("A", "B"):
            xs = {x["ramo"]: x for x in aj if x["fonte"] == f and x["torque"] == n and x["ajuste"] == ajuste}
            b, h = xs.get("baixo-mu"), xs.get("alto-mu")
            N = (b or h)["N"]
            dl = num(h["m2lnL"]) - num(b["m2lnL"]) if b and h else np.nan
            if ajuste == "A":
                lin += [N, celula(b), celula(h), f"{dl:+.1f}"]
            else:
                pref = min([y for y in (b, h) if y], key=lambda y: num(y["m2lnL"]))
                lin += [N, celula(b), celula(h), f"{num(pref['lam']):.2f}", f"{dl:+.1f}"]
        T.append(" & ".join(map(str, lin)) + r" \\")
T += [r"\hline", r"\end{tabular}", r"\end{table*}"]
open(os.path.join(OUT, "tab_torque.tex"), "w").write("\n".join(T) + "\n")

# ---------------- índice D por estimador ----------------
IND = json.load(open(os.path.join(RES, "sintese", "indice.json")))
rev = {x["fonte"]: x for x in csv.DictReader(open(os.path.join(RES, "torque", "reversao.csv")))}


def dcel(o, k):
    if k not in o:
        return "--"
    v = o[k]["D"]
    if o[k].get("limite"):
        return f"$>{dfmt(v['q05'])}$"
    cel = f"{dfmt(v['med'])} ({dfmt(v['q16'])}--{dfmt(v['q84'])})" + ("$^\\dagger$" if o[k].get("marginal") else "")
    if "frac_interior" in o[k]:
        # a fração de sorteios com solução interior vai numa segunda linha da célula (largura da página)
        return f"\\twoline{{{cel}}}{{[{100 * o[k]['frac_interior']:.0f}\\%]}}"
    return cel


_fs = IND["fontes"]["Swift J0243.6+6124"]["T2_GL79"]["L_eq_faixa"]
FX_SWIFT = f"{_fs[0] / 1e37:.1f}--${_fs[1] / 1e37:.2f}\\times10^{{37}}\\,\\ergs$"
X = [r"\begin{table*}",
     r"\centering",
     r"\small",
     r"\setlength{\tabcolsep}{3pt}",
     r"\newcommand{\twoline}[2]{\begin{tabular}[t]{@{}c@{}}#1\\\relax#2\end{tabular}}",
     r"\caption{Field-discrepancy index $\mathcal{D}=\Bcyc/\Bdip$ by estimator: median and 16--84 per cent range "
     r"of the Monte Carlo over the systematics (Section~\ref{sec:syst}); lower limits are 95 per cent. "
     r"$\Ecyc$: range of the measured fundamental; $\Kbol$: adopted range at high (T1) and low (T2) "
     r"luminosity, containing the values calibrated against published broad-band fluxes "
     r"(Section~\ref{sec:kbol}). P: propeller (\citealt{Tsygankov2016} for "
     r"4U~0115+63 and V~0332+53; for 1A~0535+262 the pulsating state at $5.5\times10^{34}\,\ergs$ of "
     r"\citealt{Tsygankov2019}; for \swj{} the absence of a transition, \citealt{Tsygankov2018} and "
     r"\citealt{Doroshenko2020}). T2: torque reversal at $L_{\rm eq}$ with $\omega_{\rm c}=0.35$ (GL79) or 1 "
     r"(linear); the $\Leq$ range sampled is the envelope of the robustness variants with $\Delta\geq4$ "
     r"(Table~\ref{tab:leq}); 4U~0115+63 shows no reversal. The $\Leq$ column is the fit to all outbursts "
     r"together; in \swj{} the reversal luminosity varies by a factor of about 3 between outbursts, and the "
     rf"sampled range, {FX_SWIFT}, includes that variation. "
     r"T1: high-luminosity spin-up (fit A, slow-rotator branch), refitted in each draw; in brackets, the "
     r"fraction of draws in which that branch has an interior solution, and $\mathcal{D}$ among those draws.}",
     r"\label{tab:index}",
     r"\begin{tabular}{lcccccccc}",
     r"\hline",
     r"Source & $\Ecyc$ & $\Kbol$ & $L_{\rm eq}$ & $\mathcal{D}_{\rm P}$ & \multicolumn{2}{c}{$\mathcal{D}_{\rm T2}$} & "
     r"\multicolumn{2}{c}{$\mathcal{D}_{\rm T1}$} \\",
     r" & (keV) & high/low $L$ & ($\ergs$) & & GL79 & linear & GL79 & linear \\",
     r"\hline"]
def ecel(x):
    return f"{x:.1f}" if x < 100 else f"{x:.0f}"


for f in ORDEM[:4]:
    o = IND["fontes"][f]
    e = E_FAIXA[f]
    kb = f"{K_ALTO[f][0]:.1f}--{K_ALTO[f][1]:.1f}" + (f" / {K_BAIXO[f][0]:.1f}--{K_BAIXO[f][1]:.1f}" if f in K_BAIXO else "")
    t2 = o.get("T2_GL79")
    leq = (sci(t2["L_eq"]) + ("$^\\dagger$" if t2["marginal"] else "")) if t2 else "--"
    ps = [k for k in ("P_Tsygankov2016", "P_Tsygankov2019", "P_Tsygankov2018", "P_Doroshenko2020") if k in o]
    pcel = "; ".join(dcel(o, k) for k in ps) or "--"
    X.append(f"{f} & {ecel(e[0])}--{ecel(e[1])} & {kb} & {leq} & {pcel} & {dcel(o, 'T2_GL79')} & {dcel(o, 'T2_lin')} & "
             f"{dcel(o, 'T1_GL79')} & {dcel(o, 'T1_lin')} \\\\")
raz = IND["razoes"]
X += [r"\hline", r"\end{tabular}", r"\end{table*}"]
open(os.path.join(OUT, "tab_index.tex"), "w").write("\n".join(X) + "\n")

# ---------------- H1: componentes da variância ----------------
H = json.load(open(os.path.join(RES, "sintese", "h1_hierarquico.json")))
ROTH = {"L": r"$\log L$", "becker": r"$\log\ell$, \citet{Becker2012}, $\Bcyc$",
        "mushtukov": r"$\log\ell$, \citet{Mushtukov2015}, $\Ecyc$", "dipolo": r"$\log\ell$, \citet{Becker2012}, $\Bdip$"}
ROTA = {"limpa": "Clean sample (adopted)", "sem_duplicatas": "Without duplicates", "todas": "All published values"}


def sg(v):
    return f"{v['med']:.1f} ({v['q16']:.1f}--{v['q84']:.1f})"


def pf_(p):
    return "1" if p >= 0.995 else (f"{p:.3f}" if p < 0.1 else f"{p:.2f}")


Hh = [r"\begin{table*}",
      r"\centering",
      r"\small",
      r"\caption{Variance components of the published transition luminosities (dex; posterior median and "
      r"16--84 per cent, uniform priors in 0--2\,dex). The clean sample excludes duplicates and values that do "
      r"not mark the onset of the column in the given source (Section~\ref{sec:ell}). $\chi^2_\nu$: reduced "
      r"$\chi^2$ about a single mean with the measurement errors alone. $p$: likelihood-ratio test of "
      r"$\sigma_{\rm source}=0$, for the luminosities as published and with narrow-band values raised by "
      r"0.2\,dex. $\Bdip$ is the propeller field for 4U~0115+63 and V~0332+53 and the torque-reversal field "
      r"(GL79) for 1A~0535+262 and \swj{}.}",
      r"\label{tab:h1}",
      r"\begin{tabular}{lcccccc}",
      r"\hline",
      r"Quantity & $N$ (sources) & $\chi^2_\nu$ & $\sigma_{\rm source}$ & $\sigma_{\rm study}$ & "
      r"$\sigma_{\rm indicator}$ & $p$ (published / band-corrected) \\",
      r"\hline"]
for am in ("limpa", "sem_duplicatas", "todas"):
    r0 = H[am]["L"]
    Hh.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{ROTA[am]}: {r0['N']} transitions, {r0['n_fontes']} sources}}}} \\")
    for k in ("L", "becker", "mushtukov", "dipolo"):
        v, w = H[am][k], H[am][k + "_banda_corrigida"]
        Hh.append(f"{ROTH[k]} & {v['N']} ({v['n_fontes']}) & {v['chi2_reduzido_media_unica']:.1f} & "
                  f"{sg(v['sigma_fonte'])} & {sg(v['sigma_trabalho'])} & {sg(v['sigma_indicador'])} & "
                  f"{pf_(v['p_LRT_mistura'])} / {pf_(w['p_LRT_mistura'])} \\\\")
Hh += [r"\hline", r"\end{tabular}", r"\end{table*}"]
open(os.path.join(OUT, "tab_h1.tex"), "w").write("\n".join(Hh) + "\n")

# ---------------- XMM ----------------
cat = list(csv.DictReader(open(os.path.join(RES, "atlas", "catalogo_obs.csv"))))
# limites a 3 sigma, tentativas, detecções marginais e PF_rms (atlas_correcoes.py)
COR = json.load(open(os.path.join(RES, "atlas", "correcoes.json")))["obs"]
ell = {x["obsid"]: x for x in csv.DictReader(open(os.path.join(RES, "sintese", "atlas_ell.csv")))}
MODO = {"IMAGING": "I", "TIMING": "T", "BURST": "B"}
EST = {"sem lista de eventos do pn": "no pn science exposure",
       "filtro fechado (CalClosed): exposição sem fonte": "pn filter closed",
       "fonte não detectada na faixa de timing": "source outside timing window"}
T = [r"\begin{table*}",
     r"\centering",
     r"\caption{XMM-Newton EPIC-pn observations of the five accreting pulsars. Mode: I = imaging, T = timing. "
     r"$t_{\rm live}$: live time after flare screening. $L_{2-10}$: 2--10\,keV luminosity from the "
     r"ARF-unfolded (absorbed) flux at the Gaia distance; $\ell=4L_{2-10}/L_{\rm crit}$ with the \citet{Becker2012} "
     r"$L_{\rm crit}(B_{\rm cyc})$. $p_H$: H-test chance probability corrected for $4\Delta\nu T$ independent "
     r"trial frequencies (one trial when the frequency is fixed). PF$_{\rm rms}$ (five harmonics) and $r_2/r_1$ "
     r"refer to 0.3--10\,keV; for non-detections PF$_{\rm rms}$ is a 3$\sigma$ upper limit on the first two "
     r"harmonics from the non-central $\chi^2$ distribution of $Z^2_2$, and -- means that it exceeds 1. "
     r"$^\ddagger$Marginal: $p_H=3\times10^{-4}$ at a fixed frequency, $1.5\times10^{-3}$ after correcting for "
     r"the five 2023 observations folded at a fixed frequency. X~Per is piled up; its profiles come from an "
     r"annulus and no flux is quoted.}",
     r"\label{tab:xmm}",
     r"\begin{tabular}{llcccccccc}",
     r"\hline",
     r"Source & ObsID & Mode & MJD & $t_{\rm live}$ (ks) & $L_{2-10}$ (erg\,s$^{-1}$) & $\ell$ & $p_H$ & "
     r"PF$_{\rm rms}$ & $r_2/r_1$ \\",
     r"\hline"]
for f in ORDEM:
    for x in sorted([y for y in cat if y["fonte"] == f], key=lambda y: num(y["mjd_meio"]) if y["mjd_meio"] else 0):
        o = x["obsid"]
        if x["estado"] != "ok":
            T.append(f"{f} & {o} & \\multicolumn{{8}}{{l}}{{\\textit{{not used: {EST.get(x['estado'], x['estado'])}}}}} \\\\")
            continue
        j = json.load(open(os.path.join(RES, "atlas", "obs", f"{o}.json")))
        det = (j.get("tempo") or {}).get("detectado")
        fl = num(x["fluxo_2-10"])
        L210 = 4 * np.pi * (num(par[f]["d_kpc"]) * KPC) ** 2 * fl if fl > 0 and o not in INVALIDO else np.nan
        el = 4 * L210 / l_crit_becker(num(par[f]["E_cyc_keV"]))
        cr = COR[o]
        ph = cr["p_tentativas"]
        det = cr["detectado"]
        phs = "$<10^{-10}$" if ph < 1e-10 else (sci(ph, 0) if ph < 0.01 else f"{ph:.2f}")
        if det and o in INVALIDO:
            pf = rr = "--"
        elif det:
            pf = f"{cr.get('pf_rms', num(x['pf_rms_0.3-10'])):.3f}"     # PF_rms de atlas_correcoes.py
            rr = f"{num(x['r2']) / num(x['r1']):.2f}"
        else:
            ul = cr.get("pf_rms_lim_sup_3sigma")
            pf = f"$<${ul:.2f}" if ul and ul < 1 else "--"
            rr = "--"
        nota = (r"$^\dagger$" if o in QUAL else "") + (r"$^\ddagger$" if cr.get("marginal") else "")
        T.append(f"{f} & {o}{nota} & {MODO.get(x['datamode'], x['datamode'])} & {num(x['mjd_meio']):.1f} & "
                 f"{num(x['expo_viva_s']) / 1e3:.1f} & {sci(L210)} & {(fx(el, 2) if el >= 0.01 else sci(el, 1))} & {phs} & {pf} & {rr} \\\\")
T += [r"\hline",
      r"\end{tabular}",
      r"\begin{minipage}{\textwidth}\footnotesize",
      r"$^\dagger$ " + "; ".join(
          f"{o}: " + {"0811212801": "the ToO was pointed at a provisional position 3.3$'$ from the source, so the source "
                                    "lies outside the timing-mode strip and only the wings of its PSF reach the "
                                    "readout. We extract the 21 columns centred on the brightest one (RAWX 49) "
                                    "against 29 background columns. The pulsation is detected, but the flux is "
                                    "450 times below MAXI on the same day, so the flux and the pulsed fraction "
                                    "are not used",
                      "0505280101": "no pulsation detected",
                      "0506190101": "source in quiescence, 57 net counts",
                      "0092820801": ""}.get(o, "")
          for o in QUAL if any(y["obsid"] == o and y["fonte"] in ORDEM for y in cat)) + ".",
      r"\end{minipage}",
      r"\end{table*}"]
open(os.path.join(OUT, "tab_xmm.tex"), "w").write("\n".join(T) + "\n")

# ---------------- robustez: L_eq e H1 ----------------
VAR = list(csv.DictReader(open(os.path.join(RES, "torque", "reversao_variantes.csv"))))
NOMEV = {"cov": "full covariance (adopted)", "diag": "diagonal covariance", "indep": "independent pairs only",
         "cov, L<=0.15Lmax": r"$L\leq0.15L_{\max}$", "cov, L<=0.5Lmax": r"$L\leq0.5L_{\max}$",
         "cov + s e erro de L": r"+ fractional and $L$-error terms", "cov, forma GL79": r"GL79 torque form"}
V = [r"\begin{table}",
     r"\centering",
     r"\footnotesize",
     r"\setlength{\tabcolsep}{3pt}",
     r"\caption{Robustness of the torque-reversal luminosity (equation~\ref{eq:rev}, with $A\geq0$). $\Delta$: "
     r"$-2\ln\mathcal{L}$ of the best torque that does not change sign within the sample ($\Leq\leq L_{\min}$, "
     r"including $\Leq=0$) minus that of the best fit. Limits: the interval $\Delta(-2\ln\mathcal{L})\leq1$ "
     r"reaches $L_{\min}$ (upper limits) or the edge of the grid, $3\times10^{39}\,\ergs$ (lower limits); "
     r"--: no constraint.}",
     r"\label{tab:leq}",
     r"\begin{tabular}{lrcr}",
     r"\hline",
     r"Variant & $N$ & $\Leq$ ($10^{36}\,\ergs$) & $\Delta$ \\",
     r"\hline"]
anterior = None
for x in VAR:
    f = x["fonte"]
    nome = NOMEV.get(x["variante"], x["variante"].replace("outburst MJD", "outburst, MJD").replace("-", "--"))
    borda = x["na_borda"] == "True"
    lo_b, hi_b = x["aberto_baixo"] == "True", x["aberto_alto"] == "True"
    if not borda:
        leq = f"{num(x['L_eq']) / 1e36:.1f} ({num(x['L_eq_lo']) / 1e36:.1f}--{num(x['L_eq_hi']) / 1e36:.1f})"
    elif hi_b and not lo_b:
        leq = f"$>{num(x['L_eq_lo']) / 1e36:.1f}$"
    elif lo_b and not hi_b:
        # sem reversão: o intervalo fica abaixo de L_min ou mal passa dele
        hi = num(x["L_eq_hi"]) if num(x["L_eq_hi"]) > 0 else num(x["L_min"])
        leq = f"$<{hi / 1e36:.1f}$"
    else:
        leq = "--"
    dl = num(x["delta_m2lnL_sem_reversao"])
    if f != anterior:
        # a fonte como linha de grupo, para a tabela caber na largura da coluna
        V.append(rf"\multicolumn{{4}}{{l}}{{\textit{{{f}}}}} \\")
    V.append(f"{nome} & {x['N']} & {leq} & {'--' if dl != dl else f'{dl:.1f}'} \\\\")
    anterior = f
V += [r"\hline", r"\end{tabular}", r"\end{table}"]
open(os.path.join(OUT, "tab_leq.tex"), "w").write("\n".join(V) + "\n")

ROB = os.path.join(RES, "sintese", "h1_robustez.json")
if os.path.exists(ROB):
    rb = json.load(open(ROB))
    CUR = {"4U 0115+63": "4U~0115", "V 0332+53": "V~0332", "1A 0535+262": "1A~0535", "Swift J0243.6+6124": "Swift"}
    R = [r"\begin{table}",
         r"\centering",
         r"\footnotesize",
         r"\setlength{\tabcolsep}{2pt}",
         r"\caption{Leave-one-source-out test of the hierarchical model (clean sample): posterior median of "
         r"$\sigma_{\rm source}$ (dex) and, in parentheses, $p$ of the test $\sigma_{\rm source}=0$ when each "
         r"source is removed. B12: \citet{Becker2012}; M15: \citet{Mushtukov2015}.}",
         r"\label{tab:loso}",
         r"\begin{tabular}{lcccc}",
         r"\hline",
         r"Removed & $\log L$ & $\log\ell$, $\Bcyc$, B12 & $\log\ell$, M15 & $\log\ell$, $\Bdip$ \\",
         r"\hline"]
    for fora in [f for f in CUR if f"L sem {f}" in rb["loso"]]:
        cel = []
        for y in ("L", "becker", "mushtukov", "dipolo"):
            r = rb["loso"][f"{y} sem {fora}"]
            cel.append(f"{r['sigma_fonte']['med']:.1f} (" + ("$<0.01$" if r["p"] < 0.005 else f"{r['p']:.2f}") + ")")
        R.append(f"{fora} & " + " & ".join(cel) + r" \\")
    R += [r"\hline", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "tab_loso.tex"), "w").write("\n".join(R) + "\n")

# ---------------- efemérides orbitais usadas na demodulação ----------------
from bexrb.fontes import FONTES  # noqa: E402
REF_ORB = {"4U 0115+63": "GBM web page$^a$", "V 0332+53": r"\citet{Doroshenko2016}",
           "1A 0535+262": "GBM web page$^b$", "Swift J0243.6+6124": "GBM web page$^c$",
           "X Per": r"\citet{DelgadoMarti2001}"}
O = [r"\begin{table*}",
     r"\centering",
     r"\footnotesize",
     r"\setlength{\tabcolsep}{3pt}",
     r"\caption{Orbital elements used for the demodulation of the XMM-Newton events. $T_0$ is the epoch of "
     r"periastron (MJD, TDB).}",
     r"\label{tab:orb}",
     r"\begin{tabular}{lcccccl}",
     r"\hline",
     r"Source & $P_{\rm orb}$ (d) & $a_x\sin i$ (lt-s) & $e$ & $\omega$ (deg) & $T_0$ & Reference \\",
     r"\hline"]
for f in ORDEM:
    ob = FONTES[f].orbita
    O.append(f"{f} & {ob.porb_d:.9g} & {ob.asini_lts:.6g} & {ob.e:.5g} & {ob.omega_deg:.5g} & "
             f"{ob.t_peri_mjd:.4f} & {REF_ORB[f]} \\\\")
O += [r"\hline",
      r"\end{tabular}",
      r"\begin{minipage}{0.75\textwidth}\footnotesize",
      r"Fermi/GBM Accreting Pulsars Program web pages (curator P. Jenke; "
      r"\url{https://gammaray.nsstc.nasa.gov/gbm/science/pulsars.html}), as of 2026 October: $^a$from "
      r"$T_{\pi/2}$; $^b$combined BATSE and GBM fit; $^c$preliminary.",
      r"\end{minipage}",
      r"\end{table*}"]
open(os.path.join(OUT, "tab_orb.tex"), "w").write("\n".join(O) + "\n")
print("tabelas em", OUT)
