"""H1, estatística hierárquica das 28 transições: separa a dispersão entre fontes, entre indicadores,
entre trabalhos (análises do mesmo outburst) e a residual, e compara com a esperada pelos erros de medida.

Modelo, para y = log10 de L, ell_Becker, ell_Mushtukov ou ell_dipolo de cada transição j:
    y_j = alfa + a_fonte + b_indicador + c_trabalho + e_j,
    a ~ N(0, sa^2), b ~ N(0, sb^2), c ~ N(0, sc^2), e_j ~ N(0, sw^2 + erro_j^2),
com alfa de prior plano (integrado analiticamente) e sa, sb, sc, sw de prior uniforme em [0, 2] dex,
numa grade. O erro de medida de cada transição soma em quadratura a meia largura da faixa publicada e a
correção de banda (0,05 dex para bandas largas ou bolométricas; 0,15 dex para bandas estreitas, que também são
deslocadas de +0,2 dex na variante "banda corrigida"). A distância do Gaia é a mesma para todas as transições
de uma fonte: entra como um bloco da covariância de medida, e_d,i e_d,j para i e j da mesma fonte, com
e_d = 2 sd/d em log L e em ell com B_cyc, e (2 - 16/15) sd/d em ell com B_dip, porque B_dip de T2 e P é
proporcional a d.

Se as transições marcassem um mesmo evento num ell universal, sa e sb seriam ~0 em ell, e sa seria ~0,5 dex
em L (a dispersão de log L_crit entre as fontes).

Uso: python h1_hierarquico.py <resultados> ['<json com D>']   (lê sintese/transicoes_h1.csv e, sem o json, o D de
sintese/indice.json; escreve sintese/h1_hierarquico.json)
"""
import csv
import json
import os
import re
import sys

import numpy as np
from scipy.stats import chi2

AQUI = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(AQUI, "resultados")
TIPOS = ["transicao_perfil", "minimo_PF", "troca_ramo_HID", "inversao_Ecyc_L"]
LARGAS = ("bolom", "2-150", "1-150", "3-100", "0.5-100", "2-100", "1-100")
# amostras: "limpa" = sem duplicatas e sem linhas que não marcam o início da coluna na fonte (coluna h1 da
# compilação = "sim"); "sem_duplicatas"; "todas" (os 28 valores publicados)
AMOSTRAS = {"limpa": lambda h: h == "sim", "sem_duplicatas": lambda h: not h.startswith("duplicata"),
            "todas": lambda h: True}


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def meia_largura(s):
    t = str(s).replace("~", "").replace(" ", "")
    m = re.fullmatch(r"([\d.]+(?:e[+-]?\d+)?)-([\d.]+(?:e[+-]?\d+)?)", t)
    return 0.5 * abs(np.log10(float(m.group(2)) / float(m.group(1)))) if m else 0.0


def carregar(m_dip, amostra="limpa"):
    par = {x["fonte"]: x for x in csv.DictReader(open(os.path.join(AQUI, "dados", "parametros_fontes.csv")))}
    tr = [x for x in csv.DictReader(open(os.path.join(RES, "sintese", "transicoes_h1.csv")))
          if x["tipo"] in TIPOS and x["teorico"] == "False" and num(x["ell_becker"]) > 0
          and AMOSTRAS[amostra](x.get("h1", "sim"))]
    out = []
    for x in tr:
        f = x["fonte"]
        larga = any(b in x["banda_L"].lower() for b in LARGAS)
        e_d = 2 * num(par[f]["d_err_kpc"]) / num(par[f]["d_kpc"]) / np.log(10)
        e_o = np.sqrt(meia_largura(x["L_erg_s"]) ** 2 + (0.05 if larga else 0.15) ** 2)
        out.append({"fonte": f, "tipo": x["tipo"], "ref": x["ref"], "larga": larga, "erro": np.hypot(e_d, e_o),
                    "erro_d": e_d, "erro_outros": e_o, "h1": x.get("h1", ""),
                    "L": np.log10(num(x["L_gaia"])), "becker": np.log10(num(x["ell_becker"])),
                    "mushtukov": np.log10(num(x["ell_mushtukov"])),
                    "dipolo": np.log10(num(x["ell_becker"]) * m_dip[f] ** (16 / 15))})
    return out


def cov_medida(tr, y_nome="L"):
    """Covariância de medida: diagonal com os erros próprios de cada transição mais o bloco da distância,
    comum às transições da mesma fonte; em ell com B_dip a distância entra com (2 - 16/15)/2 do peso."""
    fd = (2 - 16 / 15) / 2 if y_nome.startswith("dipolo") else 1.0
    ed = fd * np.array([t["erro_d"] for t in tr])
    mesma = np.array([[a["fonte"] == b["fonte"] for b in tr] for a in tr])
    return np.diag([t["erro_outros"] ** 2 for t in tr]) + mesma * np.outer(ed, ed)


def incid(rotulos):
    u = sorted(set(rotulos))
    Z = np.zeros((len(rotulos), len(u)))
    for i, r in enumerate(rotulos):
        Z[i, u.index(r)] = 1
    return Z @ Z.T


def log_ver(y, e, K, grade):
    """log p(y | sigmas) com alfa integrado; grade: (n, 4) com (sa, sb, sc, sw). e: covariância de medida
    (n, n) ou, por compatibilidade, os erros (n,) de uma covariância diagonal."""
    n = y.size
    um = np.ones(n)
    Cm = e if np.ndim(e) == 2 else np.diag(np.asarray(e) ** 2)
    out = np.empty(len(grade))
    for a in range(0, len(grade), 20000):
        g = grade[a:a + 20000]
        S = (g[:, 0, None, None] ** 2 * K[0] + g[:, 1, None, None] ** 2 * K[1] + g[:, 2, None, None] ** 2 * K[2]
             + np.eye(n) * g[:, 3, None, None] ** 2 + Cm)
        C = np.linalg.cholesky(S)
        logdet = 2 * np.sum(np.log(np.diagonal(C, axis1=1, axis2=2)), axis=1)
        Si1 = np.linalg.solve(S, np.broadcast_to(um, (len(g), n))[..., None])[..., 0]
        Siy = np.linalg.solve(S, np.broadcast_to(y, (len(g), n))[..., None])[..., 0]
        a11 = Si1 @ um
        a1y = Si1 @ y
        ayy = Siy @ y
        out[a:a + 20000] = -0.5 * (logdet + np.log(a11) + ayy - a1y ** 2 / a11)
    return out


def resumo(v, w):
    o = np.argsort(v)
    c = np.cumsum(w[o]) / np.sum(w)
    q = [float(v[o][np.searchsorted(c, p)]) for p in (0.16, 0.5, 0.84, 0.95)]
    return {"q16": q[0], "med": q[1], "q84": q[2], "q95": q[3]}


def p_mistura(lrt):
    """p do teste sigma_fonte = 0 com a mistura 50:50 de chi2_0 e chi2_1."""
    return 1.0 if lrt <= 0 else float(0.5 * chi2.sf(lrt, 1))


TETO = 2.0          # prior uniforme de cada desvio-padrão em [0, 2] dex
SG = np.arange(0.0, TETO + 1e-9, 0.1)


def ajustar(tr, y, grade, y_nome="L"):
    e = np.array([t["erro"] for t in tr])
    Cm = cov_medida(tr, y_nome)
    Ci = np.linalg.inv(Cm)
    um = np.ones(y.size)
    media = (um @ Ci @ y) / (um @ Ci @ um)
    K = [incid([t["fonte"] for t in tr]), incid([t["tipo"] for t in tr]), incid([t["ref"] for t in tr])]
    lv = log_ver(y, Cm, K, grade)
    w = np.exp(lv - lv.max())
    lrt = 2 * (lv.max() - lv[grade[:, 0] == 0].max())
    # densidade posterior marginal de sa em 0 (Savage-Dickey), com prior uniforme em [0, TETO]
    marg_sa = np.array([w[grade[:, 0] == s].sum() for s in SG])
    marg_sa /= marg_sa.sum() * (SG[1] - SG[0])
    return {"N": int(y.size), "n_fontes": int(len(set(t["fonte"] for t in tr))),
            "n_por_fonte": {f: sum(t["fonte"] == f for t in tr) for f in sorted(set(t["fonte"] for t in tr))},
            "dp_total_dex": float(np.std(y, ddof=1)), "erro_rms_dex": float(np.sqrt(np.mean(e ** 2))),
            "chi2_reduzido_media_unica": float((y - media) @ Ci @ (y - media) / (y.size - 1)),
            "sigma_fonte": resumo(grade[:, 0], w), "sigma_indicador": resumo(grade[:, 1], w),
            "sigma_trabalho": resumo(grade[:, 2], w), "sigma_residual": resumo(grade[:, 3], w),
            "LRT_sigma_fonte_0": float(lrt), "p_LRT_mistura": p_mistura(lrt),
            "BF_10_savage_dickey": float((1 / TETO) / max(marg_sa[0], 1e-12))}


def grade_d(res):
    """Valores de D da sensibilidade da normalização dipolar (cenários da seção 6.1), das medianas de
    sintese/indice.json: Swift J0243 por T2 (linear, GL79) e pelos limites do propeller (T18, D20); 1A 0535 por T2
    (linear, GL79), pelo limite do propeller (T19) e 1 e 2."""
    ind = json.load(open(os.path.join(res, "sintese", "indice.json")))["fontes"]
    s, a = ind["Swift J0243.6+6124"], ind["1A 0535+262"]
    ds = [s[k]["D"]["med"] for k in ("T2_lin", "T2_GL79", "P_Tsygankov2018", "P_Doroshenko2020")]
    da = [a[k]["D"]["med"] for k in ("T2_lin", "T2_GL79", "P_Tsygankov2019")] + [1.0, 2.0]
    return [float(round(x, 3)) for x in ds], [float(round(x, 3)) for x in da]


def d_indice(res):
    """D de cada fonte para a normalização dipolar: propeller na 4U 0115 e na V 0332, reversão (GL79) na
    1A 0535 e na Swift J0243 (medianas de sintese/indice.json)."""
    ind = json.load(open(os.path.join(res, "sintese", "indice.json")))["fontes"]
    return {"4U 0115+63": ind["4U 0115+63"]["P_Tsygankov2016"]["D"]["med"],
            "V 0332+53": ind["V 0332+53"]["P_Tsygankov2016"]["D"]["med"],
            "1A 0535+262": ind["1A 0535+262"]["T2_GL79"]["D"]["med"],
            "Swift J0243.6+6124": ind["Swift J0243.6+6124"]["T2_GL79"]["D"]["med"]}


if __name__ == "__main__":
    m_dip = json.loads(sys.argv[2]) if len(sys.argv) > 2 else d_indice(RES)
    grade = np.array(np.meshgrid(SG, SG, SG, SG, indexing="ij")).reshape(4, -1).T
    saida = {}
    for amostra in AMOSTRAS:
        saida[amostra] = {}
        tr = carregar(m_dip, amostra)
        for corrigir in (False, True):
            for y_nome in ("L", "becker", "mushtukov", "dipolo"):
                y = np.array([t[y_nome] + (0.2 if corrigir and not t["larga"] else 0.0) for t in tr])
                chave = f"{y_nome}{'_banda_corrigida' if corrigir else ''}"
                r = saida[amostra][chave] = ajustar(tr, y, grade, y_nome)
                print(amostra, chave, f"N={r['N']} fontes={r['n_fontes']} chi2nu={r['chi2_reduzido_media_unica']:.1f} "
                      f"sigma_fonte={r['sigma_fonte']['med']:.2f} ({r['sigma_fonte']['q16']:.2f}-{r['sigma_fonte']['q84']:.2f}) "
                      f"p={r['p_LRT_mistura']:.3f} BF={r['BF_10_savage_dickey']:.1f} trabalho={r['sigma_trabalho']['med']:.2f} "
                      f"indicador={r['sigma_indicador']['med']:.2f}", flush=True)
    json.dump(saida, open(os.path.join(RES, "sintese", "h1_hierarquico.json"), "w"), indent=1)
