"""Índice de discrepância de campo D = B_cyc / B_dip por três estimadores, com os sistemáticos amostrados.

Estimadores do campo dipolar (polar, B = 2 mu / R^3):
  T1  spin-up em alta L, ramo do rotador lento do ajuste A (só 1A 0535 e Swift J0243), refeito em cada sorteio
      dos sistemáticos (t1_reajuste): a escala B ~ lam^-3 d^-6 I^(7/2)... só vale para w -> 0 e falha aqui.
      Quando o ramo lento não tem mínimo interior, o sorteio não dá solução T1.
  T2  luminosidade de reversão do torque (reversao.py): r_m(L_eq) = w_c^(2/3) r_co, com w_c = 0,35 (GL79) ou 1
      (linear); B ~ xi^(-7/4) w_c^(7/6) M^(1/3) R^(-5/2) (lam L_eq)^(1/2) P^(7/6).
  P   luminosidade do propeller (Tsygankov+2016, eq. 2), da literatura e reescalada ao Gaia; limites superiores
      de L_lim dão limites superiores de B e inferiores de D.
B_cyc = (1+z) E_cyc / 11,57 keV com 1+z = (1 - 2GM/Rc^2)^(-1/2) e E_cyc uniforme na faixa da fundamental.

Sistemáticos: M ~ U(1,1; 2,0) Msol por fonte; R ~ U(11; 13) km comum (NICER: Miller+2021, Riley+2021);
I(M, R) de Lattimer & Schutz (2005); xi = k ~ U(0,5; 1,0) comum (GL79 usa 0,52; Tsygankov+2016, 0,5-0,7;
Wang 1996, ~1); d ~ N(Gaia); K_bol log-uniforme na faixa calibrada (alta L para T1, baixa L para T2).
Nas razões entre fontes, R, xi e w_c são comuns e se cancelam; M, d, K_bol e E_cyc são independentes.

L_eq: envoltória das variantes de reversao.py (covariância, pares independentes, cortes, outbursts).
Razão T2 também com xi próprio de cada estrela (sem acoplamento comum).

Uso: python indice.py <resultados>   (escreve <resultados>/sintese/indice.json)
"""
import csv
import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(AQUI, "resultados")
G, MSOL, C = 6.674e-8, 1.989e33, 2.998e10
N = 200000
rng = np.random.default_rng(20261027)
FONTES = ["4U 0115+63", "V 0332+53", "1A 0535+262", "Swift J0243.6+6124"]
P_SPIN = {"4U 0115+63": 3.614, "V 0332+53": 4.376, "1A 0535+262": 103.3, "Swift J0243.6+6124": 9.79}
# fundamental: faixa das medidas compiladas (dados/literatura/linhas_ciclotronicas.csv)
E_FAIXA = {"4U 0115+63": (10.1, 12.4), "V 0332+53": (25.9, 30.4), "1A 0535+262": (43.0, 47.7),
           "Swift J0243.6+6124": (120.0, 146.0)}
# K_bol = F(banda larga) / F_MAXI(2-20) (reversao.py, torque/kbol.csv): faixas adotadas em alta e baixa L
K_ALTO = {"4U 0115+63": (2.6, 3.0), "V 0332+53": (1.6, 2.0), "1A 0535+262": (2.6, 3.0),
          "Swift J0243.6+6124": (1.3, 1.6)}
K_BAIXO = {"1A 0535+262": (2.1, 2.5), "Swift J0243.6+6124": (1.8, 2.6)}


def lunif(a, b, n):
    return np.exp(rng.uniform(np.log(a), np.log(b), n))


def momento(m, r_km):
    x = m / r_km
    return 0.237 * m * MSOL * (r_km * 1e5) ** 2 * (1 + 4.2 * x + 90 * x ** 4)


def resumo(v):
    q = np.percentile(v, [5, 16, 50, 84, 95])
    return dict(zip(["q05", "q16", "med", "q84", "q95"], map(float, q)))


def t1_reajuste(par, n_sorteios=300):
    """T1 sem a escala do rotador lento (que não vale quando w cresce): em cada sorteio de (M, R, I(M,R),
    xi, d, K_bol), refaz o perfil do ajuste A e toma o mínimo do ramo de mu baixo. R e xi são comuns às duas
    fontes no mesmo sorteio; M, d e K_bol são próprios de cada uma. 'Interior' = o ramo lento tem um mínimo
    que não está na fronteira com o ramo de mu alto."""
    from torque_ajuste import PRESCR, dados, lmu_pico, perfil_mu, ramos
    r2 = np.random.default_rng(11)
    lmus = np.arange(26.0, 33.5, 0.02)
    Rk = r2.uniform(11.0, 13.0, n_sorteios)
    xs = r2.uniform(0.5, 1.0, n_sorteios)
    out = {}
    for f in ("1A 0535+262", "Swift J0243.6+6124"):
        d0, sd = float(par[f]["d_kpc"]), float(par[f]["d_err_kpc"])
        dd = dados(f, d0, 0.3, 0.0)
        m = r2.uniform(1.1, 2.0, n_sorteios)
        dist = np.clip(r2.normal(d0, sd, n_sorteios), 0.3 * d0, None)
        lam = lunif(*K_ALTO[f], n_sorteios) / 2 * (dist / d0) ** 2
        E = r2.uniform(*E_FAIXA[f], n_sorteios)
        out[f] = {}
        for nome, n in PRESCR.items():
            interior, lento_pref, D, B = [], [], [], []
            for k in range(n_sorteios):
                p = dict(m_=m[k] * MSOL, r_=Rk[k] * 1e5, i_=momento(m[k], Rk[k]), xi_=xs[k])
                c = perfil_mu(dd, n, lmus, lam[k], **p)
                rr = ramos(c, lmus, lmu_pico(dd, n, lmus, lam[k], **p))
                lo, hi = rr["baixo-mu"], rr["alto-mu"]
                ok = bool(lo and not lo["na_borda"])
                interior.append(ok)
                lento_pref.append(bool(lo and hi and lo["m2lnl"] < hi["m2lnl"]) or bool(lo and not hi))
                um_z = (1 - 2 * G * m[k] * MSOL / (Rk[k] * 1e5 * C ** 2)) ** -0.5
                b = 2 * 10 ** lo["lmu"] / (Rk[k] * 1e5) ** 3 if lo else np.nan
                B.append(b)
                D.append(um_z * E[k] / 11.57 * 1e12 / b)
            interior = np.array(interior)
            out[f][nome] = {"n": n_sorteios, "interior": interior, "frac_interior": float(interior.mean()),
                            "frac_lento_pref": float(np.mean(lento_pref)), "D_todos": np.array(D),
                            "D_interior": np.array(D)[interior], "B_interior": np.array(B)[interior]}
            print(f"T1 {f:20s} {nome:4s} interior {interior.mean():.2f} ramo lento preferido "
                  f"{np.mean(lento_pref):.2f}", flush=True)
    return out


def decompor_razao_t2(rev, par, n=200000):
    """Razão D_Swift / D_1A0535 por T2 (GL79), variando um grupo de parâmetros de cada vez; os outros ficam
    no valor fiducial (M = 1,4, R = 12 km, xi = 0,5 comum, d do Gaia, K_bol e L_eq no centro geométrico das
    faixas, E_cyc no centro). Grupos: luminosidade (L_eq, d, K_bol), estrutura (M de cada estrela e R, que
    entram também por 1+z), E_cyc, e acoplamento (xi próprio de cada estrela)."""
    r = np.random.default_rng(7)

    def b_cyc_e_dip(f, grupos):
        d0, sd = float(par[f]["d_kpc"]), float(par[f]["d_err_kpc"])
        lo, hi = np.log10(rev[f]["L_eq_lo"]), np.log10(rev[f]["L_eq_hi"])
        kb = K_BAIXO[f]
        e = E_FAIXA[f]
        m = r.uniform(1.1, 2.0, n) if "estrutura" in grupos else np.full(n, 1.4)
        R = grupos.get("R", np.full(n, 12.0))
        xi = r.uniform(0.5, 1.0, n) if "acoplamento" in grupos else np.full(n, 0.5)
        if "luminosidade" in grupos:
            d = np.clip(r.normal(d0, sd, n), 0.3 * d0, None)
            leq = 10 ** r.uniform(lo, hi, n) * (d / d0) ** 2
            lam = lunif(*kb, n) / 2
        else:
            leq = np.full(n, 10 ** (0.5 * (lo + hi)))
            lam = np.full(n, np.sqrt(kb[0] * kb[1]) / 2)
        E = r.uniform(*e, n) if "ecyc" in grupos else np.full(n, 0.5 * (e[0] + e[1]))
        gm = G * m * MSOL
        um_z = (1 - 2 * gm / (R * 1e5 * C ** 2)) ** -0.5
        mdot = lam * leq * R * 1e5 / gm
        rco = (gm * P_SPIN[f] ** 2 / (4 * np.pi ** 2)) ** (1 / 3)
        mu = ((0.35 ** (2 / 3) * rco / xi) ** 7 * 2 * gm * mdot ** 2) ** 0.25
        return um_z * E / 11.57 * 1e12, 2 * mu / (R * 1e5) ** 3

    out = {}
    for nome, grupos in (("fiducial", {}), ("luminosidade (L_eq, d, K_bol)", {"luminosidade": 1}),
                         ("estrutura (M de cada estrela, R)", {"estrutura": 1}), ("E_cyc", {"ecyc": 1}),
                         ("acoplamento (xi de cada estrela)", {"acoplamento": 1}),
                         ("todos, xi comum", None), ("todos, xi de cada estrela", None)):
        if grupos is None:
            continue
        if "estrutura" in grupos:
            grupos = dict(grupos, R=r.uniform(11.0, 13.0, n))
        bs, ds = b_cyc_e_dip("Swift J0243.6+6124", grupos)
        ba, da = b_cyc_e_dip("1A 0535+262", grupos)
        out[nome] = resumo((bs / ds) / (ba / da))
    return out


if __name__ == "__main__":
    par = {x["fonte"]: x for x in csv.DictReader(open(os.path.join(AQUI, "dados", "parametros_fontes.csv")))}
    aj = [x for x in csv.DictReader(open(os.path.join(RES, "torque", "ajuste_global.csv")))
          if x["ajuste"] == "A" and x["ramo"] == "baixo-mu"]
    # L_eq: envoltória das variantes sem borda e com Delta >= 4 contra a nula sem troca de sinal (covariância,
    # diagonal, pares independentes, cortes em L, forma GL79, outbursts separados). Sem reversão firme, sem T2
    rev = {}
    var = [x for x in csv.DictReader(open(os.path.join(RES, "torque", "reversao_variantes.csv")))
           if x["na_borda"] == "False"]
    for f in {x["fonte"] for x in var}:
        firmes = [x for x in var if x["fonte"] == f and float(x["delta_m2lnL_sem_reversao"]) >= 4]
        if firmes:
            rev[f] = {"L_eq_lo": min(float(x["L_eq_lo"]) for x in firmes),
                      "L_eq_hi": max(float(x["L_eq_hi"]) for x in firmes),
                      "L_eq": float([x for x in firmes if x["variante"] == "cov"][0]["L_eq"]), "marginal": False,
                      "variantes": [x["variante"] for x in firmes]}
    prop = list(csv.DictReader(open(os.path.join(RES, "sintese", "campos_propeller.csv"))))

    T1 = t1_reajuste(par)
    R = rng.uniform(11.0, 13.0, N)
    xi = rng.uniform(0.5, 1.0, N)
    am = {}
    for f in FONTES:
        m = rng.uniform(1.1, 2.0, N)
        d0, sd = float(par[f]["d_kpc"]), float(par[f]["d_err_kpc"])
        d = np.clip(rng.normal(d0, sd, N), 0.3 * d0, None)
        um_z = (1 - 2 * G * m * MSOL / (R * 1e5 * C ** 2)) ** -0.5
        bcyc = um_z * rng.uniform(*E_FAIXA[f], N) / 11.57 * 1e12
        am[f] = {"m": m, "d": d, "d0": d0, "bcyc": bcyc, "I": momento(m, R)}
    out = {"amostras": N, "fontes": {}}

    for f in FONTES:
        a = am[f]
        o = out["fontes"][f] = {"B_cyc_G": resumo(a["bcyc"])}
        # ---------- T1: reajuste do ajuste A em cada sorteio (t1_reajuste) ----------
        if f in T1:
            for nome, r in T1[f].items():
                if r["D_interior"].size:
                    a[f"T1_{nome}"] = r
                    o[f"T1_{nome}"] = {"D": resumo(r["D_interior"]), "B_dip_G": resumo(r["B_interior"]),
                                       "frac_interior": r["frac_interior"], "frac_ramo_lento_preferido":
                                       r["frac_lento_pref"], "n_sorteios": r["n"]}
        # ---------- T2 ----------
        if f in rev:
            lo, hi = np.log10(rev[f]["L_eq_lo"]), np.log10(rev[f]["L_eq_hi"])
            leq = 10 ** rng.uniform(lo, hi, N) * (a["d"] / a["d0"]) ** 2
            lam = lunif(*K_BAIXO.get(f, K_ALTO[f]), N) / 2
            gm = G * a["m"] * MSOL
            mdot = lam * leq * R * 1e5 / gm
            rco = (gm * P_SPIN[f] ** 2 / (4 * np.pi ** 2)) ** (1 / 3)
            for nome, wc in (("GL79", 0.35), ("lin", 1.0)):
                rm = wc ** (2 / 3) * rco
                mu = ((rm / xi) ** 7 * 2 * gm * mdot ** 2) ** 0.25
                b = 2 * mu / (R * 1e5) ** 3
                a[f"T2_{nome}"] = a["bcyc"] / b
                o[f"T2_{nome}"] = {"L_eq": rev[f]["L_eq"], "L_eq_faixa": [rev[f]["L_eq_lo"], rev[f]["L_eq_hi"]],
                                   "marginal": rev[f]["marginal"], "B_dip_G": resumo(b), "D": resumo(a["bcyc"] / b)}
                if nome == "GL79":
                    # o mesmo com xi próprio de cada estrela (acoplamento não comum)
                    xi2 = rng.uniform(0.5, 1.0, N)
                    a["T2_GL79_xi_proprio"] = a["bcyc"] / (b * (xi2 / xi) ** -1.75)
        # ---------- P ----------
        # a de 6e33 de Doroshenko+2020 não tem pulsação detectada: não é acreção comprovada, fica de fora
        pp = [x for x in prop if f.startswith(x["fonte"][:9]) and float(x["L_lim_lit"]) >= 1e34]
        if f == "1A 0535+262":
            # acreção estável com pulsações a 7e34 (0,5-100 keV, d = 2 kpc; Tsygankov+2019): L_lim abaixo disso
            pp = [{"L_lim_gaia": 7e34 * (a["d0"] / 2.0) ** 2, "ref": "Tsygankov+2019", "limite_superior": "True"}]
        for x in pp:
            L = float(x["L_lim_gaia"]) * (a["d"] / a["d0"]) ** 2
            b = np.sqrt(L / (4e37 * xi ** 3.5 * P_SPIN[f] ** (-7 / 3) * (a["m"] / 1.4) ** (-2 / 3)
                             * (R / 10) ** 5)) * 1e12
            lim = x["limite_superior"] == "True"
            chave = "P_" + x["ref"].replace("+", "")
            a[chave] = a["bcyc"] / b
            o[chave] = {"L_lim_gaia": float(x["L_lim_gaia"]), "limite": lim, "B_dip_G": resumo(b),
                        "D": resumo(a["bcyc"] / b)}

    # ---------- razões entre fontes (mesmo estimador; R, xi, w_c comuns) ----------
    S, A = am["Swift J0243.6+6124"], am["1A 0535+262"]
    raz = {}
    for nome in ("GL79", "lin"):
        s_, a_ = T1["Swift J0243.6+6124"][nome], T1["1A 0535+262"][nome]
        ambos = s_["interior"] & a_["interior"]
        raz[f"T1_{nome}: Swift/1A0535"] = dict(resumo(s_["D_todos"][ambos] / a_["D_todos"][ambos]) if ambos.sum() > 5
                                               else {}, frac_pares_interiores=float(ambos.mean()))
    for k in ("T2_GL79", "T2_lin", "T2_GL79_xi_proprio"):
        if k in S and k in A:
            raz[f"{k}: Swift/1A0535"] = resumo(S[k] / A[k])
    U = am["4U 0115+63"]
    for k in ("T2_GL79", "T2_lin"):
        if k in U:
            raz[f"{k}: Swift/4U0115 (reversão marginal na 4U 0115)"] = resumo(S[k] / U[k])
    # 4U 0115: reversão (GL79) contra o propeller observado, mesma fonte
    if "T2_GL79" in U:
        raz["4U0115: B_dip T2_GL79 / B_dip P (mesma fonte)"] = resumo(U["P_Tsygankov2016"] / U["T2_GL79"])
    # Swift e 1A 0535: B_T2 / B_P no mesmo sorteio (xi, M, d, E_cyc comuns). P é limite superior de B,
    # então a razão é um limite inferior do quanto T2 excede o campo permitido pelo propeller
    for f in ("Swift J0243.6+6124", "1A 0535+262"):
        for k2 in ("T2_GL79", "T2_lin"):
            for kp in [k for k in am[f] if k.startswith("P_")]:
                raz[f"{f}: B_dip {k2} / B_dip {kp} (mesma fonte, limite inferior)"] = resumo(am[f][kp] / am[f][k2])
    for outra in ("4U 0115+63", "V 0332+53"):
        for k in [k for k in am[outra] if k.startswith("P_")]:
            for ks in [k for k in S if k.startswith("P_")]:
                raz[f"{ks} (limite) / {k} {outra}"] = resumo(S[ks] / am[outra][k])
    out["razoes"] = raz
    out["decomposicao_razao_T2_GL79"] = decompor_razao_t2(rev, par)
    out["decomposicao_razao_T2_GL79"]["todos, xi comum"] = raz["T2_GL79: Swift/1A0535"]
    out["decomposicao_razao_T2_GL79"]["todos, xi de cada estrela"] = raz["T2_GL79_xi_proprio: Swift/1A0535"]
    for k, v in out["decomposicao_razao_T2_GL79"].items():
        print(f"decomposição {k:36s} med {v['med']:.3g} [{v['q16']:.3g}, {v['q84']:.3g}]  q05 {v['q05']:.3g} q95 {v['q95']:.3g}")
    json.dump(out, open(os.path.join(RES, "sintese", "indice.json"), "w"), indent=1)
    for f, o in out["fontes"].items():
        for k, v in o.items():
            if k.startswith(("T", "P")):
                print(f"{f:20s} {k:22s} D: med {v['D']['med']:.3g} [{v['D']['q16']:.3g}, {v['D']['q84']:.3g}] "
                      f"q05 {v['D']['q05']:.3g}  B_dip med {v['B_dip_G']['med']:.2e}")
    for k, v in raz.items():
        print(f"razão {k:55s} med {v['med']:.3g} [{v['q16']:.3g}, {v['q84']:.3g}] q05 {v['q05']:.3g}")
