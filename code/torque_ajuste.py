"""H2, ajuste global do torque: verossimilhança explícita, dispersão intrínseca, os dois ramos e duas
prescrições de torque. Os pontos vêm de torque.py.

Modelo de cada ponto i (nu-ponto do GBM, L do MAXI com K_bol = 2 e a distância do Gaia):
    nu-ponto_i ~ N(m_i, sig_i^2 + (s m_i)^2 + s0^2 + (dm_i/dlnL * sigL_i/L_i)^2),
    m_i = n(w) Mdot sqrt(G M r_m) / (2 pi I),  Mdot = lam L_i R / (G M),  w = (r_m/r_co)^(3/2),
com n(w) de Ghosh & Lamb (1979; fastness crítica w_c = 0,35; cortada em w = 0,95, onde diverge) ou a forma
linear n = 1 - w (Ho+2014, eq. 7), com w_c = 1, que imita a fastness crítica alta (0,88-0,95) dos modelos de
Wang (Bozzo+2009) e gira para baixo acima de w = 1 (corte numérico em w = 5). Parâmetros: log mu,
lam (escala de luminosidade: K_bol/2 vezes (d/d_Gaia)^2), s (dispersão fracionária) e s0 (piso absoluto).

Dois ajustes por fonte e prescrição:
  A: só L >= 0,3 L_max, lam = 1; perfil de -2 ln L em log mu, com os dois ramos;
  B: todos os pontos com L medida a S/N >= 3, lam livre; o trecho de baixa L, perto do equilíbrio, quebra a
     degenerescência mu-L do rotador lento (mu ~ L^-3), como no ajuste com distância livre de Doroshenko+2018.

Uso: python torque_ajuste.py <resultados> [--indep] [--I=1.43e45]   (escreve <resultados>/torque/ajuste_global.csv e perfis_*.npz)
  --I=<g cm^2>: só o ajuste A, com outro momento de inércia (ajuste_global_I<valor>.csv)
"""
import csv
import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from torque import GBM, G, M, R, I, XI, pontos  # noqa: E402

RES = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else os.path.join(AQUI, "resultados")
SAIDA = os.path.join(RES, "torque")
os.makedirs(SAIDA, exist_ok=True)
W_MAX = 0.95          # GL79 diverge em w = 1
W_MAX_LIN = 5.0       # a forma linear gira para baixo acima de w = 1; o corte é só numérico


def n_gl(w):
    w = np.minimum(w, W_MAX)
    return 1.39 * (1 - w * (4.03 * (1 - w) ** 0.173 - 0.878)) / (1 - w)


def n_lin(w):
    return 1 - np.minimum(w, W_MAX_LIN)


PRESCR = {"GL79": n_gl, "lin": n_lin}


def modelo(lmu, L, nu, n, lam=1.0, m_=M, r_=R, i_=I, xi_=XI):
    """nu-ponto para log mu (qualquer forma, com broadcast) e L, nu (vetores dos pontos). m_, r_, i_ e xi_
    (cgs) trocam os valores fiduciais (Monte Carlo do T1 em indice.py)."""
    mu = 10.0 ** lmu
    mdot = lam * L * r_ / (G * m_)
    rm = xi_ * (mu ** 4 / (2 * G * m_ * mdot ** 2)) ** (1 / 7)
    rco = (G * m_ / (2 * np.pi * nu) ** 2) ** (1 / 3)
    return n((rm / rco) ** 1.5) * mdot * np.sqrt(G * m_ * rm) / (2 * np.pi * i_)


def m2lnl(lmu, lam, s, s0, d, n):
    """-2 ln L; lmu e lam escalares ou arrays com broadcast contra os pontos (último eixo)."""
    m = modelo(lmu, d["L"], d["nu"], n, lam)
    dm = (modelo(lmu, d["L"] * 1.01, d["nu"], n, lam) - m) / np.log(1.01)
    v = d["sig"] ** 2 + (s * m) ** 2 + s0 ** 2 + (dm * d["sL"]) ** 2
    return np.sum((d["nd"] - m) ** 2 / v + np.log(v), axis=-1)


S_GRADE = np.concatenate([[0.0], np.logspace(-3, 0.5, 36)])
S0_FRAC = np.concatenate([[0.0], np.logspace(-3, 0.3, 14)])


def perfil_mu(d, n, lmus, lam=1.0, **par):
    """Perfil de -2 ln L em log mu: em cada mu, mínimo numa grade fina de (s, s0)."""
    s0g = S0_FRAC * np.median(np.abs(d["nd"]))
    m = modelo(lmus[:, None], d["L"][None, :], d["nu"][None, :], n, lam, **par)
    dm = (modelo(lmus[:, None], d["L"][None, :] * 1.01, d["nu"][None, :], n, lam, **par) - m) / np.log(1.01)
    base = d["sig"] ** 2 + (dm * d["sL"]) ** 2
    r2 = (d["nd"] - m) ** 2
    melhor = np.full(lmus.size, np.inf)
    for s in S_GRADE:
        sm2 = (s * m) ** 2
        for s0 in s0g:
            v = base + sm2 + s0 ** 2
            melhor = np.minimum(melhor, np.sum(r2 / v + np.log(v), axis=1))
    return melhor


def lmu_pico(d, n, lmus, lam=1.0, **par):
    """log mu do máximo de nu-ponto(mu) na L mediana: separa o ramo de mu baixo do de mu alto."""
    k = int(np.argmin(np.abs(d["L"] - np.median(d["L"]))))
    m = modelo(lmus, d["L"][k], d["nu"][k], n, lam, **par)
    return lmus[int(np.argmax(m))]


def ramos(c, lmus, corte):
    """Mínimo e intervalo Delta(-2lnL) <= 1 de cada lado do corte (ramo sem mínimo interno: None)."""
    out = {}
    for nome, s in (("baixo-mu", lmus < corte), ("alto-mu", lmus >= corte)):
        idx = np.where(s & np.isfinite(c))[0]
        if idx.size < 3:
            out[nome] = None
            continue
        i = idx[int(np.argmin(c[idx]))]
        borda = i in (idx[0], idx[-1])
        ok = idx[c[idx] <= c[i] + 1.0]
        out[nome] = {"i": int(i), "lmu": float(lmus[i]), "m2lnl": float(c[i]), "lo": float(lmus[ok.min()]),
                     "hi": float(lmus[ok.max()]), "na_borda": bool(borda)}
    return out


def perfil_ramo_b(c2, lmus, cortes, ramo):
    """Ajuste B: perfil de -2 ln L em log mu, minimizando sobre lam só os pontos do ramo (o corte entre os
    ramos depende de lam). O intervalo Delta <= 1 em mu sai deste perfil."""
    lado = lmus[None, :] < cortes[:, None] if ramo == "baixo-mu" else lmus[None, :] >= cortes[:, None]
    return np.where(lado, c2, np.inf).min(axis=0)


def dados(fonte, d_kpc, frac=0.3, snr=3.0):
    p, _ = pontos(fonte, d_kpc)
    L = np.array([x["L"] for x in p])
    sL = np.array([x["sig_taxa"] / x["taxa_maxi"] for x in p])
    s = (L >= frac * L.max()) & (1 / np.maximum(sL, 1e-9) >= snr)
    return {"L": L[s], "sL": sL[s], "nu": np.array([x["nu"] for x in p])[s],
            "nd": np.array([x["nudot"] for x in p])[s], "sig": np.array([x["sig"] for x in p])[s],
            "mjd": np.array([x["mjd"] for x in p])[s], "i": np.array([x["i"] for x in p])[s],
            "dt_s": np.array([x["dt_s"] for x in p])[s], "sig_nu0": np.array([x["sig_nu0"] for x in p])[s],
            "sig_nu1": np.array([x["sig_nu1"] for x in p])[s]}


def covariancia(d):
    """Covariância de medida dos nu-ponto: diagonal sig^2 e, para pares vizinhos que dividem uma
    frequência, -sig_nu^2 / (dt_a dt_b)."""
    n = d["nd"].size
    C = np.diag(d["sig"] ** 2)
    pos = {int(k): j for j, k in enumerate(d["i"])}
    for j, k in enumerate(d["i"]):
        b = pos.get(int(k) + 1)
        if b is not None:
            C[j, b] = C[b, j] = -d["sig_nu1"][j] ** 2 / (d["dt_s"][j] * d["dt_s"][b])
    return C


def independentes(d):
    """Subconjunto sem frequência compartilhada: guarda um par só se não usa a frequência do anterior."""
    o = np.argsort(d["i"])
    manter, ult = [], -10
    for j in o:
        if d["i"][j] >= ult + 2:
            manter.append(j)
            ult = d["i"][j]
    manter = np.array(sorted(manter))
    return {k: v[manter] for k, v in d.items()}


def b_polo(lmu):
    return 2 * 10 ** lmu / R ** 3


if __name__ == "__main__":
    # --indep: só pares sem frequência compartilhada (teste da covariância entre nu-ponto vizinhos)
    INDEP = "--indep" in sys.argv
    SUF = "_indep" if INDEP else ""
    # --I=<valor>: momento de inércia alternativo, só no ajuste A
    I_ALT = [float(a.split("=")[1]) for a in sys.argv if a.startswith("--I=")]
    KW = {"i_": I_ALT[0]} if I_ALT else {}
    SUF += f"_I{I_ALT[0]:.3g}" if I_ALT else ""
    par = {x["fonte"]: x for x in csv.DictReader(open(os.path.join(AQUI, "dados", "parametros_fontes.csv")))}
    LMU = np.arange(26.0, 33.5, 0.01)
    LLAM = np.arange(-0.8, 0.8001, 0.02)
    linhas = []
    for fonte in GBM:
        dk = float(par[fonte]["d_kpc"])
        for nome, n in PRESCR.items():
            # ---------- A: alta L, lam = 1 ----------
            d = dados(fonte, dk, 0.3, 0.0)
            d = independentes(d) if INDEP else d
            c = perfil_mu(d, n, LMU, **KW)
            corte = lmu_pico(d, n, LMU, **KW)
            rr = ramos(c, LMU, corte)
            melhor = min(r["m2lnl"] for r in rr.values() if r)
            for ramo, r in rr.items():
                if not r:
                    continue
                linhas.append({"fonte": fonte, "torque": nome, "ajuste": "A", "N": d["L"].size, "ramo": ramo,
                               "na_borda": r["na_borda"], "log_mu": r["lmu"], "log_mu_lo": r["lo"], "log_mu_hi": r["hi"],
                               "B_dip_G": b_polo(r["lmu"]), "B_lo_G": b_polo(r["lo"]), "B_hi_G": b_polo(r["hi"]),
                               "lam": 1.0, "lam_lo": 1.0, "lam_hi": 1.0, "m2lnL": r["m2lnl"],
                               "delta_m2lnL": r["m2lnl"] - melhor})
            np.savez(os.path.join(SAIDA, f"perfil_A_{GBM[fonte][0]}_{nome}{SUF}.npz"), lmu=LMU, c=c, corte=corte)
            if I_ALT:
                continue
            # ---------- B: todos os pontos com S/N >= 3, lam livre ----------
            d = dados(fonte, dk, 0.0, 3.0)
            d = independentes(d) if INDEP else d
            c2 = np.array([perfil_mu(d, n, LMU, 10 ** ll) for ll in LLAM])
            cortes = np.array([lmu_pico(d, n, LMU, 10 ** ll) for ll in LLAM])
            por_ramo = {"baixo-mu": [], "alto-mu": []}
            for j in range(LLAM.size):
                rr = ramos(c2[j], LMU, cortes[j])
                for ramo in por_ramo:
                    por_ramo[ramo].append(rr[ramo])
            tudo = [r["m2lnl"] for v in por_ramo.values() for r in v if r]
            melhor = min(tudo)
            for ramo, v in por_ramo.items():
                cl = np.array([r["m2lnl"] if r else np.inf for r in v])
                if not np.isfinite(cl).any():
                    continue
                j = int(np.argmin(cl))
                ok = np.where(cl <= cl[j] + 1.0)[0]
                cmu = perfil_ramo_b(c2, LMU, cortes, ramo)
                lmus_ok = LMU[cmu <= cl[j] + 1.0]
                linhas.append({"fonte": fonte, "torque": nome, "ajuste": "B", "N": d["L"].size, "ramo": ramo,
                               "na_borda": v[j]["na_borda"] or j in (0, LLAM.size - 1),
                               "log_mu": v[j]["lmu"], "log_mu_lo": lmus_ok.min(), "log_mu_hi": lmus_ok.max(),
                               "B_dip_G": b_polo(v[j]["lmu"]), "B_lo_G": b_polo(lmus_ok.min()),
                               "B_hi_G": b_polo(lmus_ok.max()), "lam": 10 ** LLAM[j],
                               "lam_lo": 10 ** LLAM[ok.min()], "lam_hi": 10 ** LLAM[ok.max()],
                               "m2lnL": cl[j], "delta_m2lnL": cl[j] - melhor})
            np.savez(os.path.join(SAIDA, f"perfil_B_{GBM[fonte][0]}_{nome}{SUF}.npz"), lmu=LMU, llam=LLAM, c=c2, cortes=cortes)
            print(fonte, nome, "feito", flush=True)
    with open(os.path.join(SAIDA, f"ajuste_global{SUF}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)
    for x in linhas:
        print(f"{x['fonte']:19s} {x['torque']:4s} {x['ajuste']} N={x['N']:3d} {x['ramo']:8s} borda={x['na_borda']!s:5s} "
              f"B={x['B_dip_G']:.2e} [{x['B_lo_G']:.2e}, {x['B_hi_G']:.2e}] lam={x['lam']:.2f} [{x['lam_lo']:.2f}, "
              f"{x['lam_hi']:.2f}] d(-2lnL)={x['delta_m2lnL']:.1f}")
