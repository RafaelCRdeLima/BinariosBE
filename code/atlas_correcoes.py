"""Detecção, limites superiores e fração pulsada do atlas XMM, calculados a partir dos produtos salvos de
rodar_atlas.py (atlas/obs/*.json, *_perfis.npz e atlas/harmonicos.csv):

1. Limite superior de PF_rms a 3 sigma: lambda_UL tal que P(Z^2_2 <= z_obs | lambda) = 0,00135,
   com Z^2_2 ~ chi^2 não central (4 graus de liberdade, não centralidade lambda = N PF_tot^2);
   PF_UL = sqrt(lambda_UL / N) N / (N - B). O z_obs vem do perfil 0,3-10 keV salvo (32 bins, com a
   exposição por fase); sem perfil salvo (poucos eventos), vale o limite que atlas_obs.py calcula dos eventos.
2. Tentativas: maximizar Z^2_4 na grade multiplica o número efetivo de frequências independentes por ~4
   (atlas_tentativas_mc.py); p_t = 1 - (1 - p_H)^(4 Delta_nu T). Com frequência fixa, 1 tentativa.
3. Detecções a frequência fixa escolhidas depois (5 observações de 2023 da 4U 0115+63): também a
   probabilidade corrigida para as observações da mesma fonte; abaixo de 1e-3 só sem essa correção =
   "marginal". PF_UL >= 1 = sem restrição.
4. PF_rms das detecções, dos harmônicos salvos (0,3-10 keV, k = 1-5): o viés de ruído 2 sig^2 é descontado
   da soma das potências e só a soma é truncada em zero (truncar cada harmônico antes de somar daria viés
   positivo mesmo sem sinal).

Uso: python atlas_correcoes.py <resultados>   (escreve <resultados>/atlas/correcoes.json)
"""
import csv
import json
import os
import sys

import numpy as np
from scipy.optimize import brentq
from scipy.stats import ncx2

RES = sys.argv[1]
OBS = os.path.join(RES, "atlas", "obs")
P3S = 0.00135          # cauda de 3 sigma


def z2_perfil(c, fe, kmax=2):
    """Z^2_kmax de um perfil binado, com as contagens corrigidas pela exposição por fase."""
    nb = c.size
    w = c / np.maximum(fe * nb, 1e-12)
    w *= c.sum() / w.sum()
    fi = 2 * np.pi * (np.arange(nb) + 0.5) / nb
    return 2.0 / c.sum() * sum(np.sum(w * np.cos(k * fi)) ** 2 + np.sum(w * np.sin(k * fi)) ** 2
                               for k in range(1, kmax + 1))


def pf_lim_sup(z, n, b, dof=4):
    if ncx2.cdf(z, dof, 1e-9) <= P3S:     # nem lambda ~ 0 é compatível: não deveria ocorrer
        return 0.0
    lam = brentq(lambda l: ncx2.cdf(z, dof, l) - P3S, 1e-9, 1e7)
    return float(np.sqrt(lam / n) * n / max(n - b, 1.0))


if __name__ == "__main__":
    cat = list(csv.DictReader(open(os.path.join(RES, "atlas", "catalogo_obs.csv"))))
    harm = {}
    for h in csv.DictReader(open(os.path.join(RES, "atlas", "harmonicos.csv"))):
        if h["banda"] == "0.3-10" and int(h["k"]) <= 5:
            harm.setdefault(h["obsid"], []).append(h)
    out = {}
    por_fonte = {}
    for x in cat:
        por_fonte.setdefault(x["fonte"], set()).add(x["obsid"])
    fixos = [x["obsid"] for x in cat if x["estado"] == "ok"
             and "tempo" in json.load(open(os.path.join(OBS, f"{x['obsid']}.json")))
             and (json.load(open(os.path.join(OBS, f"{x['obsid']}.json"))).get("nu_esperado") or {}).get("fixo")]
    for x in cat:
        if x["estado"] != "ok":
            continue
        o = x["obsid"]
        j = json.load(open(os.path.join(OBS, f"{o}.json")))
        if "tempo" not in j:        # PSR B1259-63: sem busca de pulsação
            continue
        t = j["tempo"]
        fixo = o in fixos
        dur = float(j["duracao_s"])
        n_ind = 1.0 if fixo else max(1.0, 4 * (t["busca_nu_max"] - t["busca_nu_min"]) * dur)
        p_t = float(-np.expm1(n_ind * np.log1p(-min(t["H_p"], 1 - 1e-16))))
        r = {"H_p": t["H_p"], "n_indep": n_ind, "p_tentativas": p_t, "frequencia_fixa": fixo,
             "detectado": p_t < 1e-3}
        if fixo:
            # família: as observações da mesma fonte dobradas a frequência fixa (4U 0115+63 em 2023: 5)
            fam = [y for y in fixos if y in por_fonte[x["fonte"]]]
            r["n_familia_fixos"] = len(fam)
            r["p_corrigido_n_fixos"] = float(-np.expm1(len(fam) * np.log1p(-t["H_p"])))
            r["marginal"] = bool(p_t < 1e-3 and r["p_corrigido_n_fixos"] >= 1e-3)
        z = np.load(os.path.join(OBS, f"{o}_perfis.npz"))
        n_ev = float(j["n_eventos"])
        b_ev = float(j["fracao_fundo"]["0.3-10"]) * n_ev
        if "0.3-10__contagens" in z:
            c, fe = z["0.3-10__contagens"], z["0.3-10__fracao_exposicao"]
            zz = z2_perfil(c, fe)
        else:
            # sem perfil salvo: o limite de 3 sigma que atlas_obs.py calcula do z observado nos eventos
            zz = np.nan
            if not r["detectado"] and j.get("pf_rms_lim_sup_0.3-10") is not None:
                r["pf_rms_lim_sup_3sigma"] = j["pf_rms_lim_sup_0.3-10"]
                r["sem_restricao"] = bool(j["pf_rms_lim_sup_0.3-10"] >= 1.0)
        r["z2_2_perfil"] = float(zz)
        if o in harm and len(harm[o]) == 5:
            pot = sum(float(h["r"]) ** 2 - 2 * float(h["sig"]) ** 2 for h in harm[o])
            r["pf_rms"] = float(np.sqrt(max(pot, 0.0) / 2))
        if not r["detectado"] and zz == zz:
            lim = pf_lim_sup(zz, n_ev, b_ev)
            r["pf_rms_lim_sup_3sigma"] = lim
            r["sem_restricao"] = bool(lim >= 1.0)
        out[o] = r
    json.dump({"obs": out, "fixos": fixos}, open(os.path.join(RES, "atlas", "correcoes.json"), "w"), indent=1)
    for o, r in out.items():
        print(o, f"p_H={r['H_p']:.2e} n_ind={r['n_indep']:.0f} p_t={r['p_tentativas']:.2e}",
              "det" if r["detectado"] else "---", "fixo" if r["frequencia_fixa"] else "",
              f"p5={r.get('p_corrigido_n_fixos', float('nan')):.1e}" if r["frequencia_fixa"] else "",
              "MARGINAL" if r.get("marginal") else "",
              f"PF_UL={r.get('pf_rms_lim_sup_3sigma', float('nan')):.2f}" if not r["detectado"] else "")
