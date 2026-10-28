"""Frequência de spin esperada em cada observação XMM, pelo histórico do Fermi/GBM.

O GBM publica nu(t) no referencial do pulsar (com as mesmas órbitas de bexrb/fontes.py, exceto
V 0332+53). Interpolamos linearmente entre detecções que cercam a observação a menos de 10 d;
sem cerco, aceitamos a detecção mais próxima a menos de 3 d. X Per 0151380101: período de
La Palombara & Mereghetti (2007), medido nestes mesmos dados com as três câmeras.

Saída: dados/nu_esperado.json  {obsid: {nu, sig, origem, ...}}
"""
import csv
import json
import os

import numpy as np
from astropy.io import fits

AQUI = os.path.dirname(os.path.abspath(__file__))
GBM = {"4U 0115+63": "4u0115", "V 0332+53": "v0332", "1A 0535+262": "a0535",
       "Swift J0243.6+6124": "swiftj0243"}

out = {}
for linha in csv.DictReader(open(os.path.join(AQUI, "dados", "mjd_obs.csv"))):
    nome, obsid, mjd = linha["fonte"], linha["obsid"], linha["mjd_meio"]
    if nome not in GBM or not mjd:
        continue
    mjd = float(mjd)
    d = fits.getdata(os.path.join(AQUI, "dados", "gbm", GBM[nome] + ".fits.gz"), "PSR_HIST")
    ok = d["DETECTED"] == 1
    t, nu, sg = d["BARYTIME"][ok], d["FREQUENCY"][ok], d["FREQUENCY_ERR"][ok]
    antes, depois = np.where((t <= mjd) & (t > mjd - 10))[0], np.where((t > mjd) & (t < mjd + 10))[0]
    if antes.size and depois.size:
        i, j = antes[np.argmax(t[antes])], depois[np.argmin(t[depois])]
        w = (mjd - t[i]) / (t[j] - t[i])
        out[obsid] = {"nu": float(nu[i] + w * (nu[j] - nu[i])), "sig": float(np.hypot(sg[i], sg[j])),
                      "origem": "GBM, interpolado", "mjd_gbm": [float(t[i]), float(t[j])]}
    else:
        k = np.argmin(np.abs(t - mjd)) if t.size else None
        if k is not None and abs(t[k] - mjd) < 3:
            out[obsid] = {"nu": float(nu[k]), "sig": float(sg[k]), "origem": "GBM, mais próximo",
                          "mjd_gbm": [float(t[k])]}
# 4U 0115+63 em 2023: fora do GBM; o spin em quiescência quase não muda (3,61414-3,61428 s em
# 2016-2023 nas detecções do atlas). Dobramos as observações fracas de julho-setembro de 2023 na
# frequência da detecção forte de 0883351701 (H_p ~ 1e-122), sem busca e sem tentativas.
for o in ("0883350301", "0883350401", "0883350901", "0883351001", "0883351601"):
    out[o] = {"nu": 1 / 3.6141858, "sig": 5.4e-5 / 3.6141858 ** 2, "fixo": True,
              "origem": "atlas XMM 0883351701 (2023-09-12), detecção forte"}
out["0151380101"] = {"nu": 1 / 839.3, "sig": 0.3 / 839.3 ** 2, "fixo": True,
                     "origem": "La Palombara & Mereghetti 2007: P = 839,3 +- 0,3 s (mesmos dados)"}
json.dump(out, open(os.path.join(AQUI, "dados", "nu_esperado.json"), "w"), indent=1)
for k, v in out.items():
    print(k, v["origem"], 1 / v["nu"])
