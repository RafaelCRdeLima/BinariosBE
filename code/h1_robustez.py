"""Robustez do H1 hierárquico (h1_hierarquico.py):
  1. deixar uma fonte de fora (leave-one-source-out), para cada normalização;
  2. sensibilidade da normalização dipolar ao D adotado (cenários: reversão do torque válida ou propeller válido);
  3. recuperação com dados sintéticos no mesmo desenho (fontes, indicadores, trabalhos e erros reais), com
     sigma_fonte verdadeiro = 0; 0,3; 0,5 dex e os demais componentes nas medianas observadas: taxa de falso
     positivo e poder do teste sigma_fonte = 0, e viés dos componentes.
Tudo na amostra "limpa" (sem duplicatas e sem linhas que não marcam o início da coluna).

Uso: python h1_robustez.py <resultados>   (escreve <resultados>/sintese/h1_robustez.json)
"""
import json
import os
import sys

import numpy as np
from scipy.stats import chi2

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import h1_hierarquico as H  # noqa: E402

RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(AQUI, "resultados")
H.RES = RES
AMOSTRA = "limpa"     # sem duplicatas e sem linhas que não marcam o início da coluna (h1_hierarquico.py)
IND = json.load(open(os.path.join(RES, "sintese", "indice.json")))["fontes"]
D_BASE = {"4U 0115+63": IND["4U 0115+63"]["P_Tsygankov2016"]["D"]["med"],
          "V 0332+53": IND["V 0332+53"]["P_Tsygankov2016"]["D"]["med"],
          "1A 0535+262": IND["1A 0535+262"]["T2_GL79"]["D"]["med"],
          "Swift J0243.6+6124": IND["Swift J0243.6+6124"]["T2_GL79"]["D"]["med"]}


def ajustar(tr, y, grade, y_nome="L"):
    r = H.ajustar(tr, y, grade, y_nome)
    return {"n_fontes": r["n_fontes"], "sigma_fonte": r["sigma_fonte"], "sigma_trabalho": r["sigma_trabalho"],
            "sigma_indicador": r["sigma_indicador"], "p": r["p_LRT_mistura"], "LRT": r["LRT_sigma_fonte_0"]}


if __name__ == "__main__":
    SG = H.SG
    fina = np.array(np.meshgrid(SG, SG, SG, SG, indexing="ij")).reshape(4, -1).T
    SGc = np.arange(0.0, H.TETO + 1e-9, 0.2)
    grossa = np.array(np.meshgrid(SGc, SGc, SGc, SGc, indexing="ij")).reshape(4, -1).T
    out = {"loso": {}, "dipolo": [], "sinteticos": {}}

    # ---------- 1. leave-one-source-out ----------
    tr = H.carregar(D_BASE, AMOSTRA)
    for y_nome in ("L", "becker", "mushtukov", "dipolo"):
        for fora in sorted(set(t["fonte"] for t in tr)):
            t2 = [t for t in tr if t["fonte"] != fora]
            r = ajustar(t2, np.array([t[y_nome] for t in t2]), fina, y_nome)
            out["loso"][f"{y_nome} sem {fora}"] = r
            print(f"LOSO {y_nome:9s} sem {fora:20s} N={len(t2):2d} sigma_fonte={r['sigma_fonte']['med']:.2f} "
                  f"[{r['sigma_fonte']['q16']:.2f}-{r['sigma_fonte']['q84']:.2f}] p={r['p']:.3f}", flush=True)

    # ---------- 2. normalização dipolar: sensibilidade ao D ----------
    DS, DA = H.grade_d(RES)
    for ds in DS:
        for da in DA:
            dd = dict(D_BASE, **{"Swift J0243.6+6124": ds, "1A 0535+262": da})
            t2 = H.carregar(dd, AMOSTRA)
            r = ajustar(t2, np.array([t["dipolo"] for t in t2]), fina, "dipolo")
            out["dipolo"].append({"D_Swift": ds, "D_1A0535": da, **r})
            print(f"dipolo D_Swift={ds:5.2f} D_1A={da:4.2f} sigma_fonte={r['sigma_fonte']['med']:.2f} p={r['p']:.3f}",
                  flush=True)

    # ---------- 3. dados sintéticos ----------
    rng = np.random.default_rng(1)
    # componentes não testados nas medianas observadas da amostra (normalização de Becker com B_cyc)
    obs = json.load(open(os.path.join(RES, "sintese", "h1_hierarquico.json")))[AMOSTRA]["becker"]
    S_IND, S_TRAB, S_RES = (obs[k]["med"] for k in ("sigma_indicador", "sigma_trabalho", "sigma_residual"))
    out["sinteticos_componentes"] = {"indicador": S_IND, "trabalho": S_TRAB, "residual": S_RES}
    LM = np.linalg.cholesky(H.cov_medida(tr, "becker"))
    rot = {k: np.unique([t[k] for t in tr], return_inverse=True)[1] for k in ("fonte", "tipo", "ref")}
    for s_f in (0.0, 0.3, 0.5):
        ps, med_f, med_t = [], [], []
        for _ in range(200):
            y = (rng.normal(0, s_f, rot["fonte"].max() + 1)[rot["fonte"]]
                 + rng.normal(0, S_IND, rot["tipo"].max() + 1)[rot["tipo"]]
                 + rng.normal(0, S_TRAB, rot["ref"].max() + 1)[rot["ref"]]
                 + rng.normal(0, S_RES, LM.shape[0]) + LM @ rng.standard_normal(LM.shape[0]))
            r = ajustar(tr, y, grossa, "becker")
            ps.append(r["p"])
            med_f.append(r["sigma_fonte"]["med"])
            med_t.append(r["sigma_trabalho"]["med"])
        ps = np.array(ps)
        out["sinteticos"][f"sigma_fonte={s_f}"] = {
            "frac_p<0.05": float(np.mean(ps < 0.05)), "frac_p<0.15": float(np.mean(ps < 0.15)),
            "sigma_fonte_mediana_recuperada": [float(np.percentile(med_f, q)) for q in (16, 50, 84)],
            "sigma_trabalho_mediana_recuperada": [float(np.percentile(med_t, q)) for q in (16, 50, 84)]}
        print("sintético", s_f, out["sinteticos"][f"sigma_fonte={s_f}"], flush=True)
    json.dump(out, open(os.path.join(RES, "sintese", "h1_robustez.json"), "w"), indent=1)
