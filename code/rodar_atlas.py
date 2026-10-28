"""Roda bexrb.atlas_obs em todas as observações e junta o catálogo.

Uso: python rodar_atlas.py <raiz_produtos> <saida> [--proc N] [--so OBSID ...]
Saída:
  <saida>/obs/<ObsID>.json            tudo o que foi medido
  <saida>/obs/<ObsID>_perfis.npz      perfis dobrados por banda (32 bins) e a busca em frequência
  <saida>/catalogo_obs.csv            uma linha por observação
  <saida>/harmonicos.csv              obs x banda x k
  <saida>/pf_energia.csv              obs x faixa de energia
"""
import argparse
import csv
import glob
import json
import os
import traceback
from multiprocessing import Pool

from bexrb.atlas_obs import _json_seguro, analisar


def um(args):
    pasta, saida = args
    try:
        r = analisar(pasta, saida)
    except Exception:
        r = {"obsid": os.path.basename(pasta.rstrip("/")), "estado": "erro",
             "erro": traceback.format_exc(limit=3)}
    with open(os.path.join(saida, f"{r['obsid']}.json"), "w") as f:
        json.dump(r, f, indent=1, default=_json_seguro)
    return r["obsid"], r["estado"]


def catalogo(saida):
    linhas, harm, pfe = [], [], []
    for f in sorted(glob.glob(os.path.join(saida, "obs", "*.json"))):
        r = json.load(open(f))
        base = {k: r.get(k) for k in ("fonte", "obsid", "estado", "datamode", "submode", "filtro",
                                      "mjd_meio", "duracao_s", "expo_viva_s", "fase_orbital",
                                      "dias_desde_periastro", "n_eventos", "pileup_reducao",
                                      "taxa_bruta_por_limite_pileup", "rms_frac_100s",
                                      "cor_mole", "cor_dura")}
        for b in ("0.3-10", "0.3-2", "2-10"):
            base[f"taxa_{b}"] = (r.get("taxa") or {}).get(b)
            base[f"fluxo_{b}"] = (r.get("fluxo_desdobrado_cgs") or {}).get(b)
            base[f"frac_fundo_{b}"] = (r.get("fracao_fundo") or {}).get(b)
        t = r.get("tempo") or {}
        base.update({"periodo_s": t.get("periodo_s"), "sig_nu_hz": t.get("sig_nu_hz"),
                     "H": t.get("H"), "H_p": t.get("H_p"), "H_p_tent": t.get("H_p_com_tentativas"),
                     "z2_orbita": t.get("z2_4_pico"), "z2_sem_orbita": t.get("z2_4_sem_orbita_nu_reducao"),
                     "nudot_toa": (t.get("toa") or {}).get("nudot"),
                     "sig_nudot_toa": (t.get("toa") or {}).get("sig_nudot")})
        for b in ("0.3-10", "0.3-1", "2-10"):
            p = (r.get("pf") or {}).get(b) or {}
            base[f"pf_rms_{b}"] = p.get("pf_rms")
            base[f"sig_pf_rms_{b}"] = p.get("sig_pf_rms")
            base[f"pf_minmax_{b}"] = p.get("pf_minmax")
        h = (r.get("harmonicos") or {}).get("0.3-10")
        if h:
            for k in range(4):
                base[f"r{k+1}"] = h[k]["r_corr"]
        a = (r.get("atrasos_vs_2_10") or {}).get("0.3-1") or {}
        base["atraso1_mole"], base["sig_atraso1_mole"] = (a.get("k1") or (None, None))
        base["atraso2_mole"], base["sig_atraso2_mole"] = (a.get("k2") or (None, None))
        linhas.append(base)
        for banda, hh in (r.get("harmonicos") or {}).items():
            for x in hh:
                harm.append({"fonte": r["fonte"], "obsid": r["obsid"], "banda": banda, **x})
        for x in r.get("pf_energia") or []:
            pfe.append({"fonte": r["fonte"], "obsid": r["obsid"], **x})
    for nome, dados in (("catalogo_obs.csv", linhas), ("harmonicos.csv", harm), ("pf_energia.csv", pfe)):
        if not dados:
            continue
        chaves = list(dict.fromkeys(k for d in dados for k in d))
        with open(os.path.join(saida, nome), "w", newline="") as f:
            w = csv.DictWriter(f, chaves)
            w.writeheader()
            w.writerows(dados)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("raiz")
    ap.add_argument("saida")
    ap.add_argument("--proc", type=int, default=8)
    ap.add_argument("--so", nargs="*")
    a = ap.parse_args()
    obs_dir = os.path.join(a.saida, "obs")
    os.makedirs(obs_dir, exist_ok=True)
    pastas = sorted(glob.glob(os.path.join(a.raiz, "*", "0*")))
    if a.so:
        pastas = [p for p in pastas if os.path.basename(p) in a.so]
    # as maiores primeiro, para equilibrar o pool
    pastas.sort(key=lambda p: -sum(os.path.getsize(x) for x in glob.glob(os.path.join(p, "epn_*_bary_source.fits"))))
    with Pool(a.proc) as pool:
        for obsid, estado in pool.imap_unordered(um, [(p, obs_dir) for p in pastas]):
            print(obsid, estado, flush=True)
    catalogo(a.saida)
