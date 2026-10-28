"""Medidas diretas de uma observação EPIC-pn reduzida com o SAS, sem ajuste espectral.

Para cada ObsID:
  - taxas líquidas e cores por banda (fundo da região da redução, escalado por BACKSCAL);
  - fluxo "desdobrado" pela ARF: F = sum(E_i / A(E_i)) / T_vivo, menos o mesmo termo do fundo.
    Ignora a redistribuição; serve acima de ~2 keV como estimativa sem modelo;
  - para os pulsares: tempos demodulados da órbita, frequência refinada (Z^2 + tempos de chegada
    por segmento), H-test, harmônicos 1-6 por banda (fração e fase), PF_rms e PF_minmax,
    PF(E) e atrasos de fase de cada banda em relação a 2-10 keV, perfis dobrados com a exposição
    por fase descontada.

Uso: python -m bexrb.atlas_obs <pasta_da_obs> <saida_dir>
"""
import glob
import json
import os
import sys

import numpy as np
from astropy.io import fits
from scipy.optimize import brentq
from scipy.stats import ncx2

from .fontes import POR_PASTA
from .orbita import DIA, demodular, fase_orbital

KEV_ERG = 1.602176634e-9
BANDAS = {                     # keV; as bandas de perfil e de cor
    "0.3-1": (0.3, 1.0), "1-2": (1.0, 2.0), "2-4": (2.0, 4.0), "4-7": (4.0, 7.0),
    "7-10": (7.0, 10.0), "0.3-2": (0.3, 2.0), "2-10": (2.0, 10.0), "4-10": (4.0, 10.0),
    "0.3-10": (0.3, 10.0),
}
BORDAS_PFE = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.5, 10.0]
K_MAX = 6          # harmônicos guardados
K_PF = 5           # harmônicos no PF_rms e na reconstrução do perfil (PF_minmax)
NBINS = 32
# limiares de taxa de pileup do pn (XMM Users Handbook, tabela 3; 1% de perda de fluxo/padrão)
_ARQ_NU = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dados", "nu_esperado.json")
NU_ESPERADO = json.load(open(_ARQ_NU)) if os.path.exists(_ARQ_NU) else {}
PILEUP_LIM = {"PRIMEFULLWINDOW": 6.0, "PRIMEFULLWINDOWEXTENDED": 6.0, "PRIMELARGEWINDOW": 10.0,
              "PRIMESMALLWINDOW": 50.0, "FASTTIMING": 800.0, "TIMING": 800.0, "FASTBURST": 60000.0}


def _maior_arquivo(pasta, padrao):
    arqs = glob.glob(os.path.join(pasta, padrao))
    return max(arqs, key=os.path.getsize) if arqs else None


def _gti_intervalos(hdul, ccd):
    """Interseção do GTI de flares (GTI000xx) com o STDGTI do CCD da fonte, em tempo baricêntrico."""
    def ler(nome):
        d = hdul[nome].data
        return np.column_stack([d["START"], d["STOP"]]) if d is not None and len(d) else None
    nomes = [h.name for h in hdul]
    std = next((n for n in nomes if n == f"STDGTI{ccd:02d}"), None)
    fl = next((n for n in nomes if n.startswith("GTI0")), None)
    a = ler(std) if std else None
    b = ler(fl) if fl else None
    if a is None:
        return b
    if b is None:
        return a
    out = []
    for s1, e1 in a:
        for s2, e2 in b:
            s, e = max(s1, s2), min(e1, e2)
            if e > s:
                out.append((s, e))
    return np.array(out)


def _arf(pasta):
    f = _maior_arquivo(pasta, "epn_*_src.arf")
    d = fits.getdata(f, "SPECRESP")
    e = 0.5 * (d["ENERG_LO"] + d["ENERG_HI"])
    return e, d["SPECRESP"]


def _fundo(pasta, expo_src):
    """Contagens de fundo por canal (5 eV) já escaladas para a área da fonte."""
    fs = _maior_arquivo(pasta, "epn_*_src_spec.fits")
    fb = _maior_arquivo(pasta, "epn_*_bkg_spec.fits")
    hs, hb = fits.getheader(fs, "SPECTRUM"), fits.getheader(fb, "SPECTRUM")
    cont = fits.getdata(fb, "SPECTRUM")["COUNTS"].astype(float)
    escala = hs["BACKSCAL"] / hb["BACKSCAL"] * expo_src / hb["EXPOSURE"]
    e_canal = (np.arange(cont.size) * 5 + 2.5) / 1000.0
    return e_canal, cont * escala, float(escala), float(hs["EXPOSURE"])


def _espectro_canais(pi_ev):
    return np.bincount((pi_ev // 5).astype(int), minlength=4096)[:4096].astype(float)


def _extrair(pasta, fev, expo):
    """Eventos da fonte e fundo por canal de 5 eV, já escalado para a área da fonte.

    - imaging abaixo do limite de pileup: a extração da redução (círculo de 30" e fundo em anel de 60-120");
    - imaging acima do limite: anel no mesmo centro, raio interno 7,5"/12,5"/17,5" para taxa
      bruta até 3x/8x/acima do limite; fundo da redução escalado pela área do anel. O fluxo do
      anel não representa a fonte e não é registrado;
    - timing: colunas RAWX centradas no pico de 2-10 keV (procurado nas colunas 8-57), +-10;
      fundo = média por coluna das colunas a >= 15 do pico (6-59), escalada para as da fonte.
      A janela fixa da redução (RAWX 27-47, fundo 3-5) erra a fonte quando o apontamento está
      deslocado e pega ruído de borda no fundo.
    """
    with fits.open(fev, memmap=True) as h:
        hd = h["EVENTS"].header
        ccd = int(np.bincount(np.asarray(h["EVENTS"].data["CCDNR"], int)).argmax())
        gti = _gti_intervalos(h, ccd)
    modo = str(hd.get("DATAMODE", "")).upper()
    submodo = str(hd.get("SUBMODE", "")).upper()
    info = {"metodo": "reducao"}
    if modo == "TIMING":
        fc = _maior_arquivo(pasta, "epn_*_clean_bary.fits")
        d = fits.getdata(fc, "EVENTS")
        rx, pi = np.asarray(d["RAWX"], int), np.asarray(d["PI"], float)
        dura = (rx >= 1) & (pi >= 2000) & (pi < 10000)
        hist = np.bincount(rx[dura], minlength=65)
        cols = np.arange(65)
        busca = (cols >= 8) & (cols <= 57)
        c = int(cols[busca][np.argmax(hist[busca])])
        fundo_cols = (np.abs(cols - c) >= 15) & (cols >= 6) & (cols <= 59)
        mediana = float(np.median(hist[fundo_cols]))
        signif = (hist[c] - mediana) / np.sqrt(max(mediana, 1.0))
        src_cols = (np.abs(cols - c) <= 10) & (cols >= 1) & (cols <= 64)
        info = {"metodo": "colunas", "rawx_pico": c, "signif_pico": float(signif),
                "n_col_fonte": int(src_cols.sum()), "n_col_fundo": int(fundo_cols.sum())}
        sel_f = src_cols[np.clip(rx, 0, 64)] & (rx >= 1)
        sel_b = fundo_cols[np.clip(rx, 0, 64)] & (rx >= 1)
        t = np.asarray(d["TIME"], float)[sel_f]
        e = pi[sel_f] / 1000.0
        escala = src_cols.sum() / fundo_cols.sum()
        b_c = _espectro_canais(pi[sel_b]) * escala
        if signif < 5:
            info["nao_detectada"] = True
    else:
        with fits.open(fev, memmap=True) as h:
            d = h["EVENTS"].data
            t = np.asarray(d["TIME"], float)
            e = np.asarray(d["PI"], float) / 1000.0
        e_c, b_c, escala, _ = _fundo(pasta, expo)
        lim = PILEUP_LIM.get(submodo)
        razao = t[(e >= 0.3) & (e < 10)].size / expo / lim if lim else 0.0
        if razao > 1.0:
            s = json.load(open(os.path.join(pasta, "session.json")))
            g = s["steps"]["regions"]["parameters"]["source_geometry"]
            r_in = (7.5 if razao <= 3 else 12.5 if razao <= 8 else 17.5) / 0.05
            fc = _maior_arquivo(pasta, "epn_*_clean_bary.fits")
            dd = fits.getdata(fc, "EVENTS")
            r2 = (dd["X"] - g["x"]) ** 2 + (dd["Y"] - g["y"]) ** 2
            sel = (r2 >= r_in ** 2) & (r2 <= g["radius"] ** 2)
            t = np.asarray(dd["TIME"], float)[sel]
            e = np.asarray(dd["PI"], float)[sel] / 1000.0
            f_area = 1 - (r_in / g["radius"]) ** 2
            b_c = b_c * f_area
            escala = escala * f_area
            info = {"metodo": "anel", "r_in_arcsec": r_in * 0.05, "r_out_arcsec": g["radius"] * 0.05,
                    "razao_pileup": razao}
    sel = (e >= 0.3) & (e < 10.0)
    e_c = (np.arange(4096) * 5 + 2.5) / 1000.0
    return t[sel], e[sel], e_c, b_c, float(escala), hd, gti, ccd, info


def _harmonicos(fase, peso, n_liq, n_tot, kmax=K_MAX):
    """Coeficientes fracionários: perfil/media = 1 + sum r_k cos(2 pi k (phi - psi_k))."""
    out = []
    sig = np.sqrt(2.0 * n_tot) / n_liq if n_liq > 0 else np.nan
    for k in range(1, kmax + 1):
        ang = 2 * np.pi * k * fase
        a = 2.0 * np.sum(peso * np.cos(ang)) / n_liq
        b = 2.0 * np.sum(peso * np.sin(ang)) / n_liq
        r = np.hypot(a, b)
        psi = np.mod(np.arctan2(b, a) / (2 * np.pi * k), 1.0 / k)
        out.append({"k": k, "a": a, "b": b, "r": r, "sig": sig,
                    "r_corr": np.sqrt(max(r * r - 2 * sig * sig, 0.0)),
                    "psi": psi, "sig_psi": sig / max(r, 1e-30) / (2 * np.pi * k)})
    return out


def _pf(h, k_rec=K_PF):
    """PF_rms com K_PF harmônicos (viés de ruído descontado da soma, e só a soma truncada em zero); PF_minmax do perfil
    reconstruído com os k_rec harmônicos que o H-test escolheu (com poucos fótons, mais harmônicos só põem
    ruído)."""
    rms = np.sqrt(max(sum(x["r"] ** 2 - 2 * x["sig"] ** 2 for x in h[:K_PF]), 0.0) / 2.0)
    fi = np.linspace(0, 1, 512, endpoint=False)
    perfil = 1 + sum(x["r"] * np.cos(2 * np.pi * x["k"] * (fi - x["psi"])) for x in h[:k_rec])
    mm = (perfil.max() - perfil.min()) / (perfil.max() + perfil.min())
    sig = h[0]["sig"]
    return float(rms), float(sig / np.sqrt(2)), float(mm)


def _z2_binado(t, nu, nb=64, kmax=4):
    fase = np.mod(t * nu, 1.0)
    c = np.bincount((fase * nb).astype(int), minlength=nb).astype(float)
    ang = 2 * np.pi * (np.arange(nb) + 0.5) / nb
    n = c.sum()
    return 2.0 / n * sum(np.sum(c * np.cos(k * ang)) ** 2 + np.sum(c * np.sin(k * ang)) ** 2
                         for k in range(1, kmax + 1))


def _htest(fase, mmax=20):
    n = fase.size
    z, h, melhor = 0.0, -np.inf, 1
    for m in range(1, mmax + 1):
        ang = 2 * np.pi * m * fase
        z += 2.0 / n * (np.sum(np.cos(ang)) ** 2 + np.sum(np.sin(ang)) ** 2)
        if z - 4 * m + 4 > h:
            h, melhor = z - 4 * m + 4, m
    return float(h), melhor, float(np.exp(-0.4 * h))  # de Jager & Büsching 2010


def _busca_frequencia(t, nu0, meia, dur, rng):
    """Z^2_4 binado em nu0 +- meia, passo 1/(8T); amostra até 2e6 eventos."""
    if t.size > 2_000_000:
        t = rng.choice(t, 2_000_000, replace=False)
    grade = np.arange(nu0 - meia, nu0 + meia, 1.0 / (8 * dur))
    z = np.array([_z2_binado(t, nu) for nu in grade])
    i = int(np.argmax(z))
    fina = np.linspace(grade[max(i - 2, 0)], grade[min(i + 2, grade.size - 1)], 81)
    zf = np.array([_z2_binado(t, nu) for nu in fina])
    return float(fina[np.argmax(zf)]), float(zf.max()), grade, z


def _tempos_de_chegada(t, nu, nseg):
    """Fase do 1º harmônico em segmentos iguais de tempo; regressão dá delta_nu e nu-ponto."""
    lim = np.linspace(t.min(), t.max(), nseg + 1)
    tm, ph, sp = [], [], []
    for a, b in zip(lim[:-1], lim[1:]):
        s = (t >= a) & (t < b)
        if s.sum() < 200:
            continue
        ang = 2 * np.pi * np.mod(t[s] * nu, 1.0)
        C, S = np.sum(np.cos(ang)), np.sum(np.sin(ang))
        r = 2 * np.hypot(C, S) / s.sum()
        if r < 4 * np.sqrt(2.0 / s.sum()):
            continue
        tm.append(np.mean(t[s]))
        ph.append(np.arctan2(S, C) / (2 * np.pi))
        sp.append(np.sqrt(2.0 / s.sum()) / r / (2 * np.pi))
    if len(tm) < 4:
        return None
    tm, ph, sp = map(np.array, (tm, ph, sp))
    ph = np.unwrap(ph * 2 * np.pi) / (2 * np.pi)
    w = 1 / sp ** 2
    res = {}
    for grau in (1, 2):
        A = np.vander(tm, grau + 1)
        cov = np.linalg.inv(A.T @ (A * w[:, None]))
        c = cov @ (A.T @ (w * ph))
        chi2 = float(np.sum(w * (ph - A @ c) ** 2))
        res[grau] = (c, cov, chi2)
    c1, cov1, chi1 = res[1]
    c2, cov2, chi2 = res[2]
    # dispersão das fases maior que o erro de contagem (ruído de pulso a pulso): escala os erros
    esc = np.sqrt(max(1.0, chi1 / max(len(tm) - 2, 1)))
    cov1 = cov1 * esc ** 2
    cov2 = cov2 * max(1.0, chi2 / max(len(tm) - 3, 1))
    # fase cresce com +nu*t; um atraso de fase crescente = frequência menor
    return {"n_seg": int(len(tm)), "dnu": float(-c1[0]), "sig_dnu": float(np.sqrt(cov1[0, 0])),
            "chi2_lin": chi1, "dof_lin": int(len(tm) - 2),
            "nudot": float(-2 * c2[0]), "sig_nudot": float(2 * np.sqrt(cov2[0, 0])),
            "chi2_quad": chi2, "max_desvio_fase": float(np.max(np.abs(ph - np.mean(ph))))}


def _exposicao_por_fase(gti, nu, nb):
    """Fração do tempo vivo em cada bin de fase (amostragem fina dentro dos GTIs)."""
    if gti is None or not len(gti):
        return np.full(nb, 1.0 / nb)
    passo = 1.0 / nu / (nb * 16)
    tot = np.zeros(nb)
    for s, e in gti:
        n = int((e - s) / passo)
        if n <= 0:
            continue
        tt = s + (np.arange(n) + 0.5) * passo
        tot += np.bincount((np.mod(tt * nu, 1.0) * nb).astype(int), minlength=nb)
    return tot / tot.sum()


def analisar(pasta_obs, saida):
    pasta_obs = os.path.abspath(pasta_obs)
    obsid = os.path.basename(pasta_obs.rstrip("/"))
    fonte = POR_PASTA[os.path.basename(os.path.dirname(pasta_obs))]
    fev = _maior_arquivo(pasta_obs, "epn_*_bary_source.fits")
    rng = np.random.default_rng(int(obsid))
    res = {"obsid": obsid, "fonte": fonte.nome, "arquivo": os.path.basename(fev) if fev else None}
    if fev is None:
        res["estado"] = "sem lista de eventos do pn"
        return res
    expo = float(fits.getheader(_maior_arquivo(pasta_obs, "epn_*_src_spec.fits"), "SPECTRUM")["EXPOSURE"])
    t, e, e_c, b_c, escala_t, hd, gti, ccd, extracao = _extrair(pasta_obs, fev, expo)
    res["extracao"] = extracao
    mjdref = hd["MJDREF"]
    submodo = str(hd.get("SUBMODE", "")).upper()
    t0, t1 = float(t.min()), float(t.max())
    mjd_meio = mjdref + 0.5 * (t0 + t1) / DIA
    fo, nu_verd = fase_orbital(mjd_meio, fonte.orbita)
    res.update({
        "estado": "ok", "datamode": hd.get("DATAMODE"), "submode": hd.get("SUBMODE"),
        "filtro": hd.get("FILTER"), "ccd": ccd, "expo_viva_s": expo, "duracao_s": t1 - t0,
        "mjd_ini": mjdref + t0 / DIA, "mjd_fim": mjdref + t1 / DIA, "mjd_meio": mjd_meio,
        "fase_orbital": float(fo), "anomalia_verdadeira_deg": float(nu_verd),
        "dias_desde_periastro": float(fo * fonte.orbita.porb_d),
        "escala_fundo": escala_t, "n_eventos": int(t.size),
    })
    sess = os.path.join(pasta_obs, "session.json")
    if os.path.exists(sess):
        s = json.load(open(sess))
        pu = s["steps"].get("pileup", {}).get("parameters", {}).get("result", {})
        res["pileup_reducao"] = pu.get("verdict")
        res["regiao_fonte"] = s["steps"].get("regions", {}).get("parameters", {}).get("source_description")

    # taxas, cores e fluxo desdobrado
    ea, aa = _arf(pasta_obs)
    a_ev = np.interp(e, ea, aa)
    a_c = np.interp(e_c, ea, aa)
    taxas, fluxos = {}, {}
    for nome, (lo, hi) in BANDAS.items():
        s = (e >= lo) & (e < hi)
        sb = (e_c >= lo) & (e_c < hi)
        n, b = float(s.sum()), float(b_c[sb].sum())
        taxas[nome] = ((n - b) / expo, np.sqrt(n + b * escala_t) / expo, b / max(n, 1))
        ok = s & (a_ev > 1.0)
        f = (np.sum(e[ok] / a_ev[ok]) - np.sum((b_c * e_c / np.maximum(a_c, 1.0))[sb & (a_c > 1.0)]))
        fluxos[nome] = f / expo * KEV_ERG
    res["taxa"] = {k: v[0] for k, v in taxas.items()}
    res["taxa_err"] = {k: v[1] for k, v in taxas.items()}
    res["fracao_fundo"] = {k: v[2] for k, v in taxas.items()}
    res["fluxo_desdobrado_cgs"] = fluxos
    r = res["taxa"]
    res["cor_mole"] = r["2-4"] / r["0.3-2"] if r["0.3-2"] > 0 else np.nan
    res["cor_dura"] = r["4-10"] / r["2-4"] if r["2-4"] > 0 else np.nan
    lim = PILEUP_LIM.get(submodo)
    res["taxa_bruta_por_limite_pileup"] = extracao.get("razao_pileup", (t.size / expo / lim) if lim else None)
    if extracao["metodo"] == "anel":
        res["fluxo_desdobrado_cgs"] = None   # o anel não representa o fluxo da fonte
    # variabilidade: rms fracionário da curva de 100 s (fonte bruta, 0.3-10 keV)
    bins = np.arange(t0, t1 + 100, 100.0)
    c, _ = np.histogram(t, bins)
    c = c[c > 0]
    if c.size > 10:
        var = np.var(c) - np.mean(c)
        res["rms_frac_100s"] = float(np.sqrt(max(var, 0)) / np.mean(c))

    if str(hd.get("FILTER", "")).upper() == "CALCLOSED":
        res["estado"] = "filtro fechado (CalClosed): exposição sem fonte"
        return res
    if extracao.get("nao_detectada"):
        res["estado"] = "fonte não detectada na faixa de timing"
        return res
    if not fonte.pulsa:
        return res

    # ---- tempo ----
    orb = fonte.orbita
    td = demodular(t, mjdref, orb)
    tref = 0.5 * (td.min() + td.max())
    td = td - tref
    res["t_ref_mjd"] = float(mjdref + tref / DIA)
    dur = float(td.max() - td.min())
    k_orb = 2 * np.pi * orb.asini_lts / (orb.porb_d * DIA * np.sqrt(1 - orb.e ** 2)) * (1 + orb.e)
    nu0 = 1.0 / fonte.p_spin_s
    man = glob.glob(os.path.join(pasta_obs, "pulsaris", "*_manifest.json"))
    if man:
        p_x = json.load(open(man[0])).get("phase", {}).get("period_s")
        if p_x:
            nu0 = 1.0 / p_x
            res["periodo_reducao_s"] = float(p_x)
    esp = NU_ESPERADO.get(obsid)
    if esp and esp.get("fixo"):
        # período publicado a partir destes mesmos dados (poucos ciclos: a busca não resolve)
        nu, toa = esp["nu"], None
        grade = np.array([nu]); zg = np.array([np.nan])
        z2 = float(_z2_binado(td, nu))
    else:
        if esp:
            # GBM: frequência no referencial do pulsar, com a mesma órbita: só falta a resolução
            nu0, meia = esp["nu"], 2.0 / dur + 3 * esp.get("sig", 0.0)
        else:
            # período baricêntrico da redução (efsearch e Z^2_2, sem órbita): a busca cobre o Doppler residual
            meia = 1.5 * k_orb * nu0 + 2.0 / dur
        nu, z2, grade, zg = _busca_frequencia(td, nu0, meia, dur, rng)
        toa = _tempos_de_chegada(td, nu, nseg=max(4, min(16, int(dur / (40 / nu)))))
        if toa and abs(toa["dnu"]) < 1 / dur:
            nu += toa["dnu"]
    res["nu_esperado"] = esp
    res["tempo"] = {"nu_hz": nu, "periodo_s": 1 / nu, "z2_4_pico": z2, "toa": toa,
                    "busca_nu_min": float(grade[0]), "busca_nu_max": float(grade[-1]),
                    "n_tentativas": int(grade.size), "orbita_ref": orb.ref}
    if toa:
        res["tempo"]["sig_nu_hz"] = toa["sig_dnu"]
    fase = np.mod(td * nu, 1.0)
    H, m_opt, p_h = _htest(fase if fase.size < 3_000_000 else rng.choice(fase, 3_000_000, False))
    # maximizar Z^2_4 multiplica as frequências independentes por ~4 (atlas_tentativas_mc.py)
    n_indep = 1.0 if grade.size == 1 else max(1.0, 4 * (grade[-1] - grade[0]) * dur)
    p_t = float(-np.expm1(n_indep * np.log1p(-min(p_h, 1 - 1e-16))))
    res["tempo"].update({"H": H, "H_m": m_opt, "H_p": p_h, "H_p_com_tentativas": p_t,
                         "detectado": bool(p_t < 1e-3)})
    # sem órbita, para comparar o ganho da demodulação
    if "periodo_reducao_s" in res:
        tb = t - 0.5 * (t0 + t1)
        tb = tb if tb.size < 2_000_000 else rng.choice(tb, 2_000_000, replace=False)
        res["tempo"]["z2_4_sem_orbita_nu_reducao"] = float(_z2_binado(tb, 1 / res["periodo_reducao_s"]))

    # ---- perfis e harmônicos ----
    gti_d = None
    if gti is not None and len(gti):
        gti_d = np.column_stack([demodular(gti[:, 0], mjdref, orb), demodular(gti[:, 1], mjdref, orb)]) - tref
    f_exp = _exposicao_por_fase(gti_d, nu, 64)
    peso = (1.0 / 64) / np.maximum(f_exp[(fase * 64).astype(int)], 1e-9)
    f_exp32 = f_exp.reshape(NBINS, -1).sum(1)
    harm, pf, perfis = {}, {}, {}
    k_rec = int(min(max(m_opt, 1), K_PF))
    def banda(lo, hi):
        s = (e >= lo) & (e < hi)
        b = float(b_c[(e_c >= lo) & (e_c < hi)].sum())
        n = float(s.sum())
        return s, n, b
    for nome, (lo, hi) in BANDAS.items():
        s, n, b = banda(lo, hi)
        if n - b < 50:
            continue
        h = _harmonicos(fase[s], peso[s], n - b, n)
        harm[nome] = h
        pf[nome] = _pf(h, k_rec)
        cont = np.bincount((fase[s] * NBINS).astype(int), minlength=NBINS).astype(float)
        perfis[nome] = {"contagens": cont, "fracao_exposicao": f_exp32, "fundo_total": b}
    pfe = []
    for lo, hi in zip(BORDAS_PFE[:-1], BORDAS_PFE[1:]):
        s, n, b = banda(lo, hi)
        if n - b < 100:
            continue
        h = _harmonicos(fase[s], peso[s], n - b, n, kmax=K_PF)
        rms, srms, mm = _pf(h, k_rec)
        pfe.append({"e_lo": lo, "e_hi": hi, "n": n, "fundo": b, "pf_rms": rms, "sig_pf_rms": srms,
                    "pf_minmax": mm, "r1": h[0]["r_corr"], "r2": h[1]["r_corr"],
                    "psi1": h[0]["psi"], "sig_psi1": h[0]["sig_psi"],
                    "psi2": h[1]["psi"], "sig_psi2": h[1]["sig_psi"]})
    # atrasos: fase do harmônico k em cada banda menos a de 2-10 keV, em (-1/2k, 1/2k]
    ref = harm.get("2-10")
    atrasos = {}
    if ref:
        for nome, h in harm.items():
            atrasos[nome] = {}
            for k in (1, 2):
                d = h[k - 1]["psi"] - ref[k - 1]["psi"]
                d = (d + 0.5 / k) % (1.0 / k) - 0.5 / k
                atrasos[nome][f"k{k}"] = (float(d), float(np.hypot(h[k - 1]["sig_psi"], ref[k - 1]["sig_psi"])))
    for x in pfe:
        if ref:
            for k, chave in ((1, "psi1"), (2, "psi2")):
                d = x[chave] - ref[k - 1]["psi"]
                x[f"atraso{k}"] = float((d + 0.5 / k) % (1.0 / k) - 0.5 / k)
    if not res["tempo"]["detectado"]:
        # limite superior a 3 sigma do PF_rms (2 harmônicos): Z^2_2 ~ chi^2 não central com 4 graus de
        # liberdade e lambda = N PF_tot^2; lambda_UL tal que P(Z^2_2 <= z_obs | lambda_UL) = 0,00135.
        s_, n_, b_ = banda(0.3, 10.0)
        z = _z2_binado(td[s_], nu, kmax=2)
        lam = brentq(lambda l: ncx2.cdf(z, 4, l) - 0.00135, 1e-9, 1e7) if ncx2.cdf(z, 4, 1e-9) > 0.00135 else 0.0
        res["pf_rms_lim_sup_0.3-10"] = float(np.sqrt(lam / n_) * n_ / max(n_ - b_, 1.0))
        res["pf_rms_lim_sup_metodo"] = "ncx2_3sigma"      # atlas_correcoes.py usa este limite como está
    res["harmonicos"] = harm
    res["pf"] = {k: {"pf_rms": v[0], "sig_pf_rms": v[1], "pf_minmax": v[2]} for k, v in pf.items()}
    res["pf_energia"] = pfe
    res["atrasos_vs_2_10"] = atrasos
    os.makedirs(saida, exist_ok=True)
    np.savez_compressed(os.path.join(saida, f"{obsid}_perfis.npz"),
                        **{f"{b}__{k}": np.asarray(v) for b, d in perfis.items() for k, v in d.items()},
                        busca_nu=grade, busca_z2=zg)
    return res


def _json_seguro(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


if __name__ == "__main__":
    r = analisar(sys.argv[1], sys.argv[2])
    with open(os.path.join(sys.argv[2], f"{r['obsid']}.json"), "w") as f:
        json.dump(r, f, indent=1, default=_json_seguro)
    print(r["obsid"], r["estado"])
