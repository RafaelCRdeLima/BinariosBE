"""Luminosidade de reversão do torque, L_eq, e K_bol calibrado contra luminosidades de banda larga publicadas.

L_eq: com n(w) = 1 - w, o torque de disco dá exatamente nu-ponto = a Mdot^(6/7) - b Mdot^(3/7), ou seja
    nu-ponto = A L^(3/7) (L^(3/7) - L_eq^(3/7)),   A >= 0,
e L_eq é onde o torque troca de sinal (w = w_c). Ajustamos A e L_eq aos pontos com L <= 0,3 L_max e S/N >= 3
no MAXI, com dispersão intrínseca absoluta s0: -2 ln L perfilado em L_eq (A por mínimos quadrados
generalizados, restrito a A >= 0: em alta L o torque material gira a estrela para cima). A forma vale perto
de w_c para qualquer n(w) que cruze zero linearmente; a variante "forma GL79" usa
nu-ponto = A L^(6/7) n_GL79(0,35 (L/L_eq)^(-3/7)).

Hipótese nula ("sem reversão"): o torque não troca de sinal na amostra, L_eq <= L_min com A >= 0 (inclui
L_eq = 0, o spin-up em lei de potência). Delta = -2 ln L da nula menos o do melhor ajuste, os dois com a
mesma otimização (grade em log L_eq e s0, depois refinamento contínuo).

K_bol: razão entre o fluxo de banda larga publicado (reescalado ao Gaia não muda a razão de fluxos) e o fluxo
2-20 keV do MAXI no mesmo dia (taxa / taxa da Nebulosa do Caranguejo x 3,0e-8), para as entradas de
dados/literatura/linhas_ciclotronicas.csv com data e L, mais as de luminosidades.csv da Swift J0243.

Robustez de L_eq: covariância completa dos nu-ponto (pares vizinhos dividem uma frequência; adotada),
diagonal, só pares independentes, outros cortes em L, a forma GL79, cada outburst separado e, na Swift J0243,
o outburst gigante separado dos outbursts menores de 2018-2019 (Liu, Ji & Ge 2023 acham luminosidades de torque
zero diferentes nos dois) (reversao_variantes.csv). Na variante adotada, também com os termos fracionário e de
erro de L da verossimilhança do ajuste global, otimizados em conjunto com A.

Calibração (bootstrap paramétrico sob a nula ajustada, --boot=<n>, padrão 2000): p de cada variante da 4U 0115+63
sob a sua própria nula e p da seleção a posteriori (maior Delta entre as variantes, sob a nula do conjunto
L <= 0,5 L_max).

Uso: python reversao.py <resultados> [--boot=<n>]
     (escreve <resultados>/torque/{reversao.csv, reversao_variantes.csv, reversao_bootstrap.json, kbol.csv})
"""
import csv
import json
import os
import re
import sys
from multiprocessing import Pool

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from scipy.stats import chi2

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from torque import CRAB_MAXI, ler_maxi  # noqa: E402
from torque_ajuste import covariancia, dados, independentes, n_gl  # noqa: E402

RES = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else os.path.join(AQUI, "resultados")
SAIDA = os.path.join(RES, "torque")
KPC = 3.0857e21
MAXI = {"4U 0115+63": "J0118+637", "V 0332+53": "J0334+531", "1A 0535+262": "J0538+263",
        "Swift J0243.6+6124": "J0243+614"}
# escalas internas: L em 1e37 erg/s, nu-ponto em 1e-12 Hz/s
UL, UN = 1e37, 1e-12
GRADE = np.arange(30.0, 39.5 + 1e-9, 0.005)        # log10 L_eq; L_eq = 0 é tratado à parte
S0_REL = np.concatenate([[0.0], np.logspace(-2.5, 0.5, 31)])
# divisões de um outburst em variantes (MJD): a Swift J0243 tem o outburst gigante de 2017 e outbursts menores
# em 2018-2019 no mesmo grupo (Liu, Ji & Ge 2023, períodos MJD 58150-58280 e 58280-58520)
CORTES_OUTBURST = {"Swift J0243.6+6124": (58280.0,)}


def forma_f(Ls, leq, forma):
    """F(L; L_eq) com nu-ponto = A F: Ls (n,) e leq (G,) em 1e37 erg/s; leq = 0 é o spin-up sem reversão."""
    if forma == "lin":
        x = Ls ** (3 / 7)
        return x[:, None] * (x[:, None] - leq[None, :] ** (3 / 7))
    with np.errstate(divide="ignore"):
        w = np.where(leq[None, :] > 0, 0.35 * (Ls[:, None] / np.where(leq > 0, leq, 1.0)[None, :]) ** (-3 / 7), 0.0)
    return Ls[:, None] ** (6 / 7) * n_gl(w)


class Reversao:
    """-2 ln L = r' Sigma^-1 r + ln det Sigma, Sigma = C + s0^2 1, r = nu-ponto - A F(L; L_eq), A >= 0.
    Com C = U diag(lam) U', Sigma^-1 = U diag(1/(lam + s0^2)) U': cada (L_eq, s0) custa O(n)."""

    def __init__(self, d, cov, forma="lin"):
        self.Ls = d["L"] / UL
        self.lam, self.U = np.linalg.eigh(cov / UN ** 2)
        self.y = self.U.T @ (d["nd"] / UN)
        self.forma = forma
        self.s0g = S0_REL * np.median(np.abs(d["nd"] / UN))
        self.lmin = float(np.log10(d["L"].min()))

    def _q(self, f, s0):
        v = self.lam + s0 ** 2
        a = np.maximum((f * self.y[:, None] / v[:, None]).sum(0) / (f * f / v[:, None]).sum(0), 0.0)
        return ((self.y[:, None] - a * f) ** 2 / v[:, None]).sum(0) + np.log(v).sum(), a

    def perfil(self, lgrade):
        """Perfil em s0 (grade) para cada log L_eq de lgrade (-inf = L_eq 0)."""
        leq = np.where(np.isfinite(lgrade), 10.0 ** (np.where(np.isfinite(lgrade), lgrade, 0.0) - 37), 0.0)
        f = self.U.T @ forma_f(self.Ls, leq, self.forma)
        c, s0b, ab = np.full(lgrade.size, np.inf), np.zeros(lgrade.size), np.zeros(lgrade.size)
        for s0 in self.s0g:
            q, a = self._q(f, s0)
            m = q < c
            c[m], s0b[m], ab[m] = q[m], s0, a[m]
        return c, s0b, ab

    def q(self, lg, s0):
        leq = np.array([0.0 if not np.isfinite(lg) else 10.0 ** (lg - 37)])
        q, a = self._q(self.U.T @ forma_f(self.Ls, leq, self.forma), s0)
        return float(q[0]), float(a[0])

    def refinar(self, lg0, s00, lg_max=None):
        """Mínimo contínuo em (log L_eq, log s0) a partir de um ponto da grade; lg_max limita L_eq (nula).
        Também testa L_eq = 0 quando lg_max é dado. Devolve (q, log L_eq, s0, A)."""
        ls_min = np.log10(self.s0g[1]) - 3
        ls0 = np.log10(s00) if s00 > 0 else ls_min
        cand = [(self.q(lg0, s00)[0], lg0, s00)]
        if np.isfinite(lg0):
            lo, hi = (GRADE[0], lg_max) if lg_max is not None else (self.lmin, GRADE[-1])
            r = minimize(lambda p: self.q(min(max(p[0], lo), hi), 10 ** p[1])[0], [min(max(lg0, lo), hi), ls0],
                         method="Nelder-Mead", bounds=[(lo, hi), (ls_min, 2.0)], options={"xatol": 1e-4, "fatol": 1e-5})
            cand.append((r.fun, min(max(r.x[0], lo), hi), 10 ** r.x[1]))
        if lg_max is not None or not np.isfinite(lg0):
            r = minimize(lambda p: self.q(-np.inf, 10 ** p[0])[0], [ls0], method="Nelder-Mead",
                         bounds=[(ls_min, 2.0)], options={"xatol": 1e-4, "fatol": 1e-5})
            cand.append((r.fun, -np.inf, 10 ** r.x[0]))
        q, lg, s0 = min(cand, key=lambda x: x[0])
        return q, lg, s0, self.q(lg, s0)[1]


def ajustar_leq(d, cov=None, forma="lin", grade=GRADE, refinar=True):
    """Perfil de -2 ln L em log L_eq (com L_eq = 0), melhor ajuste, intervalo Delta <= 1 e Delta contra a nula
    sem troca de sinal na amostra (L_eq <= L_min). cov: covariância de medida (None: diagonal sig^2)."""
    C = np.diag(d["sig"] ** 2) if cov is None else cov
    m = Reversao(d, C, forma)
    lg = np.concatenate([[-np.inf], grade])
    c, s0b, ab = m.perfil(lg)
    nulo = lg <= m.lmin
    i = int(np.argmin(c))
    i0 = int(np.argmin(np.where(nulo, c, np.inf)))
    if refinar:
        q0, lg0, s00, A0 = m.refinar(lg[i0], s0b[i0], lg_max=m.lmin)
        qb, lgb, s0, A = (q0, lg0, s00, A0) if nulo[i] else m.refinar(lg[i], s0b[i])
        if q0 < qb:
            qb, lgb, s0, A = q0, lg0, s00, A0
    else:
        qb, lgb, s0, A = c[i], lg[i], s0b[i], ab[i]
        q0, lg0, s00, A0 = c[i0], lg[i0], s0b[i0], ab[i0]
    ok = c <= c[i] + 1.0
    ok_rev = ok & ~nulo
    aberto_baixo = bool(ok[nulo].any())
    aberto_alto = bool(ok[-1])
    q_zero = c[0] if not refinar else m.refinar(-np.inf, s0b[0], lg_max=-np.inf)[0]
    return {"log_Leq": float(lgb), "lo": float(lg[ok_rev].min()) if ok_rev.any() else np.nan,
            "hi": float(lg[ok_rev].max()) if ok_rev.any() else np.nan,
            "aberto_baixo": aberto_baixo, "aberto_alto": aberto_alto,
            "delta_sem_reversao": float(max(q0 - qb, 0.0)), "delta_Leq_zero": float(max(q_zero - qb, 0.0)),
            "na_borda": bool(aberto_baixo or aberto_alto or nulo[i]), "A": A * UN / UL ** (6 / 7), "s0": s0 * UN,
            "nulo": {"log_Leq": float(lg0), "s0": s00 * UN, "A_int": float(A0)},
            "log_Lmin": m.lmin}


def ajustar_leq_completo(d, cov, centro, meia=0.3, passo=0.01, partidas=((0.0,), (0.1,), (0.3,))):
    """Como ajustar_leq (forma linear) com a covariância completa e os termos da verossimilhança do ajuste global
    (torque_ajuste.py): dispersão fracionária (s m)^2 e erro de L, (dm/dlnL sigma_L/L)^2. A variância depende de
    A, então A, s e s0 são otimizados em conjunto (L-BFGS-B, A >= 0) em cada log L_eq, com Sigma refeita em cada
    avaliação e várias partidas; também a nula L_eq <= L_min."""
    Ls, y, C = d["L"] / UL, d["nd"] / UN, cov / UN ** 2
    sL = d["sL"]
    x = Ls ** (3 / 7)
    med = float(np.median(np.abs(y)))

    def q_par(p, F, dF):
        A, s, s0 = p
        V = C + np.diag(s0 ** 2 + (s * A * F) ** 2 + (A * dF * sL) ** 2)
        try:
            Lc = np.linalg.cholesky(V)
        except np.linalg.LinAlgError:
            return 1e30
        r = solve_triangular(Lc, y - A * F, lower=True)
        return float(r @ r + 2 * np.sum(np.log(np.diag(Lc))))

    def q_leq(lg, ini):
        xeq = 0.0 if not np.isfinite(lg) else (10 ** (lg - 37)) ** (3 / 7)
        F = x * (x - xeq)
        dF = 6 / 7 * x ** 2 - 3 / 7 * x * xeq
        Ci = np.linalg.inv(C + np.eye(y.size) * (0.3 * med) ** 2)
        a0 = max(float(F @ Ci @ y / (F @ Ci @ F)), 1e-3)
        starts = [(a0, s, 0.3 * med) for (s,) in partidas] + ([ini] if ini is not None else [])
        best = (np.inf, None)
        for p0 in starts:
            r = minimize(q_par, p0, args=(F, dF), method="L-BFGS-B",
                         bounds=[(0.0, None), (0.0, 3.0), (0.0, 10 * med)])
            if r.fun < best[0]:
                best = (r.fun, r.x)
        return best

    def varrer(grade):
        c, ini, pars = [], None, []
        for g in grade:
            q, p = q_leq(g, ini)
            c.append(q)
            pars.append(p)
            ini = tuple(p)
        return np.array(c), pars

    grade = np.arange(centro - meia, centro + meia + 1e-9, passo)
    c, pars = varrer(grade)
    lmin = float(np.log10(d["L"].min()))
    gnulo = np.concatenate([[-np.inf], np.arange(lmin - 3.0, lmin + 1e-9, 0.05)])
    c0, _ = varrer(gnulo)
    i = int(np.argmin(c))
    ok = grade[c <= c[i] + 1]
    # convergência: dispersão do mínimo entre partidas independentes no melhor L_eq
    xeq = (10 ** (grade[i] - 37)) ** (3 / 7)
    F, dF = x * (x - xeq), 6 / 7 * x ** 2 - 3 / 7 * x * xeq
    rng = np.random.default_rng(3)
    qs = [minimize(q_par, (pars[i][0] * rng.uniform(0.5, 2), rng.uniform(0, 0.5), med * rng.uniform(0.05, 1)),
                   args=(F, dF), method="L-BFGS-B", bounds=[(0.0, None), (0.0, 3.0), (0.0, 10 * med)]).fun
          for _ in range(8)]
    return {"log_Leq": float(grade[i]), "lo": float(ok.min()), "hi": float(ok.max()),
            "na_borda": bool(i in (0, grade.size - 1) or ok.min() <= grade[0] or ok.max() >= grade[-1]),
            "delta_sem_reversao": float(max(c0.min() - c[i], 0.0)), "s": float(pars[i][1]),
            "s0": float(pars[i][2] * UN), "conv_partidas_min_menos_perfil": float(min(qs) - c[i]),
            "conv_partidas_max_menos_min": float(max(qs) - min(qs))}


def grupos(mjd, lacuna=100.0):
    """Rótulo de outburst: começa um novo a cada lacuna de mais de 100 d entre pontos."""
    o = np.argsort(mjd)
    g = np.zeros(mjd.size, int)
    g[o] = np.cumsum(np.r_[0, np.diff(mjd[o]) > lacuna])
    return g


def casos(f, tudo):
    """Variantes da reversão: (nome, índices em tudo, com covariância, forma). As frações de L são relativas ao
    L_max de todos os pontos com S/N >= 3."""
    Lmax = tudo["L"].max()
    idx = np.arange(tudo["L"].size)

    def corte(s, frac):
        return s[tudo["L"][s] <= frac * Lmax]
    ind = independentes(dict(tudo, idx=idx))["idx"]
    out = [("cov", corte(idx, 0.3), True, "lin"), ("diag", corte(idx, 0.3), False, "lin"),
           ("indep", corte(ind, 0.3), False, "lin"), ("cov, L<=0.15Lmax", corte(idx, 0.15), True, "lin"),
           ("cov, L<=0.5Lmax", corte(idx, 0.5), True, "lin"), ("cov, forma GL79", corte(idx, 0.3), True, "GL79")]
    g = grupos(tudo["mjd"])
    for k in np.unique(g):
        s = corte(idx[g == k], 0.3)
        pedacos = [s]
        for m in CORTES_OUTBURST.get(f, ()):
            if tudo["mjd"][s].min() < m < tudo["mjd"][s].max():
                pedacos += [s[tudo["mjd"][s] < m], s[tudo["mjd"][s] >= m]]
        for p in pedacos:
            if p.size >= 8:
                out.append((f"outburst MJD {tudo['mjd'][p].min():.0f}-{tudo['mjd'][p].max():.0f}", p, True, "lin"))
    return out


def subconj(d, s):
    return {k: v[s] for k, v in d.items()}


def deltas_variantes(tudo, C_tudo, cs, refinar=False):
    """Delta de cada variante para um conjunto (real ou simulado) com covariância completa C_tudo."""
    out = []
    for nome, s, com_cov, forma in cs:
        dd = subconj(tudo, s)
        r = ajustar_leq(dd, cov=C_tudo[np.ix_(s, s)] if com_cov else None, forma=forma, refinar=refinar)
        out.append(r["delta_sem_reversao"])
    return np.array(out)


def _sim(args):
    """Um conjunto simulado sob a nula: nu-ponto = A0 F(L; L_eq0) + N(0, C + s0^2 1) (A0 em unidades internas,
    L_eq0 em erg/s, 0 = sem reversão)."""
    tudo, C, cs, nulo, forma0, semente, alvo = args
    rng = np.random.default_rng(semente)
    leq0 = 0.0 if not np.isfinite(nulo["log_Leq"]) else 10 ** (nulo["log_Leq"] - 37)
    m = nulo["A_int"] * forma_f(tudo["L"] / UL, np.array([leq0]), forma0)[:, 0] * UN
    Lc = np.linalg.cholesky(C + nulo["s0"] ** 2 * np.eye(C.shape[0]))
    sim = dict(tudo, nd=m + Lc @ rng.standard_normal(C.shape[0]))
    if alvo is None:
        return deltas_variantes(sim, C, cs)
    nome, s, com_cov, forma = alvo
    dd = subconj(sim, s)
    return ajustar_leq(dd, cov=C[np.ix_(s, s)] if com_cov else None, forma=forma, refinar=False)["delta_sem_reversao"]


def calibrar(tudo, C, cs, n_boot, semente=2810):
    """p de cada variante sob a sua própria nula e p da seleção (maior Delta entre as variantes) sob a nula do
    conjunto L <= 0,5 L_max. Os Delta observados e simulados usam a mesma otimização (grade, sem refinamento)."""
    nproc = int(os.environ.get("NPROC", os.cpu_count()))
    ss = np.random.SeedSequence(semente)
    obs = deltas_variantes(tudo, C, cs)
    out = {"n_boot": n_boot, "variantes": {}}
    with Pool(nproc) as pool:
        for (nome, s, com_cov, forma), d_obs, sem in zip(cs, obs, ss.spawn(len(cs))):
            dd = subconj(tudo, s)
            r = ajustar_leq(dd, cov=C[np.ix_(s, s)] if com_cov else None, forma=forma)["nulo"]
            # a nula é ajustada só nos pontos da variante, mas simulada em todos (só os da variante entram)
            sims = np.array(pool.map(_sim, [(tudo, C, cs, r, forma, x, (nome, s, com_cov, forma))
                                            for x in sem.spawn(n_boot)], chunksize=20))
            out["variantes"][nome] = {"delta": float(d_obs), "p": float((1 + np.sum(sims >= d_obs - 1e-9)) / (n_boot + 1)),
                                      "delta_95": float(np.quantile(sims, 0.95)),
                                      "p_chi2_1": float(chi2.sf(d_obs, 1)) if d_obs > 0 else 1.0}
            print(f"  bootstrap {nome:32s} Delta={d_obs:5.2f} p={out['variantes'][nome]['p']:.4f} "
                  f"(Delta_95 sob a nula = {out['variantes'][nome]['delta_95']:.2f})", flush=True)
        uniao = [x for x in cs if x[0] == "cov, L<=0.5Lmax"][0]
        r = ajustar_leq(subconj(tudo, uniao[1]), cov=C[np.ix_(uniao[1], uniao[1])])["nulo"]
        out["nulo_selecao"] = {k: float(v) for k, v in r.items()}
        sims = np.array(pool.map(_sim, [(tudo, C, cs, r, "lin", x, None)
                                        for x in ss.spawn(len(cs) + 1)[-1].spawn(n_boot)], chunksize=10))
    mx = sims.max(axis=1)
    out["selecao"] = {"delta_max_obs": float(np.max(obs)), "variante_max": cs[int(np.argmax(obs))][0],
                      "p_selecao": float((1 + np.sum(mx >= np.max(obs) - 1e-9)) / (n_boot + 1)),
                      "delta_max_95": float(np.quantile(mx, 0.95))}
    print(f"  seleção: maior Delta = {np.max(obs):.2f} ({out['selecao']['variante_max']}), p = "
          f"{out['selecao']['p_selecao']:.4f} (Delta_max_95 sob a nula = {out['selecao']['delta_max_95']:.2f})", flush=True)
    return out


def kbol():
    crabr = CRAB_MAXI

    def fmaxi(f, m0, m1=None):
        m = ler_maxi(MAXI[f])
        m1 = m0 if m1 is None else m1
        s = (m[:, 0] >= m0 - 1.0) & (m[:, 0] <= m1 + 1.0)
        return np.mean(m[s, 1]) / crabr * 3.0e-8 if s.any() else np.nan

    out = []
    vistos = set()
    for x in csv.DictReader(open(os.path.join(AQUI, "dados", "literatura", "linhas_ciclotronicas.csv"))):
        f = x["fonte"]
        if f not in MAXI or not x["L_na_medida_erg_s"].strip():
            continue
        mm = re.findall(r"MJD\s*~?(\d{5}(?:\.\d+)?)(?:\s*-\s*(\d{5}(?:\.\d+)?))?", x["epoca"])
        if not mm or "após" in x["epoca"]:
            continue
        a, b = mm[0]
        m0, m1 = float(a), float(b) if b else float(a)
        try:
            L = float(x["L_na_medida_erg_s"].replace("~", ""))
            du = float(x["d_usada_kpc"].split()[0])
        except ValueError:
            continue
        chave = (f, x["ref"], round(m0, 1))
        if chave in vistos:
            continue
        vistos.add(chave)
        fl = L / (4 * np.pi * (du * KPC) ** 2)
        fm = fmaxi(f, m0, m1)
        if fm > 0:
            out.append({"fonte": f, "ref": x["ref"], "banda": x["banda_L"], "mjd": 0.5 * (m0 + m1), "F_lit": fl,
                        "F_maxi_2_20": fm, "K": fl / fm})
    # Swift J0243: luminosidades de banda larga com data além de Kong+2022 (que está no csv): Jaisawal+2018,
    # citado por Wilson-Hodge+2018, e Wilson-Hodge+2018 (NuSTAR escalado), ambas em 3-70 keV
    for ref, banda, L, du, m0, m1 in (("Wilson-Hodge+2018", "3-70 keV", 2.8e38, 7.0, 58098, 58105),
                                      ("Jaisawal+2018", "3-70 keV", 5.1e37, 7.0, 58031, 58031)):
        fl = L / (4 * np.pi * (du * KPC) ** 2)
        fm = fmaxi("Swift J0243.6+6124", m0, m1)
        if fm > 0:
            out.append({"fonte": "Swift J0243.6+6124", "ref": ref, "banda": banda, "mjd": 0.5 * (m0 + m1),
                        "F_lit": fl, "F_maxi_2_20": fm, "K": fl / fm})
    return out



if __name__ == "__main__":
    N_BOOT = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--boot=")), 2000))
    os.makedirs(SAIDA, exist_ok=True)
    par = {x["fonte"]: x for x in csv.DictReader(open(os.path.join(AQUI, "dados", "parametros_fontes.csv")))}
    rev, var, boot = [], [], {}

    def leq_(lg):
        return 10 ** lg if np.isfinite(lg) else 0.0

    for f in ("4U 0115+63", "1A 0535+262", "Swift J0243.6+6124"):
        tudo = dados(f, float(par[f]["d_kpc"]), 0.0, 3.0)
        C = covariancia(tudo)
        cs = casos(f, tudo)
        for nome, s, com_cov, forma in cs:
            dd = subconj(tudo, s)
            cov = C[np.ix_(s, s)] if com_cov else None
            r = ajustar_leq(dd, cov=cov, forma=forma)
            linha = {"fonte": f, "variante": nome, "N": int(s.size), "L_eq": leq_(r["log_Leq"]),
                     "L_eq_lo": leq_(r["lo"]), "L_eq_hi": leq_(r["hi"]),
                     "delta_m2lnL_sem_reversao": r["delta_sem_reversao"], "na_borda": r["na_borda"],
                     "aberto_baixo": r["aberto_baixo"], "aberto_alto": r["aberto_alto"],
                     "delta_m2lnL_Leq_zero": r["delta_Leq_zero"], "L_min": 10 ** r["log_Lmin"],
                     "A": r["A"], "s0": r["s0"], "p_bootstrap": np.nan}
            var.append(linha)
            print(f"{f:20s} {nome:28s} N={linha['N']:3d} L_eq={linha['L_eq']:.2e} [{linha['L_eq_lo']:.2e}, "
                  f"{linha['L_eq_hi']:.2e}] Delta(sem troca)={r['delta_sem_reversao']:.2f} "
                  f"Delta(L_eq=0)={r['delta_Leq_zero']:.2f} aberto={r['aberto_baixo']}/{r['aberto_alto']} "
                  f"borda={r['na_borda']}", flush=True)
            if nome == "cov":
                rev.append({k: v for k, v in linha.items() if k != "variante"})
                if not r["na_borda"]:
                    # verificação: com os termos fracionário e de erro de L da verossimilhança completa
                    rc = ajustar_leq_completo(dd, cov, r["log_Leq"])
                    var.append({"fonte": f, "variante": "cov + s e erro de L", "N": int(s.size),
                                "L_eq": 10 ** rc["log_Leq"], "L_eq_lo": 10 ** rc["lo"], "L_eq_hi": 10 ** rc["hi"],
                                "delta_m2lnL_sem_reversao": rc["delta_sem_reversao"], "na_borda": rc["na_borda"],
                                "aberto_baixo": False, "aberto_alto": False, "delta_m2lnL_Leq_zero": np.nan,
                                "L_min": 10 ** r["log_Lmin"], "A": np.nan, "s0": rc["s0"], "p_bootstrap": np.nan})
                    print(f"{f:20s} {'cov + s e erro de L':28s} L_eq={10 ** rc['log_Leq']:.2e} "
                          f"(desvio {rc['log_Leq'] - r['log_Leq']:+.3f} dex) [{10 ** rc['lo']:.2e}, {10 ** rc['hi']:.2e}] "
                          f"Delta={rc['delta_sem_reversao']:.2f} s={rc['s']:.3f} convergência: "
                          f"{rc['conv_partidas_min_menos_perfil']:+.3f} / {rc['conv_partidas_max_menos_min']:.3f}", flush=True)
        if f == "4U 0115+63" and N_BOOT > 0:
            # 4U 0115+63: p calibrado de cada variante e da escolha da variante a posteriori
            boot[f] = calibrar(tudo, C, cs, N_BOOT)
            for linha in var:
                if linha["fonte"] == f and linha["variante"] in boot[f]["variantes"]:
                    linha["p_bootstrap"] = boot[f]["variantes"][linha["variante"]]["p"]
    with open(os.path.join(SAIDA, "reversao.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rev[0].keys()))
        w.writeheader()
        w.writerows(rev)
    with open(os.path.join(SAIDA, "reversao_variantes.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, list(var[0].keys()))
        w.writeheader()
        w.writerows(var)
    if boot:
        json.dump(boot, open(os.path.join(SAIDA, "reversao_bootstrap.json"), "w"), indent=1)
    kb = kbol()
    with open(os.path.join(SAIDA, "kbol.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, list(kb[0].keys()))
        w.writeheader()
        w.writerows(kb)
    for x in kb:
        print(f"{x['fonte']:20s} {x['ref']:18s} {x['banda'][:28]:28s} MJD {x['mjd']:8.1f} K={x['K']:.2f}")


def ajuste_b_com_prior(res):
    """Ajuste B com a calibração de K_bol como prior: log lam ~ N(centro, meia largura) da faixa de K_bol
    (alta e baixa L juntas) / 2, somado ao perfil -2 ln L(lam, mu) salvo por torque_ajuste.py."""
    from indice import K_ALTO, K_BAIXO
    from torque_ajuste import GBM, b_polo, ramos
    out = []
    for f, (curto, _) in GBM.items():
        k = list(K_ALTO[f]) + list(K_BAIXO.get(f, ()))
        lo, hi = np.log10(min(k) / 2), np.log10(max(k) / 2)
        for n in ("GL79", "lin"):
            z = np.load(os.path.join(res, "torque", f"perfil_B_{curto}_{n}.npz"))
            c = z["c"] + ((z["llam"][:, None] - 0.5 * (lo + hi)) / (0.5 * (hi - lo))) ** 2
            melhor = {}
            for j in range(z["llam"].size):
                for ramo, r in ramos(c[j], z["lmu"], z["cortes"][j]).items():
                    if r and (ramo not in melhor or r["m2lnl"] < melhor[ramo][0]):
                        melhor[ramo] = (r["m2lnl"], r["lmu"], 10 ** z["llam"][j], r["na_borda"])
            b, h = melhor.get("baixo-mu"), melhor.get("alto-mu")
            linha = {"fonte": f, "torque": n, "lam_prior": f"{10 ** lo:.2f}-{10 ** hi:.2f}",
                     "B_baixo_G": b_polo(b[1]) if b else np.nan, "lam_baixo": b[2] if b else np.nan,
                     "borda_baixo": b[3] if b else "", "B_alto_G": b_polo(h[1]) if h else np.nan,
                     "lam_alto": h[2] if h else np.nan, "borda_alto": h[3] if h else "",
                     "delta_alto_menos_baixo": (h[0] - b[0]) if b and h else np.nan}
            out.append(linha)
            print(f"B+prior {f:20s} {n:4s} lam~{linha['lam_prior']}: baixo {linha['B_baixo_G']:.2e} (lam {linha['lam_baixo']:.2f},"
                  f" borda {linha['borda_baixo']}) alto {linha['B_alto_G']:.2e} (lam {linha['lam_alto']:.2f}, borda "
                  f"{linha['borda_alto']}) d(alto-baixo)={linha['delta_alto_menos_baixo']:.1f}")
    with open(os.path.join(res, "torque", "ajuste_B_prior_kbol.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, list(out[0].keys()))
        w.writeheader()
        w.writerows(out)


if __name__ == "__main__":
    ajuste_b_com_prior(RES)
