"""Demodulação orbital: leva tempos baricêntricos ao referencial da estrela de nêutrons.

Atraso de Roemer de uma órbita kepleriana (p.ex. Hilditch 2001; mesma forma usada pelo GBM):
    Delta(t) = x [ sin(omega) (cos E - e) + sqrt(1 - e^2) cos(omega) sin E ],
com x = a_x sin i / c e E a anomalia excêntrica. t_bary = t_emit + Delta(t_emit); invertemos por
iteração de ponto fixo, que converge em duas ou três passadas porque |dDelta/dt| ~ v/c << 1.
"""
import numpy as np

DIA = 86400.0


def anomalia_excentrica(m, e, iter_max=30):
    E = m + e * np.sin(m)
    for _ in range(iter_max):
        dE = (E - e * np.sin(E) - m) / (1.0 - e * np.cos(E))
        E -= dE
        if np.max(np.abs(dE)) < 1e-12:
            break
    return E


def atraso_roemer(mjd, orb):
    m = 2 * np.pi * (mjd - orb.t_peri_mjd) / orb.porb_d
    E = anomalia_excentrica(np.mod(m, 2 * np.pi), orb.e)
    w = np.radians(orb.omega_deg)
    return orb.asini_lts * (np.sin(w) * (np.cos(E) - orb.e)
                            + np.sqrt(1 - orb.e ** 2) * np.cos(w) * np.sin(E))


def demodular(t_s, mjdref, orb, iteracoes=3):
    """t_s: segundos TDB desde mjdref (tempo do XMM). Devolve os mesmos tempos sem o atraso orbital."""
    t = np.asarray(t_s, dtype=float)
    te = t.copy()
    for _ in range(iteracoes):
        te = t - atraso_roemer(mjdref + te / DIA, orb)
    return te


def fase_orbital(mjd, orb):
    """Fase da anomalia média em [0, 1) (0 = periastro) e anomalia verdadeira em graus."""
    m = np.mod(2 * np.pi * (mjd - orb.t_peri_mjd) / orb.porb_d, 2 * np.pi)
    E = anomalia_excentrica(m, orb.e)
    nu = 2 * np.arctan2(np.sqrt(1 + orb.e) * np.sin(E / 2), np.sqrt(1 - orb.e) * np.cos(E / 2))
    return m / (2 * np.pi), np.degrees(np.mod(nu, 2 * np.pi))
