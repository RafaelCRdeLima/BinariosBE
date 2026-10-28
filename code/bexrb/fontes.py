"""As seis fontes do estudo: pasta dos produtos reduzidos, período nominal e efemérides orbitais.

As órbitas das quatro transientes vêm do monitor de pulsares do Fermi/GBM (P. Jenke; página
gammaray.nsstc.nasa.gov/gbm/science/pulsars/lightcurves/<fonte>.html, lida em 27/10/2026),
exceto V 0332+53, em que a do GBM é dita preliminar e usamos Doroshenko et al. (2016, A&A 589, A72
... tabela 1, "This work"). X Per: Delgado-Martí et al. (2001, ApJ 546, 455), ajuste excêntrico
(tabela 2). PSR B1259-63 não pulsa em raios X; a órbita serve só para a fase orbital.

Convenções: tempos em MJD(TDB); `t_ref` é o periastro. Quando a fonte publica T_{pi/2} (longitude
média de 90°), o periastro é T_{pi/2} - (90° - omega)/360° * P_orb.
"""
from dataclasses import dataclass


def _periastro_de_tpi2(t_pi2, omega_deg, porb):
    return t_pi2 - (90.0 - omega_deg) / 360.0 * porb


@dataclass(frozen=True)
class Orbita:
    porb_d: float
    asini_lts: float
    e: float
    omega_deg: float
    t_peri_mjd: float
    ref: str


@dataclass(frozen=True)
class Fonte:
    nome: str
    pasta: str          # subpasta da raiz dos produtos reduzidos
    p_spin_s: float     # nominal, só para começar a busca
    pulsa: bool
    orbita: Orbita


FONTES = {
    "4U 0115+63": Fonte(
        "4U 0115+63", "4U0115_63", 3.614, True,
        Orbita(24.3168945, 141.223, 0.34107, 49.431,
               _periastro_de_tpi2(2456626.30505 - 2400000.5, 49.431, 24.3168945),
               "Fermi/GBM (Jenke, em prep.), T_pi/2 = JED 2456626.30505")),
    "V 0332+53": Fonte(
        "V 0332+53", "V0332_53", 4.3758, True,
        Orbita(33.850, 77.81, 0.3713, 277.43, 57157.88,
               "Doroshenko et al. 2016, tabela 1")),
    "1A 0535+262": Fonte(
        "1A 0535+262", "1A0535_262", 103.4, True,
        Orbita(110.9625, 323.6, 0.47, 81.6,
               _periastro_de_tpi2(2458167.5 - 2400000.5, 81.6, 110.9625),
               "Fermi/GBM (Jenke, BATSE+GBM, 2026), T_pi/2 = JED 2458167.5")),
    "Swift J0243.6+6124": Fonte(
        "Swift J0243.6+6124", "SwiftJ0243.6_6124", 9.86, True,
        Orbita(27.694313, 115.531, 0.1029, -74.05,
               _periastro_de_tpi2(2458116.097 - 2400000.5, -74.05, 27.694313),
               "Fermi/GBM (Jenke, preliminar), T_pi/2 = JED 2458116.097")),
    "X Per": Fonte(
        "X Per", "XPer", 837.7, True,
        Orbita(250.3, 454.0, 0.111, 288.0, 51353.0,
               "Delgado-Martí et al. 2001, tabela 2 (órbita excêntrica)")),
    "PSR B1259-63": Fonte(
        "PSR B1259-63", "PSRB1259-63", 0.0, False,
        Orbita(1236.724526, 1296.27448, 0.86987970, 138.665013, 53071.2447290,
               "Shannon et al. 2014, via Miller-Jones et al. 2018 tabela 3")),
}

POR_PASTA = {f.pasta: f for f in FONTES.values()}
