# Magnetic-field diagnostics in Be/X-ray pulsars

https://github.com/RafaelCRdeLima/BinariosBE

Code and derived data for the paper *Magnetic-field diagnostics in Be/X-ray pulsars: a consistency test
from torque reversals and propeller limits* (submitted to MNRAS). Everything in the paper can be regenerated
from this repository and the public data listed below: the spin-up points, the likelihood fits of the torque,
the torque-reversal luminosities and their calibration, the field-discrepancy index, the hierarchical model of
the published transition luminosities, the XMM-Newton timing atlas, and all figures and the machine-generated
tables (Tables 2-4, A1, A2, B1 and B2); Tables 1 and 5 are written by hand in the manuscript.

The code is in Python. Its comments and variable names are in Portuguese; the docstring at the top of each
script states what it does, what it reads and what it writes.

## Layout

| Path | Contents |
|---|---|
| `code/` | Analysis scripts (see the pipeline below). |
| `code/bexrb/` | XMM-Newton timing atlas (`atlas_obs.py`), orbital demodulation (`orbita.py`), orbital elements and source list (`fontes.py`). |
| `code/dados/literatura/` | Our compilation of published values: `luminosidades.csv` (luminosities of the accretion-column transitions; column `h1` says whether each value enters the clean sample and, if not, why) and `linhas_ciclotronicas.csv` (cyclotron-line measurements, used for the bolometric-correction calibration). Column `ref` gives the paper; full references are in the bibliography of the paper. |
| `code/dados/parametros_fontes.csv` | Adopted Gaia DR3 distances and cyclotron energies. |
| `code/dados/qualidade.json`, `mjd_obs.csv`, `nu_esperado.json` | Notes on individual XMM-Newton observations, their epochs, and the spin frequency expected from Fermi/GBM at each epoch. |
| `code/dados/reducao.json` | Reduction parameters of each XMM-Newton observation (see below). |
| `results/torque/` | Spin-up points (`pontos_*.csv`), global likelihood fits and profiles (`ajuste_global*.csv`, `perfil_*.npz`), torque-reversal fits and robustness variants (`reversao*.csv`), bootstrap calibration of the reversal test (`reversao_bootstrap.json`), bolometric-correction calibration (`kbol.csv`, `ajuste_B_prior_kbol.csv`). |
| `results/sintese/` | Transitions in units of the critical luminosity (`transicoes_h1.csv`), propeller fields (`campos_propeller.csv`), field-discrepancy index (`indice.json`), hierarchical model and its calibration (`h1_*.json`), luminosity of each XMM-Newton observation (`atlas_ell.csv`). |
| `results/atlas/` | XMM-Newton timing atlas: one JSON and the folded profiles per observation (`obs/`), catalogue, Fourier harmonics, pulsed fraction by energy, detections and upper limits (`correcoes.json`). |
| `run_all.sh` | Runs the whole pipeline. |

## Third-party data (not included)

The repository does not redistribute data from other projects. `python code/baixar_dados.py` downloads them
into `code/dados/` with the file names the code expects:

- **Fermi/GBM Accreting Pulsars Program** (Malacaria et al. 2020, ApJ 896, 90): pulse-frequency histories
  of 4U 0115+63, V 0332+53, 1A 0535+262 and Swift J0243.6+6124, from
  `https://gammaray.nsstc.nasa.gov/gbm/science/pulsars/lightcurves/<source>.fits.gz`. The frequencies are
  barycentred and corrected for the binary orbit by the GBM team.
- **MAXI/GSC** (Matsuoka et al. 2009, PASJ 61, 999): daily light curves in 2-20 keV and its sub-bands of the
  five accreting pulsars and of the Crab, from `http://maxi.riken.jp/star_data/<J...>/<J...>_g_lc_1day_all.dat`
  (provided by RIKEN, JAXA and the MAXI team).
- **XMM-Newton** EPIC-pn observations, public in the XMM-Newton Science Archive. The ObsIDs are listed in
  `results/atlas/catalogo_obs.csv` and in Table A2 of the paper. They were reduced with SAS 22.1.0 as described
  in Appendix A of the paper: `cifbuild`, `odfingest` and `epproc`; soft-proton flares screened with
  `tabgtigen` on the single-event light curve above 10 keV; events with FLAG = 0 and PATTERN <= 4
  (`evselect`); barycentric correction with `barycen`; responses from `rmfgen` and `arfgen`; and, for the
  observations without a Fermi/GBM frequency, a period search with `efsearch` (HEASoft 6.36).
  The reduced products are not included; the atlas products derived from them are in `results/atlas/`, and
  the next section describes what is needed to rebuild the atlas.
- **Literature values** in `code/dados/literatura/`, `code/dados/parametros_fontes.csv` and the orbital
  elements in `code/bexrb/fontes.py` are taken from the papers cited in each entry (Gaia DR3 distances as
  tabulated by Reig 2026; orbits from the Fermi/GBM pages and the papers cited in Table A1). The critical
  luminosity of Mushtukov et al. (2015) was digitized from their fig. 6; the points are in `code/sintese.py`.

The paper used the GBM and MAXI files as of 2026 October 27. Both grow with time, and MAXI revises its most
recent days, so a later download differs from that version. `MJD_MAX` in `code/torque.py` cuts all GBM and
MAXI data at MJD 61330.5 (the last GBM detection used is at MJD 61316.5), and `CRAB_MAXI` fixes the Crab rate
that normalizes all MAXI fluxes to the median used in the paper; with both, a later download gives the same
spin-up points unless the archives reprocess older data. `baixar_dados.py` reports whether each downloaded
file is identical to the one used in the paper.

## Rebuilding the XMM-Newton atlas

`code/rodar_atlas.py <root> <output>` reads, for each observation, the folder `<root>/<source>/<ObsID>/`,
with the source folder names of `code/bexrb/fontes.py` (`pasta`), and the following EPIC-pn products, where
`sNNN` is the exposure (e.g. `s003`):

| File | Contents | SAS task |
|---|---|---|
| `epn_sNNN_clean_bary.fits` | Calibrated events with FLAG = 0, PATTERN <= 4, PI 150-15000, within the flare GTI, barycentred; keeps the `STDGTInn` and `GTI0xx` extensions | `epproc`, `tabgtigen`, `evselect`, `barycen` |
| `epn_sNNN_bary_source.fits` | The same events within the source region, with the same GTI extensions (the atlas reads the GTIs from this file) | `evselect` |
| `epn_sNNN_src_spec.fits`, `epn_sNNN_bkg_spec.fits` | Source and background spectra, with `BACKSCAL` and `EXPOSURE` | `evselect`, `backscale` |
| `epn_sNNN_src.arf` | Ancillary response of the source region | `arfgen` |

`code/dados/reducao.json` gives, for each ObsID, the parameters used to make these products and that the atlas
needs: the exposure (`exposicao`); the flare screening (`gti`: the background light-curve selection
`(#XMMEA_EP) && (PI>10000&&PI<12000) && (PATTERN==0)` in 100 s bins, the `tabgtigen` threshold in counts/s,
and the `evselect` expressions of the clean and source events); the source and background regions
(`regiao_fonte`, `regiao_fundo`, in detector units of 0.05 arcsec; the atlas re-extracts piled-up imaging
observations as an annulus inside `regiao_fonte`, and timing-mode observations around the RAWX peak); the
pile-up assessment with `epatplot` (`pileup`); the position used by `barycen` and its ephemeris
(`baricentro`); and the period from the epoch-folding search (`periodo_inicial`), which is the centre of the
pulsation search for the observations without a Fermi/GBM frequency. With these products and this file,
`rodar_atlas.py` reproduces `results/atlas/`; on a different CPU the numbers can differ in the last digits
(relative differences of order 1e-14), which leaves Table A2 and Figs A1-A2 unchanged.

## Running

    pip install -r requirements.txt
    python code/baixar_dados.py
    PY=python3 NPROC=12 ./run_all.sh

`run_all.sh` rewrites `results/` and writes the tables and figures of the paper to `output/`. On 12 cores it
takes about an hour, most of it in the parametric bootstraps of the hierarchical model (10^4 simulations per
normalization in `h1_calibracao.py`); `NBOOT_H1` and `NBOOT_REV` reduce the number of simulations for a quick
check, which changes only the bootstrap p values and the calibrated thresholds. Set `XMM_PRODUCTS` to the
root of the reduced XMM-Newton products (previous section) to rebuild the atlas as well.

The products were made with Python 3.12 and the package versions in `requirements.txt`; the bootstrap
simulations of the hierarchical model and the atlas ran with Python 3.11, numpy 2.4.6, scipy 1.17.1 and
astropy 8.0.2.

## Pipeline and where each result comes from

| Step | Script | Products | In the paper |
|---|---|---|---|
| XMM-Newton atlas | `rodar_atlas.py`, `bexrb/atlas_obs.py`, `atlas_correcoes.py`, `atlas_tentativas_mc.py` | `atlas/` | Appendix A, Table A2, Figs A1-A2 |
| Spin-up points | `torque.py` | `torque/pontos_*.csv` | Section 3.1, Fig. 1 |
| Transition compilation | `sintese.py` | `sintese/transicoes_h1.csv` | Section 5, Fig. 3 |
| Global torque fits | `torque_ajuste.py` | `torque/ajuste_global*.csv`, `perfil_*.npz` | Section 4.1, Table 2, Fig. 1 |
| Torque reversal, K_bol | `reversao.py` | `torque/reversao*.csv`, `reversao_bootstrap.json`, `kbol.csv` | Sections 3.2, 4.2, Table B1 |
| Field-discrepancy index | `indice.py` | `sintese/indice.json` | Sections 4.2-4.4, Table 3, Fig. 2 |
| Hierarchical model | `h1_hierarquico.py` | `sintese/h1_hierarquico.json` | Section 5, Table 4 |
| Bootstrap, Bayes factors, prediction | `h1_calibracao.py` | `sintese/h1_calibracao.json` | Section 5 |
| Power, profile-likelihood test | `h1_poder.py`, `h1_lrt_perfil.py` | `sintese/h1_poder.json`, `h1_lrt_perfil.json` | Section 5, Appendix B |
| Leave one source out, dipole sensitivity | `h1_robustez.py` | `sintese/h1_robustez.json` | Appendix B, Table B2 |
| Tables | `tabelas_paper.py` | `output/tables/*.tex` | Tables 2-4, A1, A2, B1, B2 |
| Figures | `figuras_paper.py` | `output/figures/*.pdf` | Figs 1-3, A1, A2 |

All simulations use fixed seeds, set in each script (`indice.py`, `reversao.py`, `h1_calibracao.py`,
`h1_poder.py`, `h1_lrt_perfil.py`, `h1_robustez.py`, `atlas_tentativas_mc.py`; the atlas uses the ObsID).
Each simulation draws its own seed from them, so the results do not depend on the number of processes.

## License

MIT; see `LICENSE`.
