"""Baixa os dados públicos de terceiros que o código lê e que não fazem parte do repositório:

  dados/gbm/<fonte>.fits.gz   históricos de frequência do Fermi/GBM Accreting Pulsars Program
  dados/maxi/<J...>_1day.dat  curvas de luz diárias do MAXI/GSC (2-20 keV e bandas), das fontes e do Crab

Depois do download, compara o SHA-256 de cada arquivo com o da versão usada no paper (2026-10-27). Versões
mais novas diferem porque os dois arquivos crescem com o tempo e o MAXI revisa os dias mais recentes;
torque.MJD_MAX corta os dados antes desses dias e torque.CRAB_MAXI fixa a normalização pelo Crab, de modo que
os pontos são os mesmos enquanto os dados mais antigos não forem reprocessados.

Uso: python baixar_dados.py
"""
import hashlib
import os
import urllib.request

AQUI = os.path.dirname(os.path.abspath(__file__))
GBM = "https://gammaray.nsstc.nasa.gov/gbm/science/pulsars/lightcurves/{}.fits.gz"
MAXI = "http://maxi.riken.jp/star_data/{0}/{0}_g_lc_1day_all.dat"
# SHA-256 das versões usadas no paper
USADOS = {
    "gbm/4u0115.fits.gz": "ec13cdf9d0bfbcdb504d4ca67b36570780633849106356ab970c5b58af286342",
    "gbm/a0535.fits.gz": "f1bd7a7aa218dbbb2a2a93e7b9795c5b1a074034fa3e887b5ea7f978c627f163",
    "gbm/swiftj0243.fits.gz": "69b56d5ea0492a6f3a9804697a555292e4c4a41b64942b4f51566b487a751ae4",
    "gbm/v0332.fits.gz": "a0ea5700d675a3eaa0d6e9b63c9487155bbc50fc5c2349a460315fcb13541610",
    "maxi/J0118+637_1day.dat": "f8a69d073d2cefe60299721ac28180d277ab3ddfd800424a884f040f569931bb",
    "maxi/J0243+614_1day.dat": "75ea028bcd85b07a2f6d4a8bf5c15dface39de42675b5470389a1684ce65ff93",
    "maxi/J0334+531_1day.dat": "eee07cd016be9324d5e7269b0a20c2a7ee899d0e65e9b64b9e17823376fa7e45",
    "maxi/J0355+310_1day.dat": "2bd4f3b34ca99088b685a902e5fbbc39f3efeddd286ad49d424477ee79c7944e",
    "maxi/J0534+220_1day.dat": "79ca4fdac68b1bf1931c26373690a7a3a259cedf8c78ca1ca82110022d220f4c",
    "maxi/J0538+263_1day.dat": "d95aecf7f8fe87785e0c72260947f5e31b9b48e8fc9a542aec593a224f8761c0",
}


def baixar(url, destino):
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as r, open(destino, "wb") as f:
        f.write(r.read())


if __name__ == "__main__":
    alvos = [(GBM.format(n), f"gbm/{n}.fits.gz") for n in ("4u0115", "v0332", "a0535", "swiftj0243")]
    alvos += [(MAXI.format(n), f"maxi/{n}_1day.dat")
              for n in ("J0118+637", "J0334+531", "J0538+263", "J0243+614", "J0355+310", "J0534+220")]
    for url, rel in alvos:
        destino = os.path.join(AQUI, "dados", rel)
        baixar(url, destino)
        h = hashlib.sha256(open(destino, "rb").read()).hexdigest()
        print(f"{rel:28s} {'igual à versão do paper' if h == USADOS[rel] else 'difere da versão do paper (dados mais novos?)'}")
