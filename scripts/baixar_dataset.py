"""Baixa o dataset do Kaggle usado pelo Chapeu Seletor.

Dataset: "Harry Potter Sorting Dataset" (Sahitya Palacharla), licenca Apache 2.0.
https://www.kaggle.com/datasets/sahityapalacharla/harry-potter-sorting-dataset

O CSV ja vai versionado em data/, entao esse script serve mais para documentar
de onde o arquivo veio e para conseguir baixar de novo se precisar.

Uso: python scripts/baixar_dataset.py
"""

import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://www.kaggle.com/api/v1/datasets/download/sahityapalacharla/harry-potter-sorting-dataset"
DESTINO = Path(__file__).resolve().parents[1] / "data"
ARQUIVO = "harry_potter_1000_students.csv"


def main():
    DESTINO.mkdir(exist_ok=True)

    print(f"Baixando dataset de {URL}")
    with urllib.request.urlopen(URL) as resposta:
        conteudo = resposta.read()

    # O endpoint devolve um .zip mesmo quando tem um unico arquivo dentro
    with zipfile.ZipFile(io.BytesIO(conteudo)) as zip_file:
        print("Arquivos no pacote:", zip_file.namelist())
        zip_file.extract(ARQUIVO, DESTINO)

    caminho = DESTINO / ARQUIVO
    print(f"Pronto: {caminho} ({caminho.stat().st_size / 1024:.1f} kB)")


if __name__ == "__main__":
    main()
