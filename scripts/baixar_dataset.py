"""Baixa os dois datasets do Kaggle usados no projeto.

1. Harry Potter Sorting Dataset (Sahitya Palacharla) - Apache 2.0
   1000 alunos ficticios com notas de 0 a 10 em oito tracos.
   https://www.kaggle.com/datasets/sahityapalacharla/harry-potter-sorting-dataset

2. Hogwarts Archives (sthuthi marathe)
   5387 personagens reais extraidos do Harry Potter Fandom, dos quais 985 tem
   uma das quatro casas. E a base do modelo de triagem cadastral.
   https://www.kaggle.com/datasets/sthuthimarathe/hogwarts-archives-characters-spells-and-potions

Os CSVs ja vao versionados em data/. Esse script serve para documentar a origem
e para baixar de novo se precisar.

Uso: python scripts/baixar_dataset.py
"""

import io
import urllib.request
import zipfile
from pathlib import Path

DESTINO = Path(__file__).resolve().parents[1] / "data"
BASE_KAGGLE = "https://www.kaggle.com/api/v1/datasets/download"

DATASETS = [
    ("sahityapalacharla/harry-potter-sorting-dataset",
     "harry_potter_1000_students.csv"),
    ("sthuthimarathe/hogwarts-archives-characters-spells-and-potions",
     "harry_potter_master_data.csv"),
]


def baixar(slug, arquivo):
    url = f"{BASE_KAGGLE}/{slug}"
    print(f"Baixando {slug}")

    with urllib.request.urlopen(url) as resposta:
        conteudo = resposta.read()

    # O endpoint sempre devolve um .zip, mesmo com um unico arquivo dentro
    with zipfile.ZipFile(io.BytesIO(conteudo)) as zip_file:
        zip_file.extract(arquivo, DESTINO)

    caminho = DESTINO / arquivo
    print(f"   {caminho.name} ({caminho.stat().st_size / 1024:.1f} kB)")


def main():
    DESTINO.mkdir(exist_ok=True)
    for slug, arquivo in DATASETS:
        baixar(slug, arquivo)
    print("Pronto.")


if __name__ == "__main__":
    main()
