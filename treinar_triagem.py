"""Treina o modelo de triagem cadastral e exporta o artefato da API.

Esse e o segundo modelo do projeto. Enquanto o `treinar_modelo.py` aprende a
casa a partir das notas de tracos de personalidade (dataset sintetico, 99.5%),
aqui a entrada e a ficha cadastral de personagens reais do Fandom.

O teto e bem mais baixo - cerca de 44% contra 30.8% de chutar sempre Grifinoria -
e a razao esta no notebook 03: a wiki registra o que o personagem e (especie,
nacionalidade, cor de cabelo), nao como ele e. Por isso esse modelo entra no
produto como triagem, nunca como decisao.

Uso: python treinar_triagem.py
"""

from datetime import date
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from features_triagem import (CAMPOS_TEXTO, CASAS, CATEGORICAS, VAZIO, decada,
                              extrair_sobrenome, limpar_vazamento, procurar,
                              MADEIRAS, NUCLEOS)

RAIZ = Path(__file__).resolve().parent
DATASET = RAIZ / "data" / "harry_potter_master_data.csv"
ARTEFATO = RAIZ / "modelos" / "triagem_cadastral.pkl"

SEED = 42
C_REGULARIZACAO = 5          # escolhido na comparacao do notebook 03
VERSAO_MODELO = "1.0.0"

# Acima disso a triagem ja vale como sugestao forte. Bem mais baixo que o limite
# do outro modelo porque aqui a acuracia e de ~44% - o numero saiu da
# distribuicao de confianca observada no notebook 03.
LIMITE_CONFIANCA = 0.50


def carregar_personagens():
    """Le a wiki, fica so com personagens que tem uma das quatro casas."""
    bruto = pd.read_csv(DATASET, low_memory=False)
    personagens = bruto[bruto["entity_type"] == "characters"]
    return personagens[personagens["house"].isin(CASAS)].reset_index(drop=True)


def preparar(df):
    """Aplica a engenharia de features descrita no notebook 03."""
    df = df.copy()

    # Limpa o nome das casas dos campos de texto antes de qualquer coisa
    for coluna in CAMPOS_TEXTO:
        df[coluna] = df[coluna].map(limpar_vazamento)

    df["sobrenome"] = df["name"].map(extrair_sobrenome)
    df["varinha_madeira"] = df["wands"].map(lambda t: procurar(t, MADEIRAS))
    df["varinha_nucleo"] = df["wands"].map(lambda t: procurar(t, NUCLEOS))
    df["decada_nasc"] = df["born"].map(decada)
    df["texto"] = df[CAMPOS_TEXTO].astype(str).agg(" ".join, axis=1).str.lower()
    return df


def casa_da_familia(indice, df):
    """Casa mais comum entre os OUTROS personagens com o mesmo sobrenome.

    Leave-one-out: incluir o proprio personagem seria entregar a resposta.
    """
    sobrenome = df.loc[indice, "sobrenome"]
    if not sobrenome:
        return VAZIO
    parentes = df[(df["sobrenome"] == sobrenome) & (df.index != indice)]
    return VAZIO if parentes.empty else parentes["house"].mode().iat[0]


def montar_X(df):
    X = df[CATEGORICAS].astype("object").fillna(VAZIO).astype(str)
    X = X.apply(lambda coluna: coluna.str.strip()).replace(
        {"": VAZIO, "[]": VAZIO, "nan": VAZIO})
    X["texto"] = df["texto"].values
    return X


def construir_modelo():
    return Pipeline([
        ("pre", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=2),
             CATEGORICAS),
            ("txt", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                    sublinear_tf=True), "texto"),
        ])),
        ("clf", LogisticRegression(C=C_REGULARIZACAO, max_iter=4000,
                                   random_state=SEED)),
    ])


def avaliar(df):
    """Holdout 80/20. O mapa de sobrenomes do teste vem SO do treino."""
    treino, teste = train_test_split(df, test_size=0.2, random_state=SEED,
                                     stratify=df["house"])
    treino, teste = treino.copy(), teste.copy()

    treino["casa_sobrenome"] = [casa_da_familia(i, treino) for i in treino.index]

    mapa = (treino[treino["sobrenome"] != ""]
            .groupby("sobrenome")["house"]
            .agg(lambda casas: casas.mode().iat[0]))
    teste["casa_sobrenome"] = teste["sobrenome"].map(mapa).fillna(VAZIO)

    modelo = construir_modelo()
    modelo.fit(montar_X(treino), treino["house"])
    y_pred = modelo.predict(montar_X(teste))

    print(classification_report(teste["house"], y_pred))
    return {
        "acuracia_teste": float(accuracy_score(teste["house"], y_pred)),
        "f1_macro_teste": float(f1_score(teste["house"], y_pred, average="macro")),
    }


def main():
    print("[1/5] Carregando os personagens reais...")
    df = carregar_personagens()
    print(f"      {len(df)} personagens com casa")
    print("      " + " | ".join(f"{casa}: {n}"
                                for casa, n in df["house"].value_counts().items()))

    print("[2/5] Construindo as features...")
    df = preparar(df)
    df["casa_sobrenome"] = [casa_da_familia(i, df) for i in df.index]
    aproveitaveis = (df["sobrenome"] != "").sum()
    print(f"      {aproveitaveis} personagens com sobrenome aproveitavel")

    print("[3/5] Avaliando (holdout 80/20 + validacao cruzada)...")
    metricas = avaliar(df)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(construir_modelo(), montar_X(df), df["house"],
                             cv=cv, scoring="accuracy")
    metricas["acuracia_cv_media"] = float(scores.mean())
    metricas["acuracia_cv_desvio"] = float(scores.std())

    chute = float(df["house"].value_counts(normalize=True).max())
    metricas["baseline_classe_majoritaria"] = chute

    print(f"      Acuracia no teste: {metricas['acuracia_teste'] * 100:.1f}%")
    print(f"      Acuracia 5-fold:   {metricas['acuracia_cv_media'] * 100:.1f}% "
          f"(+/- {metricas['acuracia_cv_desvio'] * 100:.1f})")
    print(f"      Baseline (chute):  {chute * 100:.1f}%")

    print("[4/5] Retreinando com os 985 personagens...")
    modelo = construir_modelo()
    modelo.fit(montar_X(df), df["house"])

    # O mapa final usa o dataset inteiro, porque na producao qualquer personagem
    # conhecido pode ser parente do aluno que chegou
    mapa_final = (df[df["sobrenome"] != ""]
                  .groupby("sobrenome")["house"]
                  .agg(lambda casas: casas.mode().iat[0])
                  .to_dict())

    # Valores que o formulario da secretaria oferece em cada campo. Guardo aqui
    # porque quem sabe quais valores o modelo viu no treino e o treino, nao a API
    opcoes = {}
    for coluna in ["blood_status", "gender", "species", "nationality",
                   "eye_color", "hair_color", "skin_color", "marital_status",
                   "patronus", "boggart"]:
        valores = (df[coluna].astype("object").astype(str).str.strip()
                   .replace({"": None, "nan": None, "[]": None}).dropna())
        frequentes = valores.value_counts()
        opcoes[coluna] = sorted(frequentes[frequentes >= 3].index.tolist())
    opcoes["varinha_madeira"] = MADEIRAS
    opcoes["varinha_nucleo"] = NUCLEOS

    print("[5/5] Exportando o artefato...")
    artefato = {
        "modelo": modelo,
        "mapa_sobrenome": mapa_final,
        "opcoes": opcoes,
        "categoricas": CATEGORICAS,
        "classes": list(modelo.classes_),
        "versao": VERSAO_MODELO,
        "limite_confianca": LIMITE_CONFIANCA,
        "metricas": metricas,
        "treinado_em": date.today().isoformat(),
    }
    ARTEFATO.parent.mkdir(exist_ok=True)
    joblib.dump(artefato, ARTEFATO)

    tamanho = ARTEFATO.stat().st_size / 1024
    print(f"      Salvo em {ARTEFATO.relative_to(RAIZ)} ({tamanho:.1f} kB)")
    print(f"      {len(mapa_final)} sobrenomes conhecidos no mapa de familias")


if __name__ == "__main__":
    main()
