"""Treina o Chapeu Seletor e exporta o artefato que a API carrega.

A escolha do algoritmo e do hiperparametro esta documentada no notebook
`notebooks/02_experimentacao_modelos.ipynb`. Resumo da decisao:

- seis algoritmos empataram acima de 99% de acuracia, entao acuracia sozinha nao
  serviu como criterio;
- o Gradient Boosting (o mais preciso) foi descartado porque quebra com alunos
  bons em varios atributos ao mesmo tempo - manda Harry e Hermione pra Lufa-Lufa;
- ficou a Regressao Logistica com C=5: mesma acuracia dos outros, melhor log loss
  entre os empatados, responde de forma coerente quando um atributo muda e deixa
  explicar a decisao pelos coeficientes.

Uso: python treinar_modelo.py
"""

from datetime import date
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             log_loss)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parent
DATASET = RAIZ / "data" / "harry_potter_1000_students.csv"
ARTEFATO = RAIZ / "modelos" / "chapeu_seletor.pkl"

# A ordem importa: a API monta o vetor de entrada seguindo exatamente essa lista
FEATURES = [
    "Bravery",
    "Intelligence",
    "Loyalty",
    "Ambition",
    "Dark Arts Knowledge",
    "Quidditch Skills",
    "Dueling Skills",
    "Creativity",
]
ALVO = "House"

SEED = 42
C_REGULARIZACAO = 5
VERSAO_MODELO = "1.0.0"

# Abaixo disso a aplicacao avisa que o chapeu ficou em duvida e sugere revisao
# humana. O valor saiu da distribuicao de confianca analisada no notebook 02.
LIMITE_CONFIANCA = 0.80


def carregar_dados():
    """Le o CSV e devolve as features e o alvo.

    `Blood Status` fica de fora de proposito: no notebook 01 o teste de
    qui-quadrado mostrou que a variavel e independente da casa (p = 0.53), e um
    sistema que decide o futuro de um aluno nao deveria olhar a ascendencia dele
    nem se ajudasse.
    """
    df = pd.read_csv(DATASET)
    return df[FEATURES], df[ALVO]


def construir_modelo():
    """Pipeline de padronizacao + regressao logistica multiclasse."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C_REGULARIZACAO,
            max_iter=5000,
            random_state=SEED,
        )),
    ])


def avaliar(X, y):
    """Treina em 80% dos dados e mede o desempenho nos 20% separados."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    modelo = construir_modelo()
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    probabilidades = modelo.predict_proba(X_test)

    metricas = {
        "acuracia_teste": float(accuracy_score(y_test, y_pred)),
        "f1_macro_teste": float(f1_score(y_test, y_pred, average="macro")),
        "log_loss_teste": float(log_loss(y_test, probabilidades,
                                         labels=list(modelo.classes_))),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(construir_modelo(), X, y, cv=cv, scoring="accuracy")
    metricas["acuracia_cv_media"] = float(scores.mean())
    metricas["acuracia_cv_desvio"] = float(scores.std())

    print(classification_report(y_test, y_pred))
    return metricas


def main():
    print("[1/4] Carregando o dataset...")
    X, y = carregar_dados()
    print(f"      {len(X)} alunos, {len(FEATURES)} atributos, "
          f"{y.nunique()} casas")

    print("[2/4] Avaliando o modelo (holdout 80/20 + validacao cruzada)...")
    metricas = avaliar(X, y)
    print(f"      Acuracia no teste:  {metricas['acuracia_teste'] * 100:.2f}%")
    print(f"      F1 macro no teste:  {metricas['f1_macro_teste'] * 100:.2f}%")
    print(f"      Log loss no teste:  {metricas['log_loss_teste']:.4f}")
    print(f"      Acuracia 5-fold:    {metricas['acuracia_cv_media'] * 100:.2f}% "
          f"(+/- {metricas['acuracia_cv_desvio'] * 100:.2f})")

    print("[3/4] Retreinando com os 1000 alunos para a versao de producao...")
    modelo = construir_modelo()
    modelo.fit(X, y)

    print("[4/4] Exportando o artefato...")
    # Guardo o modelo junto com os metadados porque a API precisa saber a ordem
    # das features e as classes sem ter que repetir essa lista no codigo dela
    artefato = {
        "modelo": modelo,
        "features": FEATURES,
        "classes": list(modelo.classes_),
        "versao": VERSAO_MODELO,
        "limite_confianca": LIMITE_CONFIANCA,
        "metricas": metricas,
        "treinado_em": date.today().isoformat(),
    }
    ARTEFATO.parent.mkdir(exist_ok=True)
    joblib.dump(artefato, ARTEFATO)

    tamanho = ARTEFATO.stat().st_size / 1024
    print(f"      Modelo salvo em {ARTEFATO.relative_to(RAIZ)} ({tamanho:.1f} kB)")


if __name__ == "__main__":
    main()
