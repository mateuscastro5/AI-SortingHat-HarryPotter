"""Treina o Chapeu Seletor Completo e exporta o artefato da API.

Terceiro modelo do projeto. Os outros dois sao independentes: o Chapeu Seletor
(`treinar_modelo.py`) decide a partir das notas de personalidade; a Triagem
Cadastral (`treinar_triagem.py`) sugere a partir da ficha. Aqui os dois se
conectam: o Completo recebe as 8 notas *mais* o resultado da triagem (o
"resmungo" do chapeu antes de decidir), imitando a cena do filme em duas
etapas.

O problema de dados: nenhum aluno do acervo tem notas de personalidade E ficha
cadastral genuinas ao mesmo tempo (dataset 1 e so notas, dataset 2 e so ficha -
ver `context.md`). Pra treinar o Completo, cada um dos 1000 alunos ficticios
recebe um sobrenome sintetico - ou nenhum -, calibrado em duas taxas MEDIDAS
no acervo real, nao inventadas:

  - so 14% tem linhagem reconhecida (o resto fica "desconhecido", igual a
    cobertura real medida no acervo - ver Limitacoes no README);
  - entre os que tem, o sobrenome bate com a casa verdadeira do aluno em
    59.9% das vezes (a mesma taxa de concordancia ja medida no notebook 03).

Sem a primeira taxa, todo aluno do treino teria familia conhecida - raro na
producao real - e o modelo nunca aprenderia a lidar com o caso comum (sem
familia). Sem a segunda, o sobrenome vira ruido sem correlacao com o rotulo.

Criterio de sucesso (os tres precisam passar - ver context.md):
  1. Guardrail:  acuracia 5-fold nao pode cair mais que ~1 p.p. da original.
  2. Valor real: nos casos ambiguos (confianca do modelo original < 95%), o
     Completo acerta MAIS que o modelo original.
  3. Sanidade:   a acuracia do Completo fica bem acima dos ~44% da triagem
     sozinha - confirma que as notas continuam sendo o sinal dominante.

Comparacao de acuracia bruta entre original e Completo NAO decide sozinha -
o Completo injeta um sinal real (59.9%) que puxa a acuracia pra baixo por
construcao, mesmo sendo genuino. Ver a justificativa completa no context.md.

Testa 3 formatos de entrada pra "resultado da triagem" e varios C de
regularizacao pra cada um, porque o C=5 do modelo original foi calibrado pra
8 features, nao pras 12+ daqui:
  A - as 4 probabilidades da triagem, cruas.
  B - casa sugerida (categorica) + confianca.
  C - features de interacao nota-do-traco x probabilidade-da-triagem-da-casa
      correspondente ("reforco"), sem as probabilidades cruas - deixa o
      modelo aprender a reforcar so quando os dois sinais concordam, em vez
      de aplicar o mesmo empurrao linear mesmo quando a triagem errou.

Uso: python treinar_completo.py
"""

from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features_triagem import VAZIO, montar_linha

RAIZ = Path(__file__).resolve().parent
DATASET = RAIZ / "data" / "harry_potter_1000_students.csv"
ARTEFATO_TRIAGEM = RAIZ / "modelos" / "triagem_cadastral.pkl"
ARTEFATO = RAIZ / "modelos" / "chapeu_seletor_completo.pkl"

FEATURES = [
    "Bravery", "Intelligence", "Loyalty", "Ambition",
    "Dark Arts Knowledge", "Quidditch Skills", "Dueling Skills", "Creativity",
]
ALVO = "House"

SEED = 42
C_PADRAO = 5
GRADE_C = [0.5, 1, 2, 5, 10, 20, 50]
VERSAO_MODELO = "1.0.0"
LIMITE_CONFIANCA = 0.80
LIMITE_AMBIGUO = 0.95  # corte usado so pra escolher a variante (mais amostra que o LIMITE_CONFIANCA)

TAXA_CONCORDANCIA_FAMILIA = 0.599
TAXA_COBERTURA_LINHAGEM = 0.14

TRACO_PRINCIPAL = {
    "Gryffindor": "Bravery",
    "Ravenclaw": "Intelligence",
    "Hufflepuff": "Loyalty",
    "Slytherin": "Ambition",
}


# --------------------------------------------------------------------------
# Dados
# --------------------------------------------------------------------------

def carregar_dados():
    df = pd.read_csv(DATASET)
    return df[FEATURES].reset_index(drop=True), df[ALVO].reset_index(drop=True)


def carregar_triagem():
    """Reaproveita o modelo e o mapa de familias ja treinados em producao."""
    artefato = joblib.load(ARTEFATO_TRIAGEM)
    return artefato["modelo"], artefato["mapa_sobrenome"], artefato["classes"]


def agrupar_sobrenomes_por_casa(mapa_sobrenome):
    grupos = {}
    for sobrenome, casa in mapa_sobrenome.items():
        grupos.setdefault(casa, []).append(sobrenome)
    return grupos


def sortear_sobrenomes(casas_verdadeiras, grupos_por_casa, rng):
    """Um sobrenome sintetico por aluno - ou nenhum -, nas duas taxas reais
    descritas no cabecalho do arquivo.
    """
    todas_casas = list(grupos_por_casa.keys())
    sobrenomes = []
    for casa_verdadeira in casas_verdadeiras:
        if rng.random() >= TAXA_COBERTURA_LINHAGEM:
            sobrenomes.append(None)
            continue
        if rng.random() < TAXA_CONCORDANCIA_FAMILIA:
            casa_sorteio = casa_verdadeira
        else:
            outras = [c for c in todas_casas if c != casa_verdadeira]
            casa_sorteio = rng.choice(outras)
        sobrenomes.append(rng.choice(grupos_por_casa[casa_sorteio]))
    return sobrenomes


def rodar_triagem_sintetica(sobrenomes, modelo_triagem, mapa_sobrenome, classes_triagem):
    """Roda a Triagem de verdade em cima do sobrenome sintetico de cada aluno.

    Ficha em branco no resto - exatamente como a Triagem ja trata hoje um
    aluno com ficha incompleta em producao.
    """
    linhas = []
    for sobrenome in sobrenomes:
        nome = f"Aluno {sobrenome.capitalize()}" if sobrenome else ""
        linhas.append(montar_linha({"nome": nome}, mapa_sobrenome))
    entrada = pd.concat(linhas, ignore_index=True)
    probabilidades = modelo_triagem.predict_proba(entrada)

    colunas_prob = [f"triagem_{casa}" for casa in classes_triagem]
    df_prob = pd.DataFrame(probabilidades, columns=colunas_prob)

    indice_top = np.argmax(probabilidades, axis=1)
    df_prob["triagem_casa_sugerida"] = [classes_triagem[i] for i in indice_top]
    df_prob["triagem_confianca"] = probabilidades[np.arange(len(probabilidades)), indice_top]
    df_prob["casa_sobrenome"] = entrada["casa_sobrenome"].values
    return df_prob


def adicionar_reforco(dados, classes_triagem):
    """Feature de interacao: nota do traco principal x probabilidade da
    triagem pra casa correspondente (`Bravery x triagem_Gryffindor`, etc.).
    """
    dados = dados.copy()
    colunas_reforco = []
    for casa in classes_triagem:
        traco = TRACO_PRINCIPAL.get(casa)
        if traco is None:
            continue
        coluna = f"reforco_{casa}"
        dados[coluna] = dados[traco] * dados[f"triagem_{casa}"]
        colunas_reforco.append(coluna)
    return dados, colunas_reforco


# --------------------------------------------------------------------------
# Modelos candidatos
# --------------------------------------------------------------------------

def construir_original(c=C_PADRAO):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=c, max_iter=5000, random_state=SEED)),
    ])


def construir_variante_a(colunas_prob, c=C_PADRAO):
    """As 4 probabilidades da triagem, cruas."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=c, max_iter=5000, random_state=SEED)),
    ]), FEATURES + colunas_prob


def construir_variante_b(c=C_PADRAO):
    """Casa sugerida (categorica) + confianca (numerica)."""
    numericas = FEATURES + ["triagem_confianca"]
    pipeline = Pipeline([
        ("pre", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["triagem_casa_sugerida"]),
            ("num", StandardScaler(), numericas),
        ])),
        ("clf", LogisticRegression(C=c, max_iter=5000, random_state=SEED)),
    ])
    return pipeline, numericas + ["triagem_casa_sugerida"]


def construir_variante_c(colunas_reforco, c=C_PADRAO):
    """Notas + interacao nota x triagem ("reforco"), sem as probs cruas."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=c, max_iter=5000, random_state=SEED)),
    ]), FEATURES + colunas_reforco


# --------------------------------------------------------------------------
# Avaliacao
# --------------------------------------------------------------------------

def avaliar_candidato(nome, c, construtor, colunas, dados, y, cv, classes_ord, limiar_ambiguo):
    pipeline, _ = construtor
    X = dados[colunas]
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    proba_oof = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")
    pred_oof = np.array(classes_ord)[proba_oof.argmax(axis=1)]
    confianca_oof = proba_oof.max(axis=1)
    return {
        "variante": nome, "C": c,
        "acc_geral": scores.mean(), "acc_geral_desvio": scores.std(),
        "pred_oof": pred_oof, "confianca_oof": confianca_oof,
    }


def gerar_perfis_de_robustez(n, rng):
    """Mesmo teste do notebook 02: 4 tracos principais entre 5 e 10, resto
    neutro em 5. Sem casa verdadeira - so serve pra comparar cada modelo
    contra a 'regra do maior atributo'.
    """
    tracos_principais = ["Bravery", "Intelligence", "Loyalty", "Ambition"]
    perfis = pd.DataFrame({
        traco: rng.integers(5, 11, size=n) for traco in tracos_principais
    })
    for traco in ["Dark Arts Knowledge", "Quidditch Skills", "Dueling Skills", "Creativity"]:
        perfis[traco] = 5
    perfis = perfis[FEATURES]

    ordem_casas = [TRACO_PRINCIPAL_INVERSO[t] for t in tracos_principais]
    indice_dominante = perfis[tracos_principais].values.argmax(axis=1)
    dominante = [ordem_casas[i] for i in indice_dominante]
    return perfis, dominante


TRACO_PRINCIPAL_INVERSO = {v: k for k, v in TRACO_PRINCIPAL.items()}


def main():
    rng = np.random.default_rng(SEED)

    print("[1/6] Carregando os 1000 alunos ficticios e a Triagem ja treinada...")
    X_notas, y = carregar_dados()
    modelo_triagem, mapa_sobrenome, classes_triagem = carregar_triagem()
    grupos_por_casa = agrupar_sobrenomes_por_casa(mapa_sobrenome)
    print(f"      {len(X_notas)} alunos | {len(mapa_sobrenome)} sobrenomes reais conhecidos")

    print(f"[2/6] Sorteando sobrenome sintetico por aluno "
          f"({TAXA_COBERTURA_LINHAGEM * 100:.0f}% com linhagem, calibrado em "
          f"{TAXA_CONCORDANCIA_FAMILIA * 100:.1f}% de concordancia entre esses)...")
    sobrenomes = sortear_sobrenomes(y.tolist(), grupos_por_casa, rng)

    print("[3/6] Rodando a Triagem de verdade sobre o sobrenome sintetico...")
    df_triagem = rodar_triagem_sintetica(sobrenomes, modelo_triagem, mapa_sobrenome, classes_triagem)
    cobertura = (df_triagem["casa_sobrenome"] != VAZIO).mean()
    print(f"      {cobertura * 100:.1f}% dos alunos sinteticos ficaram com familia conhecida")

    dados = pd.concat([X_notas, df_triagem], axis=1)
    dados, colunas_reforco = adicionar_reforco(dados, classes_triagem)
    colunas_prob = [f"triagem_{casa}" for casa in classes_triagem]
    cols_a = FEATURES + colunas_prob
    cols_b = FEATURES + ["triagem_confianca", "triagem_casa_sugerida"]
    cols_c = FEATURES + colunas_reforco

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    classes_ord = sorted(y.unique())
    y_arr = y.to_numpy()

    print("[4/6] Baseline: modelo original (so as 8 notas)...")
    scores_original = cross_val_score(construir_original(), X_notas, y, cv=cv, scoring="accuracy")
    proba_original_oof = cross_val_predict(construir_original(), X_notas, y, cv=cv, method="predict_proba")
    pred_original_oof = np.array(classes_ord)[proba_original_oof.argmax(axis=1)]
    confianca_original_oof = proba_original_oof.max(axis=1)
    ambiguos = confianca_original_oof < LIMITE_AMBIGUO
    acc_original_geral = scores_original.mean()
    acc_original_amb = accuracy_score(y_arr[ambiguos], pred_original_oof[ambiguos])
    print(f"      Acuracia geral: {acc_original_geral * 100:.2f}% "
          f"(+/- {scores_original.std() * 100:.2f})")
    print(f"      Acuracia nos {ambiguos.sum()} casos ambiguos (confianca <{LIMITE_AMBIGUO*100:.0f}%): "
          f"{acc_original_amb * 100:.1f}%")

    print("[5/6] Testando variantes A/B/C x grade de C de regularizacao...")
    candidatas = [("B", cols_b, construir_variante_b(c=C_PADRAO), C_PADRAO)]
    for c in GRADE_C:
        candidatas.append(("A", cols_a, construir_variante_a(colunas_prob, c=c), c))
        candidatas.append(("C", cols_c, construir_variante_c(colunas_reforco, c=c), c))

    resultados = []
    for nome, colunas, construtor, c in candidatas:
        r = avaliar_candidato(nome, c, construtor, colunas, dados, y, cv, classes_ord, LIMITE_AMBIGUO)
        acc_amb = accuracy_score(y_arr[ambiguos], r["pred_oof"][ambiguos])
        r["acc_ambiguos"] = acc_amb
        r["guardrail"] = r["acc_geral"] >= acc_original_geral - 0.01
        r["valor"] = acc_amb > acc_original_amb
        r["sanidade"] = r["acc_geral"] > 0.44
        resultados.append(r)
        marca = "OK " if (r["guardrail"] and r["valor"] and r["sanidade"]) else "   "
        print(f"      [{marca}] {nome} C={c:<5} geral={r['acc_geral']*100:5.2f}%  "
              f"ambiguos={acc_amb*100:5.1f}%  "
              f"{'guardrail ' if not r['guardrail'] else ''}"
              f"{'valor ' if not r['valor'] else ''}")

    aprovados = [r for r in resultados if r["guardrail"] and r["valor"] and r["sanidade"]]
    if aprovados:
        vencedor = max(aprovados, key=lambda r: (r["acc_ambiguos"], r["acc_geral"]))
        print(f"      >>> Passou nos 3 criterios: variante {vencedor['variante']} C={vencedor['C']}")
    else:
        vencedor = max(resultados, key=lambda r: (r["acc_ambiguos"], r["acc_geral"]))
        print(f"      >>> Nenhuma combinacao passou nos 3 criterios. "
              f"Melhor tentativa: variante {vencedor['variante']} C={vencedor['C']}")

    print("[6/6] Teste de robustez do vencedor (dois tracos altos, sem familia conhecida)...")
    perfis, dominante = gerar_perfis_de_robustez(1377, rng)

    original_final = construir_original()
    original_final.fit(X_notas, y)
    concordancia_original = accuracy_score(dominante, original_final.predict(perfis))

    linha_vazia = montar_linha({"nome": ""}, mapa_sobrenome)
    proba_vazia = modelo_triagem.predict_proba(linha_vazia)[0]

    if vencedor["variante"] == "A":
        construtor_venc = construir_variante_a(colunas_prob, c=vencedor["C"])
        colunas_venc = cols_a
        perfis_venc = perfis.copy()
        for casa, p in zip(classes_triagem, proba_vazia):
            perfis_venc[f"triagem_{casa}"] = p
    elif vencedor["variante"] == "C":
        construtor_venc = construir_variante_c(colunas_reforco, c=vencedor["C"])
        colunas_venc = cols_c
        perfis_venc = perfis.copy()
        for casa, p in zip(classes_triagem, proba_vazia):
            traco = TRACO_PRINCIPAL.get(casa)
            if traco:
                perfis_venc[f"reforco_{casa}"] = perfis_venc[traco] * p
    else:  # B
        construtor_venc = construir_variante_b(c=vencedor["C"])
        colunas_venc = cols_b
        indice_vazio = int(np.argmax(proba_vazia))
        perfis_venc = perfis.copy()
        perfis_venc["triagem_confianca"] = proba_vazia[indice_vazio]
        perfis_venc["triagem_casa_sugerida"] = classes_triagem[indice_vazio]

    modelo_vencedor, _ = construtor_venc
    modelo_vencedor.fit(dados[colunas_venc], y)
    concordancia_vencedor = accuracy_score(dominante, modelo_vencedor.predict(perfis_venc[colunas_venc]))

    print(f"      Original:                          {concordancia_original * 100:.1f}%")
    print(f"      {vencedor['variante']} (C={vencedor['C']}), sem familia conhecida: "
          f"{concordancia_vencedor * 100:.1f}%")

    metricas = {
        "acuracia_cv_original": float(acc_original_geral),
        "acuracia_ambiguos_original": float(acc_original_amb),
        "concordancia_robustez_original": float(concordancia_original),
        "variante_escolhida": vencedor["variante"],
        "c_escolhido": float(vencedor["C"]),
        "acuracia_cv_completo": float(vencedor["acc_geral"]),
        "acuracia_ambiguos_completo": float(vencedor["acc_ambiguos"]),
        "concordancia_robustez_completo": float(concordancia_vencedor),
        "taxa_concordancia_familia_sintetica": TAXA_CONCORDANCIA_FAMILIA,
        "taxa_cobertura_linhagem_sintetica": TAXA_COBERTURA_LINHAGEM,
        "guardrail_passou": bool(vencedor["guardrail"]),
        "valor_real_passou": bool(vencedor["valor"]),
        "sanidade_passou": bool(vencedor["sanidade"]),
        "todas_as_tentativas": [
            {"variante": r["variante"], "C": float(r["C"]),
             "acuracia_geral": float(r["acc_geral"]),
             "acuracia_ambiguos": float(r["acc_ambiguos"])}
            for r in resultados
        ],
    }

    artefato = {
        "modelo": modelo_vencedor,
        "variante": vencedor["variante"],
        "features": FEATURES,
        "colunas_entrada": colunas_venc,
        "classes": list(modelo_vencedor.classes_),
        "classes_triagem": classes_triagem,
        "versao": VERSAO_MODELO,
        "limite_confianca": LIMITE_CONFIANCA,
        "taxa_concordancia_familia_sintetica": TAXA_CONCORDANCIA_FAMILIA,
        "taxa_cobertura_linhagem_sintetica": TAXA_COBERTURA_LINHAGEM,
        "metricas": metricas,
        "treinado_em": date.today().isoformat(),
    }
    ARTEFATO.parent.mkdir(exist_ok=True)
    joblib.dump(artefato, ARTEFATO)

    tamanho = ARTEFATO.stat().st_size / 1024
    print(f"      Salvo em {ARTEFATO.relative_to(RAIZ)} ({tamanho:.1f} kB)")


if __name__ == "__main__":
    main()
