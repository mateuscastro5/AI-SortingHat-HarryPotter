"""API do Chapeu Seletor.

Microsservico que carrega o modelo treinado por `treinar_modelo.py` e expoe as
rotas usadas pela secretaria de Hogwarts:

- POST /api/v1/selecionar        -> cerimonia de um aluno so
- POST /api/v1/selecionar-turma  -> lote, para processar a turma inteira
- GET  /api/v1/saude             -> status e metadados do modelo em uso

Rodar: uvicorn api.app:app --reload
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ARTEFATO = Path(__file__).resolve().parents[1] / "modelos" / "chapeu_seletor.pkl"

# Texto que a secretaria mostra pro aluno junto com o resultado
DESCRICAO_CASAS = {
    "Gryffindor": {
        "nome_pt": "Grifinoria",
        "lema": "Coragem, ousadia e determinacao",
        "cor": "#7F0909",
    },
    "Hufflepuff": {
        "nome_pt": "Lufa-Lufa",
        "lema": "Lealdade, trabalho duro e paciencia",
        "cor": "#D3A625",
    },
    "Ravenclaw": {
        "nome_pt": "Corvinal",
        "lema": "Inteligencia, criatividade e sabedoria",
        "cor": "#222F5B",
    },
    "Slytherin": {
        "nome_pt": "Sonserina",
        "lema": "Ambicao, astucia e engenhosidade",
        "cor": "#0D6217",
    },
}

# Nome dos atributos em portugues, pra justificativa nao sair em ingles
ATRIBUTOS_PT = {
    "Bravery": "coragem",
    "Intelligence": "inteligencia",
    "Loyalty": "lealdade",
    "Ambition": "ambicao",
    "Dark Arts Knowledge": "conhecimento em artes das trevas",
    "Quidditch Skills": "habilidade no quadribol",
    "Dueling Skills": "habilidade em duelos",
    "Creativity": "criatividade",
}

try:
    artefato = joblib.load(ARTEFATO)
except FileNotFoundError:
    raise RuntimeError(
        f"Modelo nao encontrado em {ARTEFATO}. "
        "Rode 'python treinar_modelo.py' antes de subir a API."
    )

modelo = artefato["modelo"]
FEATURES = artefato["features"]
CLASSES = artefato["classes"]
LIMITE_CONFIANCA = artefato["limite_confianca"]

app = FastAPI(
    title="Chapeu Seletor IA",
    description=(
        "Servico de selecao de casas de Hogwarts. Recebe o resultado da "
        "avaliacao de ingresso de um aluno e devolve a casa, a confianca da "
        "decisao e a justificativa."
    ),
    version=artefato["versao"],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FichaDoAluno(BaseModel):
    """Notas de 0 a 10 preenchidas pela comissao de ingresso."""

    nome: str = Field("Aluno sem nome", max_length=80, examples=["Harry Potter"])
    bravery: int = Field(..., ge=0, le=10, examples=[10], description="Coragem")
    intelligence: int = Field(..., ge=0, le=10, examples=[6], description="Inteligencia")
    loyalty: int = Field(..., ge=0, le=10, examples=[8], description="Lealdade")
    ambition: int = Field(..., ge=0, le=10, examples=[4], description="Ambicao")
    dark_arts_knowledge: int = Field(..., ge=0, le=10, examples=[3],
                                     description="Conhecimento em artes das trevas")
    quidditch_skills: int = Field(..., ge=0, le=10, examples=[10],
                                  description="Habilidade no quadribol")
    dueling_skills: int = Field(..., ge=0, le=10, examples=[9],
                                description="Habilidade em duelos")
    creativity: int = Field(..., ge=0, le=10, examples=[5], description="Criatividade")

    def para_vetor(self):
        """Monta a linha na ordem exata em que o modelo foi treinado."""
        valores = {
            "Bravery": self.bravery,
            "Intelligence": self.intelligence,
            "Loyalty": self.loyalty,
            "Ambition": self.ambition,
            "Dark Arts Knowledge": self.dark_arts_knowledge,
            "Quidditch Skills": self.quidditch_skills,
            "Dueling Skills": self.dueling_skills,
            "Creativity": self.creativity,
        }
        return [valores[feature] for feature in FEATURES]


class TurmaDeIngresso(BaseModel):
    alunos: list[FichaDoAluno] = Field(..., min_length=1, max_length=500)


class ResultadoSelecao(BaseModel):
    nome: str
    casa: str
    casa_pt: str
    lema: str
    cor: str
    confianca: float
    probabilidades: dict[str, float]
    decisao_apertada: bool
    justificativa: list[str]
    recomendacao: str


class ResumoDaTurma(BaseModel):
    total: int
    por_casa: dict[str, int]
    decisoes_apertadas: int
    resultados: list[ResultadoSelecao]


def justificar(vetor, casa):
    """Diz quais atributos mais empurraram o aluno pra casa escolhida.

    Como o modelo e uma regressao logistica dentro de um pipeline com
    StandardScaler, a contribuicao de cada atributo e simplesmente
    coeficiente * valor padronizado. Pego as tres maiores contribuicoes
    positivas.

    Atencao ao sinal: um atributo tambem pesa a favor da casa quando esta
    *abaixo* da media e o coeficiente e negativo (e o caso da Lufa-Lufa com
    habilidade em duelos, por exemplo). Por isso a frase e montada olhando o
    valor padronizado, e nao so a contribuicao.
    """
    scaler = modelo.named_steps["scaler"]
    classificador = modelo.named_steps["clf"]

    entrada = pd.DataFrame([vetor], columns=FEATURES)
    padronizado = scaler.transform(entrada)[0]
    coeficientes = classificador.coef_[list(classificador.classes_).index(casa)]

    contribuicoes = sorted(
        zip(FEATURES, coeficientes * padronizado, padronizado),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        f"{ATRIBUTOS_PT[feature]} "
        f"{'acima' if valor > 0 else 'abaixo'} da media dos candidatos"
        for feature, peso, valor in contribuicoes[:3]
        if peso > 0
    ]


def recomendar(casa, confianca):
    """Regra de negocio: o que a secretaria faz com o resultado."""
    if confianca < LIMITE_CONFIANCA:
        return (
            "Decisao apertada. Encaminhar o aluno para entrevista com a "
            "coordenacao antes de confirmar a casa - o chapeu ficou em duvida "
            "entre mais de uma opcao."
        )
    return (
        f"Casa confirmada. Emitir a carta de boas-vindas da "
        f"{DESCRICAO_CASAS[casa]['nome_pt']}, alocar o dormitorio e avisar o "
        f"monitor responsavel."
    )


def selecionar(ficha: FichaDoAluno) -> ResultadoSelecao:
    vetor = ficha.para_vetor()
    entrada = pd.DataFrame([vetor], columns=FEATURES)

    probabilidades = modelo.predict_proba(entrada)[0]
    indice = int(np.argmax(probabilidades))
    casa = CLASSES[indice]
    confianca = float(probabilidades[indice])

    return ResultadoSelecao(
        nome=ficha.nome,
        casa=casa,
        casa_pt=DESCRICAO_CASAS[casa]["nome_pt"],
        lema=DESCRICAO_CASAS[casa]["lema"],
        cor=DESCRICAO_CASAS[casa]["cor"],
        confianca=round(confianca, 4),
        probabilidades={
            nome: round(float(valor), 4)
            for nome, valor in zip(CLASSES, probabilidades)
        },
        decisao_apertada=confianca < LIMITE_CONFIANCA,
        justificativa=justificar(vetor, casa),
        recomendacao=recomendar(casa, confianca),
    )


@app.get("/api/v1/saude")
def saude():
    """Status do servico e metadados do modelo carregado."""
    return {
        "status": "ok",
        "versao_modelo": artefato["versao"],
        "treinado_em": artefato["treinado_em"],
        "casas": CLASSES,
        "atributos": FEATURES,
        "limite_confianca": LIMITE_CONFIANCA,
        "metricas": artefato["metricas"],
    }


@app.post("/api/v1/selecionar", response_model=ResultadoSelecao)
def selecionar_aluno(ficha: FichaDoAluno):
    """Cerimonia de selecao de um aluno."""
    try:
        return selecionar(ficha)
    except Exception as erro:
        raise HTTPException(status_code=500, detail=f"Erro na inferencia: {erro}")


@app.post("/api/v1/selecionar-turma", response_model=ResumoDaTurma)
def selecionar_turma(turma: TurmaDeIngresso):
    """Cerimonia em lote, para a secretaria processar a turma inteira de uma vez.

    Alem dos resultados individuais devolve a contagem por casa, que e o que a
    coordenacao usa pra saber se algum dormitorio vai estourar a capacidade.
    """
    try:
        resultados = [selecionar(ficha) for ficha in turma.alunos]
    except Exception as erro:
        raise HTTPException(status_code=500, detail=f"Erro na inferencia: {erro}")

    por_casa = {casa: 0 for casa in CLASSES}
    for resultado in resultados:
        por_casa[resultado.casa] += 1

    return ResumoDaTurma(
        total=len(resultados),
        por_casa=por_casa,
        decisoes_apertadas=sum(r.decisao_apertada for r in resultados),
        resultados=resultados,
    )
