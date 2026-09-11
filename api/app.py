"""API do sistema de selecao de Hogwarts.

Serve os dois modelos do projeto e a interface web da secretaria.

Modelo de decisao (tracos de personalidade, 99.5% de acuracia):
  POST /api/v1/selecionar        -> cerimonia de um aluno
  POST /api/v1/selecionar-turma  -> lote, para a turma inteira

Modelo de triagem (ficha cadastral, 41% de acuracia):
  POST /api/v1/triagem           -> sugestao antes da cerimonia
  GET  /api/v1/opcoes            -> valores validos dos campos do formulario

  GET  /api/v1/saude             -> status e metadados dos dois modelos

Rodar: uvicorn api.app:app --reload
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from features_triagem import montar_linha  # noqa: E402

ARTEFATO_DECISAO = RAIZ / "modelos" / "chapeu_seletor.pkl"
ARTEFATO_TRIAGEM = RAIZ / "modelos" / "triagem_cadastral.pkl"
WEB = RAIZ / "web"

DESCRICAO_CASAS = {
    "Gryffindor": {"nome_pt": "Grifinória",
                   "lema": "Coragem, ousadia e determinação",
                   "cor": "#7F0909", "mascote": "Leão"},
    "Hufflepuff": {"nome_pt": "Lufa-Lufa",
                   "lema": "Lealdade, trabalho duro e paciência",
                   "cor": "#D3A625", "mascote": "Texugo"},
    "Ravenclaw": {"nome_pt": "Corvinal",
                  "lema": "Inteligência, criatividade e sabedoria",
                  "cor": "#222F5B", "mascote": "Águia"},
    "Slytherin": {"nome_pt": "Sonserina",
                  "lema": "Ambição, astúcia e engenhosidade",
                  "cor": "#0D6217", "mascote": "Serpente"},
}

ATRIBUTOS_PT = {
    "Bravery": "coragem",
    "Intelligence": "inteligência",
    "Loyalty": "lealdade",
    "Ambition": "ambição",
    "Dark Arts Knowledge": "conhecimento em artes das trevas",
    "Quidditch Skills": "habilidade no quadribol",
    "Dueling Skills": "habilidade em duelos",
    "Creativity": "criatividade",
}


def carregar(caminho, nome):
    try:
        return joblib.load(caminho)
    except FileNotFoundError:
        raise RuntimeError(
            f"Modelo de {nome} nao encontrado em {caminho}. "
            f"Rode os scripts de treino antes de subir a API."
        )


decisao = carregar(ARTEFATO_DECISAO, "decisao")
triagem = carregar(ARTEFATO_TRIAGEM, "triagem")

modelo_decisao = decisao["modelo"]
FEATURES = decisao["features"]
CLASSES = decisao["classes"]
LIMITE_DECISAO = decisao["limite_confianca"]

modelo_triagem = triagem["modelo"]
MAPA_SOBRENOME = triagem["mapa_sobrenome"]
CLASSES_TRIAGEM = triagem["classes"]
LIMITE_TRIAGEM = triagem["limite_confianca"]

app = FastAPI(
    title="Sistema de Selecao de Hogwarts",
    description="Chapeu Seletor digital: triagem cadastral e cerimonia de selecao.",
    version=decisao["versao"],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Modelo de decisao: tracos de personalidade
# --------------------------------------------------------------------------

class FichaDoAluno(BaseModel):
    """Notas de 0 a 10 preenchidas pela comissao de ingresso."""

    nome: str = Field("Aluno sem nome", max_length=80, examples=["Harry Potter"])
    bravery: int = Field(..., ge=0, le=10, examples=[10], description="Coragem")
    intelligence: int = Field(..., ge=0, le=10, examples=[6], description="Inteligência")
    loyalty: int = Field(..., ge=0, le=10, examples=[8], description="Lealdade")
    ambition: int = Field(..., ge=0, le=10, examples=[4], description="Ambição")
    dark_arts_knowledge: int = Field(..., ge=0, le=10, examples=[3])
    quidditch_skills: int = Field(..., ge=0, le=10, examples=[10])
    dueling_skills: int = Field(..., ge=0, le=10, examples=[9])
    creativity: int = Field(..., ge=0, le=10, examples=[5])

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
    mascote: str
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
    StandardScaler, a contribuicao de cada atributo e coeficiente * valor
    padronizado.

    Atencao ao sinal: um atributo tambem pesa a favor da casa quando esta
    *abaixo* da media e o coeficiente e negativo (e o caso da Lufa-Lufa com
    habilidade em duelos). Por isso a frase olha o valor padronizado, nao so a
    contribuicao.
    """
    scaler = modelo_decisao.named_steps["scaler"]
    classificador = modelo_decisao.named_steps["clf"]

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
        f"{'acima' if valor > 0 else 'abaixo'} da média dos candidatos"
        for feature, peso, valor in contribuicoes[:3]
        if peso > 0
    ]


def recomendar(casa, confianca):
    """Regra de negocio: o que a secretaria faz com o resultado."""
    if confianca < LIMITE_DECISAO:
        return (
            "Decisão apertada. Encaminhar o aluno para entrevista com a "
            "coordenação antes de confirmar a casa — o chapéu ficou em dúvida "
            "entre mais de uma opção."
        )
    return (
        f"Casa confirmada. Emitir a carta de boas-vindas da "
        f"{DESCRICAO_CASAS[casa]['nome_pt']}, alocar o dormitório e avisar o "
        f"monitor responsável."
    )


def selecionar(ficha: FichaDoAluno) -> ResultadoSelecao:
    vetor = ficha.para_vetor()
    entrada = pd.DataFrame([vetor], columns=FEATURES)

    probabilidades = modelo_decisao.predict_proba(entrada)[0]
    indice = int(np.argmax(probabilidades))
    casa = CLASSES[indice]
    confianca = float(probabilidades[indice])
    info = DESCRICAO_CASAS[casa]

    return ResultadoSelecao(
        nome=ficha.nome,
        casa=casa,
        casa_pt=info["nome_pt"],
        lema=info["lema"],
        cor=info["cor"],
        mascote=info["mascote"],
        confianca=round(confianca, 4),
        probabilidades={nome: round(float(valor), 4)
                        for nome, valor in zip(CLASSES, probabilidades)},
        decisao_apertada=confianca < LIMITE_DECISAO,
        justificativa=justificar(vetor, casa),
        recomendacao=recomendar(casa, confianca),
    )


@app.post("/api/v1/selecionar", response_model=ResultadoSelecao)
def selecionar_aluno(ficha: FichaDoAluno):
    """Cerimonia de selecao de um aluno, a partir da avaliacao de tracos."""
    try:
        return selecionar(ficha)
    except Exception as erro:
        raise HTTPException(status_code=500, detail=f"Erro na inferência: {erro}")


@app.post("/api/v1/selecionar-turma", response_model=ResumoDaTurma)
def selecionar_turma(turma: TurmaDeIngresso):
    """Cerimonia em lote, para a secretaria processar a turma inteira.

    Alem dos resultados individuais devolve a contagem por casa, que e o que a
    coordenacao usa pra saber se algum dormitorio vai estourar a capacidade.
    """
    try:
        resultados = [selecionar(ficha) for ficha in turma.alunos]
    except Exception as erro:
        raise HTTPException(status_code=500, detail=f"Erro na inferência: {erro}")

    por_casa = {casa: 0 for casa in CLASSES}
    for resultado in resultados:
        por_casa[resultado.casa] += 1

    return ResumoDaTurma(
        total=len(resultados),
        por_casa=por_casa,
        decisoes_apertadas=sum(r.decisao_apertada for r in resultados),
        resultados=resultados,
    )


# --------------------------------------------------------------------------
# Modelo de triagem: ficha cadastral
# --------------------------------------------------------------------------

class FichaCadastral(BaseModel):
    """Dados que a escola ja tem do aluno antes da cerimonia.

    Todos os campos exceto o nome sao opcionais - a ficha de um aluno novo
    costuma vir cheia de lacuna, e o modelo foi treinado com 'desconhecido'.
    """

    nome: str = Field(..., max_length=120, examples=["Scorpius Malfoy"])
    blood_status: str = Field("", max_length=60)
    gender: str = Field("", max_length=40)
    species: str = Field("", max_length=60)
    nationality: str = Field("", max_length=60)
    eye_color: str = Field("", max_length=40)
    hair_color: str = Field("", max_length=40)
    skin_color: str = Field("", max_length=40)
    marital_status: str = Field("", max_length=40)
    born: str = Field("", max_length=80, examples=["2006"])
    wands: str = Field("", max_length=120, examples=["10\" hawthorn unicorn hair"])
    titles: str = Field("", max_length=200)
    jobs: str = Field("", max_length=200)
    boggart: str = Field("", max_length=120)
    patronus: str = Field("", max_length=80)
    animagus: str = Field("", max_length=80)
    family_members: str = Field("", max_length=300)
    romances: str = Field("", max_length=200)


class ResultadoTriagem(BaseModel):
    nome: str
    casa_sugerida: str
    casa_pt: str
    cor: str
    confianca: float
    probabilidades: dict[str, float]
    sugestao_confiavel: bool
    casa_da_familia: str | None
    acuracia_do_modelo: float
    aviso: str


@app.post("/api/v1/triagem", response_model=ResultadoTriagem)
def triar_aluno(ficha: FichaCadastral):
    """Sugere uma casa a partir da ficha cadastral, antes da cerimonia.

    Esse modelo acerta cerca de 41% (contra 31% de chutar Grifinoria), entao a
    resposta e explicitamente uma **sugestao**. Quem decide a casa e a
    cerimonia, com o modelo de tracos.
    """
    try:
        entrada = montar_linha(ficha.model_dump(), MAPA_SOBRENOME)
        probabilidades = modelo_triagem.predict_proba(entrada)[0]
    except Exception as erro:
        raise HTTPException(status_code=500, detail=f"Erro na inferência: {erro}")

    indice = int(np.argmax(probabilidades))
    casa = CLASSES_TRIAGEM[indice]
    confianca = float(probabilidades[indice])

    casa_familia = entrada["casa_sobrenome"].iat[0]
    if casa_familia == "desconhecido":
        casa_familia = None

    return ResultadoTriagem(
        nome=ficha.nome,
        casa_sugerida=casa,
        casa_pt=DESCRICAO_CASAS[casa]["nome_pt"],
        cor=DESCRICAO_CASAS[casa]["cor"],
        confianca=round(confianca, 4),
        probabilidades={nome: round(float(valor), 4)
                        for nome, valor in zip(CLASSES_TRIAGEM, probabilidades)},
        sugestao_confiavel=confianca >= LIMITE_TRIAGEM,
        casa_da_familia=casa_familia,
        acuracia_do_modelo=triagem["metricas"]["acuracia_teste"],
        aviso=(
            "Triagem estatística com acurácia de "
            f"{triagem['metricas']['acuracia_teste'] * 100:.0f}%. "
            "Serve para priorizar atendimento, nunca para definir a casa — "
            "a casa é definida na cerimônia de seleção."
        ),
    )


@app.get("/api/v1/opcoes")
def opcoes_do_formulario():
    """Valores que o modelo de triagem viu no treino, para montar o formulario."""
    return triagem["opcoes"]


# --------------------------------------------------------------------------

@app.get("/api/v1/saude")
def saude():
    """Status do servico e metadados dos dois modelos."""
    return {
        "status": "ok",
        "casas": [
            {"chave": casa, **DESCRICAO_CASAS[casa]} for casa in CLASSES
        ],
        "modelo_decisao": {
            "versao": decisao["versao"],
            "treinado_em": decisao["treinado_em"],
            "atributos": FEATURES,
            "limite_confianca": LIMITE_DECISAO,
            "metricas": decisao["metricas"],
        },
        "modelo_triagem": {
            "versao": triagem["versao"],
            "treinado_em": triagem["treinado_em"],
            "limite_confianca": LIMITE_TRIAGEM,
            "sobrenomes_conhecidos": len(MAPA_SOBRENOME),
            "metricas": triagem["metricas"],
        },
    }


# Interface da secretaria. Fica por ultimo para nao capturar as rotas da API.
if WEB.exists():
    @app.get("/", include_in_schema=False)
    def pagina_inicial():
        return FileResponse(WEB / "index.html")

    app.mount("/", StaticFiles(directory=WEB), name="web")
