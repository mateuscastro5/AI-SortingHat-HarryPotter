"""Engenharia de features do modelo de triagem cadastral.

Mora aqui porque tanto o treino (`treinar_triagem.py`) quanto a API precisam
montar a linha de entrada exatamente do mesmo jeito. Se essa logica ficasse
duplicada nos dois lugares, uma hora eles iam divergir e o modelo ia receber
uma coluna diferente da que foi treinada.

A derivacao esta explicada no notebook 03.
"""

import re

import pandas as pd

CASAS = ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]
PADRAO_CASA = re.compile("|".join(CASAS), re.I)

# Colunas categoricas, na ordem em que o ColumnTransformer espera
CATEGORICAS = [
    "casa_sobrenome",
    "blood_status",
    "eye_color",
    "hair_color",
    "skin_color",
    "gender",
    "species",
    "varinha_madeira",
    "varinha_nucleo",
    "boggart",
    "patronus",
    "animagus",
    "titles",
    "jobs",
    "marital_status",
    "nationality",
    "decada_nasc",
]

# Campos que viram um unico texto vetorizado por TF-IDF
CAMPOS_TEXTO = ["titles", "jobs", "family_members", "born", "wands",
                "romances", "boggart", "patronus"]

MADEIRAS = ["holly", "vine", "ash", "willow", "oak", "elder", "yew", "cherry",
            "mahogany", "hawthorn", "walnut", "cypress", "chestnut", "fir",
            "birch", "alder", "elm", "maple", "hornbeam", "rosewood", "ebony"]
NUCLEOS = ["phoenix", "dragon", "unicorn", "veela", "thestral", "troll"]

# Palavras que aparecem no fim de nomes genericos da wiki e nao sao sobrenome
GENERICOS = {"girl", "boy", "student", "prefect", "champion", "member", "witch",
             "wizard", "man", "woman", "child", "father", "mother", "son",
             "daughter", "sibling", "friend", "captain", "keeper", "seeker",
             "beater", "chaser", "professor", "teacher", "ghost", "portrait"}

VAZIO = "desconhecido"


def extrair_sobrenome(nome):
    """Sobrenome utilizavel, ou string vazia quando o nome nao serve.

    A wiki tem muito personagem sem nome proprio ("Unidentified 2010s Gryffindor
    Girl"). Pegar a ultima palavra desses nomes criaria familias falsas do tipo
    "girl", que juntam gente das quatro casas.
    """
    original = str(nome)
    if re.match(r"(?i)^\s*(unidentified|unnamed)", original):
        return ""
    if PADRAO_CASA.search(original):
        return ""

    limpo = re.sub(r"\(.*?\)", "", original).strip()
    partes = [parte for parte in limpo.split() if parte.isalpha()]
    if len(partes) < 2:
        return ""

    ultima = partes[-1]
    if not ultima[0].isupper() or ultima.lower() in GENERICOS:
        return ""
    return ultima.lower()


def procurar(texto, opcoes):
    """Acha a primeira opcao que aparece no texto livre (madeira, nucleo)."""
    minusculo = str(texto).lower()
    return next((opcao for opcao in opcoes if opcao in minusculo), VAZIO)


def limpar_vazamento(valor):
    """Apaga o nome das casas dos campos de texto.

    Sem isso o modelo le 'Head of Gryffindor House' em titles e acerta de graca.
    """
    texto = str(valor) if valor is not None else ""
    texto = PADRAO_CASA.sub(" ", texto)
    return re.sub(r"[\[\]'�]", " ", texto)


def decada(valor):
    """Extrai a decada de nascimento de um texto de data."""
    achado = re.search(r"(1[6-9]\d{2})", str(valor))
    return str(int(achado.group(1)) // 10 * 10) if achado else "-1"


def montar_linha(ficha, mapa_sobrenome):
    """Transforma a ficha cadastral de um aluno na linha que o modelo espera.

    `ficha` e um dicionario com os campos do formulario e `mapa_sobrenome` e o
    dicionario sobrenome -> casa da familia aprendido no treino.
    """
    linha = {}

    sobrenome = extrair_sobrenome(ficha.get("nome", ""))
    linha["casa_sobrenome"] = mapa_sobrenome.get(sobrenome, VAZIO) if sobrenome else VAZIO

    for coluna in ["blood_status", "eye_color", "hair_color", "skin_color",
                   "gender", "species", "animagus", "marital_status",
                   "nationality"]:
        valor = str(ficha.get(coluna) or "").strip()
        linha[coluna] = valor or VAZIO

    varinha = limpar_vazamento(ficha.get("wands", ""))
    linha["varinha_madeira"] = procurar(varinha, MADEIRAS)
    linha["varinha_nucleo"] = procurar(varinha, NUCLEOS)
    linha["decada_nasc"] = decada(ficha.get("born", ""))

    for coluna in ["boggart", "patronus", "titles", "jobs"]:
        valor = limpar_vazamento(ficha.get(coluna, "")).strip()
        linha[coluna] = valor or VAZIO

    partes_texto = [limpar_vazamento(ficha.get(coluna, "")) for coluna in CAMPOS_TEXTO]
    linha["texto"] = " ".join(partes_texto).lower()

    return pd.DataFrame([linha], columns=CATEGORICAS + ["texto"])
