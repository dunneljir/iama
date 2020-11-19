import re
from typing import List

from nltk.tokenize import regexp_tokenize


def tok(sentence: str) -> List[str]:
    regex_str = r"[\[\(\{][a-zA-Z]* *\/*,*-* *[0-9][0-9]? *\/*,*-* *[a-zA-Z]*[\]\)\}]|\w+"
    return regexp_tokenize(sentence, regex_str)


def get_feature_1(sentence: str, index: int) -> str:
    """Token to the left."""
    tokens = tok(sentence)
    return tokens[index - 1].lower() if index >= 1 else ""


def get_feature_2(sentence: str, index: int) -> str:
    """Token to the right."""
    tokens = tok(sentence)
    return tokens[index + 1].lower() if index < len(tokens) - 1 else ""


def get_feature_3(token1: str, token2: str) -> str:
    """Conjunction of features 1 and 2."""
    if token1 == "":
        token1 = "..."
    if token2 == "":
        token2 = "..."
    return token1 + "^" + token2


def get_feature_4(sentence: str, token: str) -> int:
    """If gender token is the first token in title."""
    tokens = tok(sentence)
    return 1 if tokens[0] == token else 0


def get_feature_5(token: str) -> str:
    """Returns the alphabetic part of a token"""
    alphabetic = re.sub(r"[^A-Za-z]+", "", token.lower())
    return alphabetic if alphabetic != "" else "..."
