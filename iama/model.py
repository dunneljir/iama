import re
import pickle
import joblib
import numpy as np
import pkg_resources

from typing import Optional, Tuple

from .features import (
    tok,
    get_feature_1,
    get_feature_2,
    get_feature_3,
    get_feature_4,
    get_feature_5,
    token_gender,
    token_age,
)


REGEX_STR = (
    r"[\[\(\{] *[a-zA-Z]* *\/*,*-* *[0-9][0-9]? *\/*,*-* *[a-zA-Z]* *[\]\)\}]"
)


def generate_features(title: str, index: str):
    """
    Given a string generate features necessary for prediction.
    """
    token = tok(title)[index]
    feature_1 = get_feature_1(title, index)
    feature_2 = get_feature_2(title, index)
    feature_3 = get_feature_3(feature_1, feature_2)
    feature_4 = get_feature_4(title, token)
    feature_5 = get_feature_5(token)
    features = {
        "feature_1": feature_1,
        "feature_2": feature_2,
        "feature_3": feature_3,
        "feature_4": feature_4,
        "feature_5": feature_5,
    }
    return features


class Iama:
    def __init__(self):
        self._model = None
        self._dv = None
        self.load()
        pass

    def predict(self, title: str) -> Optional[Tuple[int, str]]:
        """
        Given a title, predict the token (e.g. [27M]) that refers 
        to the post's author.
        """
        X = []
        tokens = []
        tok_title = tok(title)
        has_token = False
        for idx in range(len(tok_title)):
            if re.match(REGEX_STR, tok_title[idx]):
                has_token = True
                tokens.append(tok_title[idx])
                X.append(generate_features(title, idx))

        if not has_token:
            return None

        X_encoded = self._dv.transform(X)
        probs_one = self._model.predict_log_proba(X_encoded)[:, 1]
        idx_max = np.argmax(probs_one)
        answer = tokens[idx_max]
        return (token_age(answer), token_gender(answer))

    def load(self):
        with open(
            pkg_resources.resource_filename(__name__, "models/model.pkl"), "rb"
        ) as model:
            self._model = pickle.load(model)
        self._dv = joblib.load(
            pkg_resources.resource_filename(
                __name__, "models/dict_vectorizer.pkl"
            ),
        )
        return self

