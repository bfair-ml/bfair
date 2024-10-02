import spacy

from typing import Sequence
from abc import ABC, abstractmethod
from spacy.tokens import Token

try:
    from bfair.utils.spacy_trf_vecs import get_model_with_trf_vectors
except:
    print(
        "[BFAIR ⚠️]: Failed to load the semantic module.",
        "You can safely ignore this message if the module is not required for your current use.",
    )

    def get_model_with_trf_vectors(model_name):
        raise Exception("Attempting to use the semantic module, but it was not loaded correctly.")


def best_fit(categories: Sequence[Token], word: Token) -> Token:
    return max(categories, key=lambda x: word.similarity(x))


class ITokenChecker(ABC):
    @abstractmethod
    def check_token(self, word: Token) -> bool:
        pass


class DummyChecker(ITokenChecker):
    def check_token(self, word: Token) -> bool:
        return True


class BooleanChecker(ITokenChecker):
    def __init__(self, true_word: Token, false_word: Token):
        self.true = true_word
        self.false = false_word

    def check_token(self, word: Token) -> bool:
        return best_fit((self.true, self.false), word) == self.true


class PersonCheckerForSpanish(BooleanChecker):
    def __init__(self, nlp=None, *, model: str = None):
        if nlp is None:
            if model is None:
                nlp = get_model_with_trf_vectors("es_dep_news_trf")
            else:
                nlp = spacy.load(model)

        self.nlp = nlp

        true_word = nlp("sí es persona")[-1]
        false_word = nlp("no es persona")[-1]
        super().__init__(true_word, false_word)
