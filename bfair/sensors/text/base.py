import spacy as sp

from abc import ABC, abstractmethod
from typing import Sequence, Callable, Tuple

from bfair.utils.spacy import get_model
from bfair.metrics.lm.words import IGroupWords, FalseGroupWords
from bfair.metrics.lm.endings import spanish_split_gender_endings


class IAnnotatedToken(ABC):
    @property
    @abstractmethod
    def text(self):
        pass

    @property
    @abstractmethod
    def pos_(self):
        pass

    @property
    @abstractmethod
    def lemma_(self):
        pass

    @property
    @abstractmethod
    def lower_(self):
        pass

    @property
    @abstractmethod
    def is_space(self):
        pass


class IPosTokenizer(ABC):
    @abstractmethod
    def __call__(self, text: str) -> Sequence[Tuple[str, str, IAnnotatedToken]]:
        pass


class SpacyPosTokenizer(IPosTokenizer):  # TODO: integrate @ bscore.
    LANGUAGE2ENDINGS = {"spanish": spanish_split_gender_endings}

    def __init__(
        self,
        *preprocessings: Sequence[Callable[[str], str]],
        nlp=None,
        model_name=None,
        language=None,
        inmutable_words: IGroupWords = FalseGroupWords(),
        lower_proper_nouns=True,
        lemmatize=True,
    ) -> None:

        self.nlp = (
            nlp
            if nlp is not None
            else get_model(
                language=language,
                model_name=model_name,
                add_transformer_vectors=False,
            )
        )
        self.language = language
        self.inmutable_words = inmutable_words
        self.lower_proper_nouns = lower_proper_nouns
        self.lemmatize = lemmatize
        self.preprocessings = preprocessings

    def __call__(self, text: str) -> Sequence[IAnnotatedToken]:
        for preprocessor in self.preprocessings:
            text = preprocessor(text)

        return [
            (
                # TEXT
                token.text
                if (
                    self.inmutable_words.includes(token.text, token.pos_)
                    or (token.pos_ == "PROPN" and not self.lower_proper_nouns)
                )
                else token.lower_
                if (
                    not self.lemmatize
                    or token.pos_ in ("PRON", "DET")
                    or self.inmutable_words.includes(token.lower_, token.pos_)
                    or self.inmutable_words.includes(token.lemma_.lower(), token.pos_)
                )
                else token.lemma_.lower(),
                # POS
                token.pos_,
                # TOKEN
                token,
            )
            for token in self.nlp(text)
            if not token.is_space
        ]

    @classmethod
    def get_preprocessors(cls, language, try_to_split_endings=True):
        preprocessors = []

        # ENDINGS
        if try_to_split_endings and language in cls.LANGUAGE2ENDINGS:
            item = cls.LANGUAGE2ENDINGS[language]
            preprocessors.append(item)

        # OTHERS
        # ...

        return preprocessors
