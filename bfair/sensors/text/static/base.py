from typing import List, Set, Union
from collections import defaultdict

from autogoal.kb import SemanticType, Text
from bfair.sensors.base import Sensor, P_GENDER
from bfair.sensors.text.base import IPosTokenizer, SpacyPosTokenizer
from bfair.sensors.text.embedding.aggregators import Aggregator, UnionAggregator
from bfair.metrics.lm.words import EnglishGenderedWords, SpanishGenderedWords


class StaticWordsSensor(Sensor):
    FEATURES2WORDS = {
        (P_GENDER, "english"): lambda: EnglishGenderedWords().get_group_words(),
        (P_GENDER, "spanish"): lambda: SpanishGenderedWords().get_group_words(),
    }

    def __init__(
        self,
        language,
        aggregator: Aggregator,
        tokenizer: IPosTokenizer = None,
        restricted_to: Union[str, Set[str]] = None,
    ):
        self.tokenizer = tokenizer
        self.aggregator = aggregator
        self.language = language
        super().__init__(restricted_to)

    def __call__(self, text, attributes: List[str], attr_cls: str):
        group_words = self.get_group_words(attr_cls)
        tokenizer = (
            self.get_default_tokenizer(group_words, language=self.language)
            if self.tokenizer is None
            else self.tokenizer
        )

        scored_tokens = []
        for word_text, word_pos, word in tokenizer(text):
            scores = [
                (attribute, 1)
                for attribute in attributes
                if group_words[attribute].includes(word_text, word_pos)
            ]
            scored_tokens.append(scores)

        selected = self.aggregator(scored_tokens)
        values = {value for value, _ in selected}
        return [attribute for attribute in attributes if attribute.lower() in values]

    def _get_input_type(self) -> SemanticType:
        return Text

    def get_group_words(self, attr_cls):
        try:
            init = self.FEATURES2WORDS[attr_cls, self.language]
            return init()
        except KeyError:
            raise ValueError(
                f"Language/Attribute pair not supported ({self.language, attr_cls})"
            )

    @classmethod
    def get_default_tokenizer(
        cls,
        inmutable_words,
        model=None,
        model_name=None,
        language=None,
        lower_proper_nouns=True,
        lemmatize=True,
        try_to_split_endings=True,
    ):
        preprocessings = SpacyPosTokenizer.get_preprocessors(
            language=language,
            try_to_split_endings=try_to_split_endings,
        )
        return SpacyPosTokenizer(
            *preprocessings,
            inmutable_words=inmutable_words,
            nlp=model,
            model_name=model_name,
            language=language,
            lower_proper_nouns=lower_proper_nouns,
            lemmatize=lemmatize,
        )

    @classmethod
    def build(
        cls,
        *,
        language,
        tokenizer=None,
        model=None,
        model_name=None,
        inmutable_words=None,
        lower_proper_nouns=True,
        lemmatize=True,
        try_to_split_endings=True,
        aggregator=None,
        **kwargs,
    ):

        tokenizer = (
            tokenizer
            if tokenizer is not None
            else None
            if inmutable_words is None
            else cls.get_default_tokenizer(
                inmutable_words,
                nlp=model,
                model_name=model_name,
                language=language,
                lower_proper_nouns=lower_proper_nouns,
                lemmatize=lemmatize,
                try_to_split_endings=try_to_split_endings,
            )
        )
        aggregator = UnionAggregator() if aggregator is None else aggregator

        return cls(
            language,
            aggregator=aggregator,
            tokenizer=tokenizer,
            **kwargs,
        )
