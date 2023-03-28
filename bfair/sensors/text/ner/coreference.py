import spacy
import neuralcoref

from typing import List
from bfair.sensors.base import Sensor, P_GENDER
from bfair.sensors.text.ner.base import NERBasedSensor
from bfair.sensors.text.embedding.aggregators import Aggregator, CountAggregator


class CoreferenceNERSensor(NERBasedSensor):
    LANGUAGE_TO_MODEL = {
        "english": "en_core_web_sm",
    }
    GENDER_PRONOUNS = {
        "female": ["she", "her", "hers"],
        "male": ["he", "him", "his"],
    }

    def __init__(self, model, aggregator: Aggregator):
        self.aggregator = aggregator
        super().__init__(model)

    @classmethod
    def build(
        cls,
        *,
        model=None,
        spacy_model_name=None,
        language="english",
        aggregator=None,
        filter=None,
        threshold=None,
    ):
        if model is None:
            if spacy_model_name is None:
                try:
                    spacy_model_name = cls.LANGUAGE_TO_MODEL[language]
                except KeyError:
                    raise ValueError(f'Language "{language}" not supported.')
            model = spacy.load(spacy_model_name)

        neuralcoref.add_to_pipe(model)

        if aggregator is not None and (filter is not None and threshold is not None):
            raise ValueError(
                "Only one between `aggregator`, `filter` and `threshold` should be provided."
            )

        aggregator = (
            CountAggregator(attr_filter=filter, threshold=threshold)
            if aggregator is None
            else aggregator
        )

        return cls(model, aggregator)

    def extract_attributes(self, entity, attributes: List[str], attr_cls: str):
        if not entity._.is_coref:
            return ()

        scored_tokens = []
        constant_score = 1 / len(entity._.coref_cluster)
        for coref in entity._.coref_cluster:
            coref = coref.text.lower()
            actives = [
                (attr, constant_score)
                for attr in attributes
                if coref in self.GENDER_PRONOUNS[attr.lower()]
            ]
            scored_tokens.append(actives)

        syntesis = self.aggregator(scored_tokens)[0]
        labels = [attr for attr, _ in syntesis]
        return labels

    def _extracts(self, attr_cls: str) -> bool:
        return attr_cls == P_GENDER
