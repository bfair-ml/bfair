import spacy
import neuralcoref

from typing import List, Set
from bfair.sensors.base import Sensor, P_GENDER
from bfair.sensors.text.ner.base import NERBasedSensor
from bfair.sensors.text.embedding.aggregators import Aggregator, CountAggregator


class CoreferenceNERSensor(NERBasedSensor):
    LANGUAGE_TO_MODEL = {  # duplicated because `neuralcoref`` is more restrictive
        "english": "en_core_web_sm",
    }
    GENDER_PRONOUNS = {
        "female": ["she", "her", "hers"],
        "male": ["he", "him", "his"],
    }

    def __init__(self, model, aggregator: Aggregator, entity_labels=None):
        self.aggregator = aggregator
        super().__init__(model, entity_labels, restricted_to=P_GENDER)

    @classmethod
    def build(
        cls,
        *,
        model=None,
        language="english",
        entity_labels=None,
        just_people=True,
        aggregator=None,
        filter=None,
        threshold=None,
    ):
        if model is None:
            try:
                model = cls.LANGUAGE_TO_MODEL[language]
            except KeyError:
                raise ValueError(f'Language "{language}" not supported.')

        if isinstance(model, str):
            model = spacy.load(model)

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

        return super().build(
            model=model,
            language=language,
            entity_labels=entity_labels,
            just_people=just_people,
            aggregator=aggregator,
        )

    def extract_entities(self, document):
        return [group.main for group in document._.coref_clusters]

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

        syntesis = self.aggregator(scored_tokens)
        labels = [attr for attr, _ in syntesis]
        return labels
