import spacy

from bfair.sensors.base import Sensor
from autogoal.kb import SemanticType, Text
from typing import List, Set, Union


class NERBasedSensor(Sensor):
    LANGUAGE_TO_MODEL = {
        "english": "en_core_web_sm",
    }
    MODEL_TO_PERSON_LABEL = {
        "en_core_web_sm": "PERSON",
        "xx_ent_wiki_sm": "PER",
    }

    def __init__(
        self,
        model,
        entity_labels=None,
        restricted_to: Union[str, Set[str]] = None,
    ):
        self.model = model
        self.entity_labels = entity_labels
        super().__init__(restricted_to)

    @classmethod
    def build(
        cls,
        *,
        model=None,
        language="english",
        entity_labels=None,
        just_people=True,
        **kwargs,
    ):

        if model is None:
            model = cls.LANGUAGE_TO_MODEL.get(language, "xx_ent_wiki_sm")

        if isinstance(model, str):
            model = spacy.load(model)

        if entity_labels is None and just_people:
            model_name = f'{model.meta["lang"]}_{model.meta["name"]}'
            entity_labels = cls.MODEL_TO_PERSON_LABEL.get(model_name)

        return cls(model, entity_labels=entity_labels, **kwargs)

    def __call__(self, text, attributes: List[str], attr_cls: str):
        document = self.model(text)
        named_entities = self.extract_entities(document)

        labeled_entities = {}
        for entity in named_entities:
            predicted = self.extract_attributes(entity, attributes, attr_cls)
            labeled_entities[entity] = predicted

        labels = {attr for attrs in labeled_entities.values() for attr in attrs}
        return labels

    def _get_input_type(self) -> SemanticType:
        return Text

    def extract_entities(self, document):
        return [
            ent
            for ent in document.ents
            if self.entity_labels is None or ent.label_ in self.entity_labels
        ]

    def extract_attributes(self, entity, attributes: List[str], attr_cls: str):
        raise NotImplementedError()
