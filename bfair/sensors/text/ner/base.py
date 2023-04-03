import spacy

from bfair.sensors.base import Sensor
from autogoal.kb import SemanticType, Text
from typing import List


class NERBasedSensor(Sensor):
    def __init__(self, model):
        self.model = model

    @classmethod
    def build(cls, *, model=None, language="english"):
        if model is None:
            model = spacy.load("xx_ent_wiki_sm")
        return cls(model)

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
        return document.ents

    def extract_attributes(self, entity, attributes: List[str], attr_cls: str):
        raise NotImplementedError()