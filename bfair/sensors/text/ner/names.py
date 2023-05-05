from typing import List

import datasets as db

from bfair.sensors import P_GENDER
from bfair.sensors.text.ner.base import NERBasedSensor
from bfair.sensors.text.embedding.aggregators import Aggregator, SumAggregator
from bfair.sensors.text.embedding.filters import BestScoreFilter

MALE_VALUE = "Male"
FEMALE_VALUE = "Female"

_GENDER_MAP = {
    0: MALE_VALUE,
    1: FEMALE_VALUE,
}


class NameGenderSensor(NERBasedSensor):
    dataset = None

    def __init__(
        self,
        model,
        aggregator: Aggregator = None,
        attention_step=0.75,
        entity_labels=None,
    ):
        self.dataset = self._load_dataset_of_names()
        self.aggregator = aggregator
        self.attention_step = attention_step
        super().__init__(model, entity_labels, restricted_to=P_GENDER)

    @classmethod
    def _load_dataset_of_names(cls):
        if cls.dataset is None:
            level = db.logging.get_verbosity()
            db.logging.set_verbosity_error()
            source = db.load_dataset("md_gender_bias", "name_genders", split="yob2018")
            db.logging.set_verbosity(level)
            cls.dataset = source.to_pandas()
        return cls.dataset

    def extract_attributes(self, entity, attributes: List[str], attr_cls: str):
        df = self.dataset
        properties = ["assigned_gender", "count"]

        attention = 1
        scored_tokens = []
        for name in entity.text.split():
            matches = df[df["name"] == name]
            scores = {
                gender: count for _, gender, count in matches[properties].itertuples()
            }
            total = sum(scores.values())
            normalized = [
                (_GENDER_MAP[gender], attention * count / total)
                for gender, count in scores.items()
            ]
            scored_tokens.append(normalized)
            attention *= self.attention_step

        selected = self.aggregator(scored_tokens)
        values = {value for value, _ in selected}

        return [attribute for attribute in attributes if attribute.title() in values]

    @classmethod
    def build(
        cls,
        *,
        model=None,
        language="english",
        entity_labels=None,
        just_people=True,
        attention_step=0.75,
        aggregator=None,
        filter=None,
        threshold=None,
    ):

        aggregator = (
            SumAggregator(
                attr_filter=(
                    BestScoreFilter(threshold=threshold or 0.75, zero_threshold=0)
                    if filter is None
                    else filter
                ),
            )
            if aggregator is None
            else aggregator
        )

        return super().build(
            model=model,
            language=language,
            entity_labels=entity_labels,
            just_people=just_people,
            aggregator=aggregator,
            attention_step=attention_step,
        )
