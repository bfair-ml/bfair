import random
from typing import List, Set, Union
from bfair.sensors.base import Sensor
from autogoal.kb import SemanticType


class MockSensor(Sensor):
    def _get_input_type(self) -> SemanticType:
        return SemanticType


class FixValueSensor(MockSensor):
    def __init__(self, value: str, restricted_to: Union[str, Set[str]] = None):
        self.value = value
        super().__init__(restricted_to)

    def __call__(self, item, attributes: List[str], attr_cls: str):
        return [self.value]


class RandomValueSensor(MockSensor):
    def __init__(
        self, seed=0, distribution=None, restricted_to: Union[str, Set[str]] = None
    ):
        super().__init__(restricted_to)
        self.rng = random.Random(seed)
        self.distribution = distribution

    def __call__(self, item, attributes: List[str], attr_cls: str):
        return [
            attr
            for attr in attributes
            if self.rng.random() > self._get_from_distribution(attr)
        ]

    def _get_from_distribution(self, attribute):
        if self.distribution is None:
            return 0.5
        try:
            return self.distribution[attribute]
        except KeyError:
            raise ValueError(f"Invalid distribution. Missing value {attribute}.")
