from typing import Sequence, List
from bfair.sensors.base import Sensor, P_ANY
from autogoal.kb import SemanticType


class SensorHandler:
    def __init__(self, sensors: Sequence[Sensor], merge=None):
        self.sensors = sensors
        self.merge = merge if merge else UnionMerge()

    def annotate(self, data, stype, attributes, attr_cls=P_ANY):
        sensors = self.select(stype, attr_cls)
        annotations = [sensor(data, attributes) for sensor in sensors]
        final = self.merge(annotations)
        return final

    def select(self, stype: SemanticType, attr_cls: str) -> List[Sensor]:
        return [sensor for sensor in self.sensors if sensor.match(stype, attr_cls)]


class UnionMerge:
    def __call__(self, annotations):
        return set([attr for attributes in annotations for attr in attributes])


class IntersectionMerge:
    def __call__(self, annotations):
        return set.intersection(*(set(attributes) for attributes in annotations))
