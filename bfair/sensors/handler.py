from typing import Sequence, List
from bfair.sensors import Sensor
from autogoal.kb import SemanticType


class SensorHandler:
    def __init__(self, sensors: Sequence[Sensor], merge=None):
        self.sensors = sensors
        self.merge = merge if merge else UnionMerge()

    def annotate(self, data, stype, attributes):
        sensors = self.select(stype)
        annotations = [sensor(data, attributes) for sensor in sensors]
        final = self.merge(annotations)
        return final

    def select(self, stype: SemanticType) -> List[Sensor]:
        return [sensor for sensor in self.sensors if sensor.match(stype)]


class UnionMerge:
    def __call__(self, annotations):
        return set([attr for attributes in annotations for attr in attributes])


class IntersectionMerge:
    def __call__(self, annotations):
        return set.intersection(*(set(attributes) for attributes in annotations))
