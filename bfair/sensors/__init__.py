from typing import List
from autogoal.kb import SemanticType


class Sensor:
    def __call__(self, item, attributes: List[str]):
        raise NotImplementedError()

    def match(self, stype: SemanticType):
        input_type = self._get_input_type()
        return issubclass(stype, input_type)
    
    def _get_input_type(self) -> SemanticType:
        raise NotImplementedError()
