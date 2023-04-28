from typing import List, Set, Union
from autogoal.kb import SemanticType

P_GENDER = "gender"
P_RACE = "race"
P_AGE = "age"
P_SEXUAL_ORIENTATION = "sexual orientation"
P_DISABILITY = "disability"
P_MARITAL_STATUS = "marital status"
P_PREGNANCY = "pregnancy"
P_RELIGION = "religion"
P_POLITICAL_OPINION = "political opinion"
P_NATIONAL_EXTRACTION = "national extraction"
P_ETHNIC = "ethnic"
P_BREASTFEEDING = "breastfeeding"


class Sensor:
    def __init__(self, restricted_to: Union[str, Set[str]] = None):
        self.restricted_to = restricted_to

    def __call__(self, item, attributes: List[str], attr_cls: str):
        raise NotImplementedError()

    def match(self, stype: SemanticType, attr_cls: str):
        input_type = self._get_input_type()
        return issubclass(stype, input_type) and self._extracts(attr_cls)

    def _get_input_type(self) -> SemanticType:
        raise NotImplementedError()

    def _extracts(self, attr_cls: str) -> bool:
        return (
            self.restricted_to is None
            or attr_cls == self.restricted_to
            or attr_cls in self.restricted_to
        )
