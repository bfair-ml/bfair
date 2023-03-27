from bfair.sensors.base import Sensor, P_GENDER


class CoreferenceNERSensor(Sensor):
    def _extracts(self, attr_cls: str) -> bool:
        return attr_cls == P_GENDER
