import re
import time
import json
import requests
from typing import List
from pathlib import Path

from autogoal.kb import SemanticType, Text
from bfair.sensors.base import Sensor, P_GENDER
from bfair.sensors.text.ner.names import NameGenderSensor


class TwitterNERSensor(Sensor):
    def __init__(
        self,
        name_sensor: NameGenderSensor,
        access_token: str,
        cache_path: str = None,
    ):
        self.name_sensor = name_sensor
        self.access_token = access_token
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.cache = self.load_cache(self.cache_path)
        super().__init__(restricted_to=P_GENDER)

    def __call__(self, text, attributes: List[str], attr_cls: str):
        usernames = self.extract_entities(text)

        names = []
        for username in usernames:
            name = self.get_name_by_username(username)
            if name is None:
                continue
            names.append(name)
        self.dump_cache()

        labeled_entities = {}
        for name in names:
            entity = MockEntity(name)
            predicted = self.extract_attributes(entity, attributes, attr_cls)
            labeled_entities[entity] = predicted

        labels = {attr for attrs in labeled_entities.values() for attr in attrs}
        return labels

    def extract_entities(self, text):
        return re.findall(r"@(\w+)", text)

    def get_name_by_username(self, username):
        try:
            return self.cache[username]
        except KeyError:
            data = self.fetch_data_from_twitter(username)
            if data is None:
                return None
            name = data.get("name")
            self.cache[username] = name
            return name

    def fetch_data_from_twitter(self, username):
        url = f"https://api.twitter.com/2/users/by/username/{username}"
        headers = {"Authorization": f"Bearer {self.access_token}"}

        while True:
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                return response.json()["data"]
            elif response.status_code == 429:
                print("Rate limit exceeded. Waiting...")
                time.sleep(60)
            else:
                print(f"Error: {response.status_code} - {response.text}")
                return None

    def extract_attributes(self, entity, attributes: List[str], attr_cls: str):
        return self.name_sensor.extract_attributes(entity, attributes, attr_cls)

    @classmethod
    def load_cache(cls, cache_path):
        if cache_path is None or not cache_path.exists():
            return {}
        with open(cache_path, "r") as file:
            return json.load(file)

    def dump_cache(self):
        if self.cache_path is not None:
            with open(self.cache_path, "w") as file:
                json.dump(self.cache, file)

    def _get_input_type(self) -> SemanticType:
        return Text

    @classmethod
    def build(
        cls,
        *,
        access_token: str,
        cache_path: str = None,
        model=None,
        language="english",
        entity_labels=None,
        just_people=True,
        attention_step=0,
        aggregator=None,
        filter=None,
        threshold=None,
    ):

        name_sensor = NameGenderSensor.build(
            model=model,
            language=language,
            entity_labels=entity_labels,
            just_people=just_people,
            attention_step=attention_step,
            aggregator=aggregator,
            filter=filter,
            threshold=threshold,
        )

        return cls(name_sensor, access_token, cache_path)


class MockEntity:
    def __init__(self, text):
        self.text = text


class DummyNameGenderSensor:
    def extract_attributes(self, entity, attributes: List[str], attr_cls: str):
        print("Extracting attributes for", entity.text)
        return attributes
