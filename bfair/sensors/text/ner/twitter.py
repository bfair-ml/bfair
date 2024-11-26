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
        retries=None,
        log=False,
    ):
        self.name_sensor = name_sensor
        self.access_token = access_token
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.cache = self.load_cache(self.cache_path)
        self.retries = retries
        self.logger = (
            (lambda *args: print(*args, flush=True)) if log else (lambda *args: None)
        )
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
            name = self.cache[username]
            self.logger(f"Cache hit: {username} -> {name}")
            return name
        except KeyError:
            self.logger(f"Cache miss: {username}. Fetching from Twitter ...")
            data, ok = self.fetch_data_from_twitter(username)
            self.logger(f"Fetch ended {'successfully' if ok else 'with error'}")

            if data is not None:
                name = self.cache[username] = data.get("name")
                self.logger(f"Cache updated: {username} -> {name}")
                return name if ok else None
            return None

    def fetch_data_from_twitter(self, username):
        url = f"https://api.twitter.com/2/users/by/username/{username}"
        headers = {"Authorization": f"Bearer {self.access_token}"}

        sleep_time = 60
        remaining_retries = self.retries

        while True:
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                content = response.json()
                try:
                    return content["data"], True
                except KeyError:
                    print(f"Error: {content}")
                    try:
                        error = content["errors"][0]
                        return {"name": error["title"]}, False
                    except (KeyError, IndexError):
                        return {"name": str(content)}, False
            elif response.status_code == 429 and (
                remaining_retries is None or remaining_retries > 0
            ):
                print("Rate limit exceeded. Waiting...")
                time.sleep(sleep_time)
                sleep_time *= 2
                remaining_retries -= 1
            else:
                print(f"Error: {response.status_code} - {response.text}")
                return None, False

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
