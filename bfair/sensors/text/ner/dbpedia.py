import requests
from urllib.parse import quote
from pathlib import Path
from functools import lru_cache
from difflib import SequenceMatcher

from typing import List, Dict, Any, Tuple
from bfair.sensors.text.ner.base import NERBasedSensor
from bfair.sensors.base import P_GENDER


class GenderStandarizer:
    MALE_KEYS = {
        "Male",
        "Feminine & Masculine",
        "M",
        "Female , Male",
        "Male ♂",
        "Male And Female",
        "Men'S",
        "Coeducational",
        "Man'S High School",
        "Co-Educational",
        "Males And Females",
        "Coeducation",
        "Male9",
        'Androgynous, But Can Choose To Be Male Or Female, Or "Both Or Neither"',
        "Simulated Male",
        "Male\nNickname = The Rigg",
        "Mixed-Sex_Education",
        "Female, Unisex",
        "Boys",
        "Males",
        "All-Male",
        "Mixed",
        "Boy",
        "All Boy",
        "Masculine",
        "Boys And Girls",
        "Co-Ed",
    }

    FEMALE_KEYS = {
        "Feminine & Masculine",
        "Female , Male",
        "Male And Female",
        "Coeducational",
        "Co-Educational",
        "Males And Females",
        "Coeducation",
        "Women",
        'Androgynous, But Can Choose To Be Male Or Female, Or "Both Or Neither"',
        "F",
        "Female",
        "Girls",
        "Mixed-Sex_Education",
        "Female, Unisex",
        "Mixed",
        "Girl",
        "Women'S",
        "W",
        "Boys And Girls",
        "Co-Ed",
    }

    IGNORE_KEYS = {
        "Director",
        "Unknown",
        "Single-Sex_Education",
        "D",
        "Dog",
    }

    MAP = {"male": MALE_KEYS, "female": FEMALE_KEYS}

    def __call__(self, value: str) -> Tuple[str]:
        return tuple(gender for gender, keys in self.MAP.items() if value in keys)


class DBPediaSensor(NERBasedSensor):
    DEFAULT_STANDARIZERS = {P_GENDER: GenderStandarizer()}

    def __init__(self, model, fuzzy_cutoff=0.6, **standarizers):
        self.dbpedia = FuzzyDBPediaWrapper(cutoff=fuzzy_cutoff)
        self.standarizers = dict(self.DEFAULT_STANDARIZERS, **standarizers)
        super().__init__(model)

    @classmethod
    def build(cls, *, model=None, language="english", fuzzy_cutoff=0.6, **standarizers):
        return super().build(
            model=model,
            language=language,
            fuzzy_cutoff=fuzzy_cutoff,
            **standarizers,
        )

    def extract_attributes(self, entity, attributes: List[str], attr_cls: str):
        entity_resource = self._entity_to_resource(entity)
        property_resource = self._property_to_resource(attr_cls)

        values = self.dbpedia.get_property_of(entity_resource, property_resource)

        standarizer = self.standarizers.get(attr_cls, self._standarize)
        standarized_values = set(s for v in values for s in standarizer(v))

        return [
            attribute
            for attribute in attributes
            for value in self._standarize(attribute)
            if value in standarized_values
        ]

    def _entity_to_resource(self, entity):
        resource = entity.text
        resource = resource.replace(" ", "_")
        return resource

    def _property_to_resource(self, property):
        resource = property.lower()
        return resource

    def _standarize(self, value: str) -> Tuple[str]:
        return (value.strip().lower(),)


class DBPediaWrapper:
    SPARQL_ENDPOINT = "https://dbpedia.org/sparql"
    DEFAULT_GRAPH_URI = "http://dbpedia.org"
    FORMAT = "json"
    DEFAULT_PARAMS = {"default-graph-uri": DEFAULT_GRAPH_URI, "format": FORMAT}
    LIMIT = 10000

    def __init__(self):
        pass

    def _get_params(self, **others):
        return dict(self.DEFAULT_PARAMS, **others)

    def _build_payload(self, params: Dict[str, Any]):
        return "&".join(f"{k}={quote(v, safe='+{}')}" for k, v in params.items())

    def _do_query(self, query, default=None):
        params = self._get_params(query=query)
        payload = self._build_payload(params)
        response = requests.get(self.SPARQL_ENDPOINT, params=payload)
        if response.ok:
            return response.json()["results"]
        if default is None:
            raise RuntimeError(response.content)

    def _get_values(self, results, *keys) -> List[Tuple]:
        bindings = results["bindings"]
        values = [tuple(row[key]["value"] for key in keys) for row in bindings]
        return values

    def _do_large_query_and_merge_values(self, query, key, *keys):
        limit = self.LIMIT
        values = []
        offset = 0
        while True:
            partial = self._do_query(
                query=query
                + f"""+
                ORDER+BY+ASC(?{key})+
                LIMIT+{limit}+
                OFFSET+{offset * limit}
                """,
                default=(),
            )
            if not partial:
                break

            partial_values = self._get_values(partial, key, *keys)
            if not partial_values:
                break

            values.extend(partial_values)
            offset += 1
        return values

    def _merge_at_index(self, values, *indexes, size=1):
        if not indexes:
            indexes = range(size)

        merged = []
        for row in values:
            value = tuple(self._default_transformation(row[i]) for i in indexes)
            if len(value) == 1:
                value = value[0]
            if value:
                merged.append(value)

        return merged

    def _default_transformation(self, value):
        return Path(value.title()).name

    def get_property_of(self, entity, property):
        values = self._do_large_query_and_merge_values(
            f"""
            SELECT+?{property}+
            WHERE+{{
                dbr:{entity}+dbp:{property}+?{property}+.
            }}
            """,
            property,
        )
        values = self._merge_at_index(values)
        return set(values)

    def get_people_with_property(self, property, include_property=False):
        var_name = "who"
        values = self._do_large_query_and_merge_values(
            f"""
            SELECT+?{var_name},+?{property}+
            WHERE+{{
                ?{var_name}+a+dbo:Person+.+
                ?{var_name}+dbp:{property}+?{property}+.
            }}
            """,
            var_name,
            property,
        )

        values = self._merge_at_index(values, size=(2 if include_property else 1))
        return set(values)

    def get_all_of_type(self, type_name):
        var_name = "who"
        values = self._do_large_query_and_merge_values(
            f"""
            SELECT+?{var_name}+
            WHERE+{{
                ?{var_name}+a+dbo:{type_name}+.
            }}""",
            var_name,
        )
        values = self._merge_at_index(values)
        return set(values)


class FuzzyDBPediaWrapper(DBPediaWrapper):
    def __init__(self, cutoff=0.6):
        self.cutoff = cutoff
        super().__init__()

    def get_property_of(self, entity, property):
        people_with_property = self._get_candidate_matches(property)
        matches = self._fuzzy_match(entity, people_with_property)
        return {value for _, value in matches if value}

    @lru_cache()
    def _get_candidate_matches(self, property):
        return self.get_people_with_property(property, include_property=True)

    def _fuzzy_match(self, entity, people_with_property):
        matches = set()

        s = SequenceMatcher()
        s.set_seq2(entity)
        for person, property in people_with_property:
            s.set_seq1(person)
            if (
                s.real_quick_ratio() > self.cutoff
                and s.quick_ratio() >= self.cutoff
                and s.ratio() >= self.cutoff
            ):
                matches.add((person, property))

        return matches
