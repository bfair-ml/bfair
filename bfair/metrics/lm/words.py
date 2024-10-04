import pandas as pd
from pathlib import Path
from typing import Sequence, Tuple
from collections import defaultdict


class GroupWords:
    def __init__(self) -> None:
        self.group_words = defaultdict(WordsByPos)
        self.all_available_pos = set()

    def add(self, group: str, word: str, pos: str):
        self.group_words[group][pos].add(word)
        self.all_available_pos.add(pos)

    def update(self, group: str, words: Sequence[str], pos: str):
        self.group_words[group][pos].update(words)
        self.all_available_pos.add(pos)

    def groups(self):
        return self.group_words.keys()

    def get_all_words(self):
        return {
            word
            for group, gwords in self.items()
            for pos, words in gwords.items()
            for word in words
        }

    def get_words_per_group(self):
        for group in self.groups():
            yield group, self.get_words(group)

    def get_words(self, group, pos_list=None):
        if pos_list is None:
            pos_list = self.all_available_pos
        return {
            word for pos in pos_list for word in self.words_with_pos_for(group, pos)
        }

    def get_words_with_pos_for(self, group, pos):
        return self.group_words[group][pos]

    def __getitem__(self, group) -> "WordsByPos":
        return self.group_words[group]

    def items(self) -> Sequence[Tuple[str, "WordsByPos"]]:
        return self.group_words.items()

    def includes(self, word: str, pos: str = None) -> bool:
        pos_list = self.all_available_pos if pos is None else [pos]
        return any(
            words.includes(word, pos) for _, words in self.items() for pos in pos_list
        )


class WordsByPos:
    def __init__(self) -> None:
        self.pos2words = defaultdict(set)

    def add(self, word: str, pos: str) -> None:
        self.pos2words[pos].add(word)

    def includes(self, word: str, pos: str) -> bool:
        return word in self.pos2words[pos]

    def items(self) -> Sequence[Tuple[str, set]]:
        return self.pos2words.items()

    def __getitem__(self, pos) -> set:
        return self.pos2words[pos]


MALE = "male"
FEMALE = "female"

GERDER_PAIR_ORDER = [MALE, FEMALE]

DM_GENDER_PAIRS = [
    ("NOUN", "actor", "actress"),
    ("NOUN", "boy", "girl"),
    ("NOUN", "boyfriend", "girlfriend"),
    ("NOUN", "boys", "girls"),
    ("NOUN", "father", "mother"),
    ("NOUN", "fathers", "mothers"),
    ("NOUN", "gentleman", "lady"),
    ("NOUN", "gentlemen", "ladies"),
    ("NOUN", "grandson", "granddaughter"),
    ("PRON", "he", "she"),
    ("PRON", "him", "her"),
    ("PRON", "his", "her"),
    ("NOUN", "husbands", "wives"),
    ("NOUN", "kings", "queens"),
    ("NOUN", "male", "female"),
    ("NOUN", "males", "females"),
    ("NOUN", "man", "woman"),
    ("NOUN", "men", "women"),
    ("NOUN", "prince", "princess"),
    ("NOUN", "son", "daughter"),
    ("NOUN", "sons", "daughters"),
    ("NOUN", "spokesman", "spokeswoman"),
    ("NOUN", "stepfather", "stepmother"),
    ("NOUN", "uncle", "aunt"),
    ("NOUN", "husband", "wife"),
    ("NOUN", "king", "queen"),
    ("NOUN", "brother", "sister"),
    ("NOUN", "brothers", "sisters"),
]

PENN_GENDER_PAIRS = [
    ("NOUN", "actor", "actress"),
    ("NOUN", "boy", "girl"),
    ("NOUN", "father", "mother"),
    ("PRON", "he", "she"),
    ("PRON", "him", "her"),
    ("PRON", "his", "her"),
    ("NOUN", "male", "female"),
    ("NOUN", "man", "woman"),
    ("NOUN", "men", "women"),
    ("NOUN", "son", "daughter"),
    ("NOUN", "sons", "daughters"),
    ("NOUN", "spokesman", "spokeswoman"),
    ("NOUN", "husband", "wife"),
    ("NOUN", "king", "queen"),
    ("NOUN", "brother", "sister"),
]

WIKI_GENDER_PAIRS = [
    ("NOUN", "actor", "actress"),
    ("NOUN", "boy", "girl"),
    ("NOUN", "boyfriend", "girlfriend"),
    ("NOUN", "boys", "girls"),
    ("NOUN", "father", "mother"),
    ("NOUN", "fathers", "mothers"),
    ("NOUN", "gentleman", "lady"),
    ("NOUN", "gentlemen", "ladies"),
    ("NOUN", "grandson", "granddaughter"),
    ("PRON", "he", "she"),
    ("NOUN", "hero", "heroine"),
    ("PRON", "him", "her"),
    ("PRON", "his", "her"),
    ("NOUN", "husband", "wife"),
    ("NOUN", "husbands", "wives"),
    ("NOUN", "king", "queen"),
    ("NOUN", "kings", "queens"),
    ("NOUN", "male", "female"),
    ("NOUN", "males", "females"),
    ("NOUN", "man", "woman"),
    ("NOUN", "men", "women"),
    ("PROPN", "mr.", "mrs."),
    ("PROPN", "mr", "mrs"),
    ("PROPN", "Mr.", "Mrs."),
    ("PROPN", "Mr", "Mrs"),
    ("NOUN", "prince", "princess"),
    ("NOUN", "son", "daughter"),
    ("NOUN", "sons", "daughters"),
    ("NOUN", "spokesman", "spokeswoman"),
    ("NOUN", "stepfather", "stepmother"),
    ("NOUN", "uncle", "aunt"),
    # ("NOUN", "wife", "husband"),
    ("NOUN", "king", "queen"),
]


class EnglishGenderedWords:
    def __init__(
        self,
        sources=(DM_GENDER_PAIRS, PENN_GENDER_PAIRS, WIKI_GENDER_PAIRS),
        order=GERDER_PAIR_ORDER,
    ):
        self.sources = sources
        self.order = order

    def get_group_words(self):
        group_words = GroupWords()
        for pairs in self.sources:
            for pos, *pair in pairs:
                for i, gender in enumerate(self.order):
                    group_words.add(gender, pair[i], pos)
        return group_words


class SpanishGenderedWords:
    MALE = 0
    FEMALE = 1

    def __init__(
        self,
        path=Path("datasets/victoria/Seeds"),
        male_dir="Masculine",
        female_dir="Feminine",
        male_suffix="masc",
        female_suffix="fem",
        ignore="00_Ignore",
        pronouns="01_Pron",
        nouns="02_Sust",
        professions="03_Professions_list",
        names="04_Nombres",
        basic="05_Heteronimos",
        adjectives="06_Heteronimos_adj",
        abbreviations="07_Abreviaturas",
    ):
        self.gender_order = [MALE, FEMALE]

        ignore_male = self.read(path, male_dir, male_suffix, ignore)
        ignore_female = self.read(path, female_dir, female_suffix, ignore)

        gender_info = [
            (male_dir, male_suffix, ignore_male),
            (female_dir, female_suffix, ignore_female),
        ]

        self.pronouns = []
        self.nouns = []
        self.professions = []
        self.names = []
        self.basic = []
        self.adjectives = []
        self.abbreviations = []

        for collection, category in [
            (self.pronouns, pronouns),
            (self.nouns, nouns),
            (self.professions, professions),
            (self.names, names),
            (self.basic, basic),
            (self.adjectives, adjectives),
            (self.abbreviations, abbreviations),
        ]:
            for gender, suffix, to_ignore in gender_info:
                info = self.read(path, gender, suffix, category, to_ignore)
                collection.append(info)

    def get_group_words(self):
        group_words = GroupWords()

        for i, gender in enumerate(self.gender_order):
            for collection in [
                self.pronouns,
                self.nouns,
                self.professions,
                self.basic,
            ]:
                group_words.update(gender, collection[i], "NOUN")

            for collection in [
                self.names,
                self.abbreviations,
            ]:
                group_words.update(gender, collection[i], "PROPN")

            group_words.update(gender, self.adjectives[i], "ADJ")

        return group_words

    @classmethod
    def read(
        cls,
        root: Path,
        gender: str,
        suffix: str,
        category: str,
        to_ignore: set = frozenset(),
    ):
        path = root / gender / f"{category}_{suffix}.csv"
        data = pd.read_csv(path, names=["words"], header=None)
        words = data["words"].str.strip()
        return set(words) - to_ignore
