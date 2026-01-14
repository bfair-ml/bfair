import pandas as pd
from pathlib import Path
from typing import Sequence, Tuple, Set, Union, Dict
from collections import defaultdict
from abc import ABC, abstractmethod


class IGroupWords(ABC):
    @abstractmethod
    def items(self) -> Sequence[Tuple[str, Union["IGroupWords", Set]]]:
        pass

    @abstractmethod
    def includes(self, word: str, key: str = None) -> bool:
        pass

    @abstractmethod
    def __getitem__(self, key: str) -> Union["IGroupWords", Set]:
        pass

    @abstractmethod
    def groups(self):
        pass

    def advance_context(self):
        """
        By default, do nothing.
        """
        pass

    def reset_context(self):
        """
        By default, do nothing.
        """
        pass


class FalseGroupWords(IGroupWords):
    def items(self) -> Sequence[Tuple[str, Union[IGroupWords, Set]]]:
        return ()

    def includes(self, word: str, key: str = None) -> bool:
        return False

    def __getitem__(self, key: str) -> IGroupWords:
        return FalseGroupWords()

    def groups(self):
        return ()


class GroupWords(IGroupWords):
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


class WordsByPos(IGroupWords):
    def __init__(self) -> None:
        self.pos2words = defaultdict(set)

    def add(self, word: str, pos: str) -> None:
        self.pos2words[pos].add(word)

    def includes(self, word: str, pos: str = None) -> bool:
        return (
            word in self.pos2words[pos]
            if pos is not None
            else any(word in items for items in self.pos2words.values())
        )

    def items(self) -> Sequence[Tuple[str, set]]:
        return self.pos2words.items()

    def __getitem__(self, pos) -> set:
        return self.pos2words[pos]

    def groups(self):
        return self.pos2words.keys()


class WordsByAny(IGroupWords):
    def __init__(self) -> None:
        self.words = set()

    def add(self, word: str) -> None:
        self.words.add(word)

    def items(self) -> Sequence[Tuple[str, Union[IGroupWords, Set]]]:
        yield None, self.words

    def includes(self, word: str, key: str = None) -> bool:
        return word in self.words

    def __getitem__(self, key: str) -> Union[IGroupWords, Set]:
        return self.words

    def groups(self):
        return ()


class DynamicGroupWords(IGroupWords):
    def __init__(
        self,
        texts: Sequence[str],
        annotations: Sequence[Tuple[str, bool, str]],
        group_order: Sequence[str],
        check_groups: bool = True,
    ) -> None:

        self.texts = texts
        self.annotations = annotations
        self.list_of_group_words = self._initilize_group_words(annotations)
        self.current = 0

        all_groups = {
            group
            for group_words in self.list_of_group_words
            for group in group_words.keys()
        }

        self.group_order = {group: None for group in group_order}.keys()  # ordered set
        if all_groups != self.group_order:
            msg = f"Group order does not match groups in annotations. {all_groups} vs {group_order}"
            msg += f"\nAnnotations: {[str(item) for item in self.annotations]}"
            if check_groups:
                raise ValueError(msg)
            else:
                print(f"⚠️ [JUST-WARNING due to `check_groups=False`] {msg}")

    @property
    def group_words(self) -> Dict[str, WordsByAny]:
        return self.list_of_group_words[self.current]

    def groups(self):
        return self.group_order

    @classmethod
    def _initilize_group_words(cls, annotations) -> Dict[str, WordsByAny]:
        list_of_group_words = []
        for annotation in annotations:
            group_words = defaultdict(WordsByAny)
            for word, include, group in annotation:
                if not include:
                    continue
                if " " in word:
                    print(
                        f"⚠️ Word contains space but current tokenization strategies do not support multi-word tokens."
                    )
                group_words[group].add(word)
            list_of_group_words.append(group_words)
        return list_of_group_words

    def items(self) -> Sequence[Tuple[str, WordsByAny]]:
        return self.group_words.items()

    def includes(self, word: str, key: str = None) -> bool:
        list_of_group_words = (
            [self.group_words]
            if self.current < len(self.texts)
            else self.list_of_group_words
        )
        return any(
            words.includes(word)
            for group_words in list_of_group_words
            for _, words in group_words.items()
        )

    def __getitem__(self, group: str) -> WordsByAny:
        return self.group_words[group]

    def advance_context(self):
        self.current += 1

    def reset_context(self):
        self.current = 0


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
        heteronyms="05_Heteronimos",
        adjectives="06_Heteronimos_adj",
        abbreviations="07_Abreviaturas",
        singular_suffix="",
        plural_suffix="_pl",
        *,
        include_professions=True,
        include_plurals=False,
    ):
        self.gender_order = [MALE, FEMALE]

        gender_info = [
            (male_dir, male_suffix),
            (female_dir, female_suffix),
        ]
        number_suffixes = (
            (singular_suffix, plural_suffix) if include_plurals else (singular_suffix,)
        )

        self.ignore = []
        self.pronouns = []
        self.nouns = []
        self.professions = []
        self.names = []
        self.heteronyms = []
        self.adjectives = []
        self.abbreviations = []

        for collection, category in [
            (self.ignore, ignore),
            (self.pronouns, pronouns),
            (self.nouns, nouns),
            (self.professions, professions),
            (self.names, names),
            (self.heteronyms, heteronyms),
            (self.adjectives, adjectives),
            (self.abbreviations, abbreviations),
        ]:
            for gender, suffix in gender_info:
                info = set()
                for number in number_suffixes:
                    try:
                        partial_info = self.read(
                            path, gender, suffix + number, category
                        )
                    except FileNotFoundError:
                        print(
                            f"⚠️ Number ({number}) not found for: ({category}, {gender})"
                        )
                        continue
                    info.update(partial_info)
                collection.append(info)

        self.exclude(self.professions, self.ignore)
        self.exclude(self.pronouns, self.ignore)
        self.exclude(self.nouns, self.ignore, self.professions)
        self.exclude(self.names, self.ignore)
        self.exclude(self.heteronyms, self.ignore)
        self.exclude(self.adjectives, self.ignore)
        self.exclude(self.abbreviations, self.ignore)

        self.include_professions = include_professions

    @staticmethod
    def exclude(collection, *exclusion_sources):
        for to_exclude in exclusion_sources:
            if len(collection) != len(to_exclude):
                raise ValueError("Collections do not have the same size.")
            for i in range(len(collection)):
                collection[i] = collection[i] - to_exclude[i]

    def get_group_words(self):
        group_words = GroupWords()

        noun_collections = [
            self.nouns,
            self.heteronyms,
        ]

        if self.include_professions:
            noun_collections.append(self.professions)

        for i, gender in enumerate(self.gender_order):
            for collection in noun_collections:
                group_words.update(gender, collection[i], "NOUN")

            for collection in [
                self.names,
                self.abbreviations,
            ]:
                group_words.update(gender, collection[i], "PROPN")

            group_words.update(gender, self.pronouns[i], "PRON")
            group_words.update(gender, self.adjectives[i], "ADJ")

        return group_words

    def get_professions_as_group_words(self):
        output = GroupWords()
        for i, gender in enumerate(self.gender_order):
            output.update(gender, self.professions[i], "NOUN")
        return output

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
