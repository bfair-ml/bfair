import pandas as pd
from pathlib import Path

from .base import Dataset

SENTENCE = "Sentence"
ANALYSIS = "Analysis"

MALE_VALUE = "Male"
FEMALE_VALUE = "Female"
GENDER_VALUES = [MALE_VALUE, FEMALE_VALUE]


def load_dataset(**kargs):
    return Ilenia.load(**kargs)


def parse_analysis(analysis):
    letter2person = {
        "S": True,
        "N": False,
    }

    letter2gender = {
        "M": MALE_VALUE,
        "F": FEMALE_VALUE,
    }
    output = []
    for per_word in analysis.splitlines():
        try:
            word, annotation = per_word.split(" - ")
            is_person, gender = annotation.split(",")
        except ValueError:
            print(f"⚠️ Error @ annotation: {per_word}.")
            continue

        word, is_person, gender = [x.strip() for x in (word, is_person, gender)]
        try:
            output.append((word, letter2person[is_person], letter2gender[gender]))
        except KeyError as e:
            print(f"⚠️ Unexpected token {e} @ {per_word}.")
    return output


class Ilenia(Dataset):
    @classmethod
    def load(
        cls,
        path="datasets/ilenia",
        split_seed=None,
        stratify_by=None,
    ):
        path = Path(path)
        global_data = pd.DataFrame(columns=[SENTENCE, ANALYSIS])
        documents = path.iterdir() if path.is_dir() else [path]
        for data_path in documents:
            data = pd.read_csv(data_path, sep=";", usecols=[SENTENCE, ANALYSIS])
            data[ANALYSIS] = data[ANALYSIS].apply(parse_analysis)
            global_data = pd.concat([global_data, data], axis=0)
        return Ilenia(data=global_data, split_seed=split_seed, stratify_by=stratify_by)

    @staticmethod
    def language():
        return "spanish"
