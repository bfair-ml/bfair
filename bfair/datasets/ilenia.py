import pandas as pd
from pathlib import Path

from pandas import DataFrame

from .base import Dataset
from bfair.envs import ILENIA_DATASET

SENTENCE = "Sentence"
ANALYSIS = "Analysis"

MALE_VALUE = "male"
FEMALE_VALUE = "female"
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
    LANGUAGE2KEY = {"english": "en", "spanish": "es", "valencian": "va"}
    KEY2LANGUAGE = {key: language for language, key in LANGUAGE2KEY.items()}

    def __init__(
        self,
        data: DataFrame,
        annotated: bool,
        *,
        validation: DataFrame = None,
        test: DataFrame = None,
        split_seed=None,
        stratify_by=None,
        language=None,
    ):
        self.language = language
        self.annotated = annotated
        super().__init__(
            data,
            validation=validation,
            test=test,
            split_seed=split_seed,
            stratify_by=stratify_by,
        )

    @classmethod
    def load(
        cls,
        path=ILENIA_DATASET,
        language=None,
        annotated=True,
        split_seed=None,
        stratify_by=None,
    ):
        annotated = annotated in ["yes", "True", True]

        path = Path(path)

        if language is not None and path.is_dir():
            key = cls.LANGUAGE2KEY[language]
            path = path / key
        elif language is None:
            key = path.name if path.is_dir() else path.parent.name
            try:
                language = cls.KEY2LANGUAGE[key]
            except KeyError:
                raise ValueError(
                    f"Unable to infer language from '{key}' ({path})"
                ) from None

        global_data = pd.DataFrame(columns=[SENTENCE, ANALYSIS])
        documents = path.iterdir() if path.is_dir() else [path]
        for data_path in documents:
            if annotated:
                data = pd.read_csv(data_path, sep=";", usecols=[SENTENCE, ANALYSIS])
                data[ANALYSIS] = data[ANALYSIS].apply(parse_analysis)
            else:
                data = pd.read_csv(
                    data_path, index_col=False, header=None, names=[SENTENCE]
                )
            global_data = pd.concat([global_data, data], axis=0)

        global_data.reset_index(drop=True, inplace=True)

        return Ilenia(
            data=global_data,
            split_seed=split_seed,
            stratify_by=stratify_by,
            language=language,
            annotated=annotated,
        )
