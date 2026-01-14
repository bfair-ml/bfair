import re
import pandas as pd
from pathlib import Path
import random

from pandas import DataFrame

from .base import Dataset
from bfair.envs import RHOPA64_DATASET

SEED = 37

GROUP_ID = "group"
SUBTHEME_ID = "id"
THEME = "theme"
SUBTHEME = "subtheme"
LANG = "lang"
EXPECTED = "expected"
PROMPT_SPECIFIC = "prompt_specific"
OUTPUT = "output"
ANNOTATIONS = "noun_count"

NEUTRAL_TAGS = {"N", "Ｎ"}
MALE_TAGS = {"M", "Ｍ"} 
FEMALE_TAGS = {"F", "Ｆ"}

MALE_VALUE = "male"
FEMALE_VALUE = "female"
GENDER_VALUES = [MALE_VALUE, FEMALE_VALUE]

LETTER2GENDER = {}
for tag in MALE_TAGS:
    LETTER2GENDER[tag] = MALE_VALUE
for tag in FEMALE_TAGS:
    LETTER2GENDER[tag] = FEMALE_VALUE

def load_dataset(**kargs):
    return RhoPa64.load(**kargs)

def parse_encoded(encoded):
    parsed = eval(encoded)
    if len(parsed) != 4:
        raise ValueError(f"⚠️ Wrong number of stories. Expected 4, got {len(parsed)}.")
    
    stories = []
    for item in parsed:
        if not item.startswith("Story "):
            raise ValueError(f"⚠️ Story does not start with 'Story N: ': {item}")
        story = item[len("Story N: "):]
        stories.append(story.strip())

    return stories
    
def parse_output(encoded):
    return parse_encoded(encoded)
    
def parse_annotations(encoded):
    items = parse_encoded(encoded)
    output = []
    for story_anns in items:
        annotations = []
        for ann in story_anns.splitlines():
            try:
                match = re.search(r"(\s?[-‑–=]\s)|(\s[-‑–=]\s?)", ann)
                if match is None:
                    raise ValueError(f"Annotation does not contain valid separator")
                else:
                    word, tag = ann.split(match.group(0))
            except ValueError as e:
                print(f"⚠️ Error [{e}] @ annotation: {ann}.")
                continue
            tag = tag.strip().upper()

            word = word.strip()
            if tag in NEUTRAL_TAGS:
                include = False
                group = None
            elif tag in LETTER2GENDER:
                include = True
                group = LETTER2GENDER[tag]
            else:
                print(f"⚠️ Unknown tag '{tag}' in annotation: {word}. Skipping.")
                continue

            annotations.append((word, include, group))
        output.append(annotations)
    return output


class RhoPa64(Dataset):
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
        model=None,
        theme_id=None,
        subtheme_id=None,
    ):
        self.language = language
        self.model = model
        self.theme_id = theme_id
        self.subtheme_id = subtheme_id
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
        path=RHOPA64_DATASET,
        language=None,
        model=None,
        theme_id=None,
        subtheme_id=None,
        annotated=True,
        split_seed=None,
        stratify_by=None,
    ):
        random.seed(SEED)
        annotated = annotated in ["yes", "True", True]

        path = str(Path(path) / model) + ".tsv"
        data = pd.read_csv(path, sep="\t", dtype={GROUP_ID: str, SUBTHEME_ID: str})

        if language is not None:
            key = cls.LANGUAGE2KEY.get(language.lower(), language.lower())
            data = data[data[LANG] == key]

        if theme_id is not None:
            data = data[data[GROUP_ID] == theme_id]

        if subtheme_id is not None:
            data = data[data[SUBTHEME_ID] == subtheme_id]


        data[OUTPUT] = data[OUTPUT].apply(parse_output)
        if annotated:
            data[ANNOTATIONS] = data[ANNOTATIONS].apply(parse_annotations)
            
            def sample_annotations_and_outputs(row):
                assert len(row[ANNOTATIONS]) == len(row[OUTPUT])
                pairs = [ (ann, out) for ann, out in zip(row[ANNOTATIONS], row[OUTPUT]) if len(ann) > 0 ]
                if len(pairs) > 2:
                    pairs = random.sample(pairs, 2)
                elif len(pairs) < 2:
                    print(f"🔴 Not enough annotation sets: {row}")
                row[ANNOTATIONS] = [ ann for ann, _ in pairs ]
                row[OUTPUT] = [ out for _, out in pairs ]
                return row

            data = data.apply(sample_annotations_and_outputs, axis=1)
        
        return RhoPa64(
            data=data,
            split_seed=split_seed,
            stratify_by=stratify_by,
            language=language,
            theme_id=theme_id,
            subtheme_id=subtheme_id,
            annotated=annotated,
        )
