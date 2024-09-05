import pandas as pd
from pathlib import Path


class SpanishGenderedWords:
    def __init__(
        self,
        path="datasets/victoria/Seeds",
        male_dir="Masculine",
        female_dir="Feminine",
    ):
        path = Path(path)
        male_path = path / male_dir
        female_path = path / female_dir

        self.male = set()
        self.female = set()

        for collection, path in ((self.male, male_path), (self.female, female_path)):
            for file in path.iterdir():
                data = pd.read_csv(file, names=["words"], header=None)
                collection.update(data["words"].str.strip())

    def get_male_words(self):
        return self.male

    def get_female_words(self):
        return self.female
