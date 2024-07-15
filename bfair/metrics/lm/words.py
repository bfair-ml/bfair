import pandas as pd
from pathlib import Path

class SpanishGenderedWords:
    def __init__(self, path="datasets/victoria"):
        path = Path(path)
        male_path = path / "05_Masc_final.csv"
        female_path = path / "05_Fem_final.csv"

        male_data = pd.read_csv(male_path, names=["words"], header=None)
        female_data = pd.read_csv(female_path, names=["words"], header=None)

        self.male = set(male_data["words"])
        self.female = set(female_data["words"])

    def get_male_words(self):
        return self.male
    
    def get_female_words(self):
        return self.female
