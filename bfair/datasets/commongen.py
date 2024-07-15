from .base import Dataset

import datasets as db


CONCEPTS = "concepts"
TARGET = "target"


def load_dataset(**kargs):
    return CommonGen.load()


class CommonGen(Dataset):
    @classmethod
    def load(cls):
        source = db.load_dataset("common_gen", "default")
        train = source["train"].to_pandas()
        validation = source["validation"].to_pandas()
        test = source["test"].to_pandas()
        return CommonGen(data=train, validation=validation, test=test)
    
    @staticmethod
    def language():
        return "english"
