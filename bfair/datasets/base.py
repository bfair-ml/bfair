import pandas as pd
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self, data: pd.DataFrame, test: pd.DataFrame = None, split_seed=None):
        self.data = data
        self.test = (
            test
            if test is not None
            else self._empty(data)
            if split_seed is None
            else self._split(data, split_seed)
        )

    @staticmethod
    def _split(data, split_seed=0):
        train, test = train_test_split(
            data, test_size=0.3, random_state=split_seed, shuffle=True
        )
        return train, test

    @staticmethod
    def _empty(data):
        test = pd.DataFrame(columns=data.columns).astype(data.dtypes)
        return test

    @staticmethod
    def cost_matrix(gold, prediction):
        return 1
