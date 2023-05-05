import pandas as pd
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(
        self,
        data: pd.DataFrame,
        *,
        validation: pd.DataFrame = None,
        test: pd.DataFrame = None,
        split_seed=None,
        stratify_by=None,
    ):
        self.data, self.test = (
            (data, test)
            if test is not None
            else (data, self._empty(data))
            if split_seed is None
            else self._split(data, split_seed, stratify_by)
        )
        self.validation = self._empty(data) if validation is None else validation

    @property
    def columns(self):
        return self.data.columns

    @staticmethod
    def _split(data, split_seed=0, stratify_by=None):
        train, test = train_test_split(
            data,
            test_size=0.3,
            random_state=split_seed,
            shuffle=True,
            stratify=None if stratify_by is None else data[stratify_by],
        )
        return train, test

    @staticmethod
    def _empty(data):
        test = pd.DataFrame(columns=data.columns).astype(data.dtypes)
        return test

    @staticmethod
    def cost_matrix(gold, prediction):
        return 1
