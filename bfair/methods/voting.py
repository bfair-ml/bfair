import numpy as np
from collections import Counter


class VotingClassifier:
    def __init__(self, estimators):
        self.estimators = estimators

    @property
    def fitted(self):
        return True

    def fit_estimators(self, X, y):
        for estimator in self.estimators:
            estimator.fit(X, y)

    def fit(self, X, y):
        pass

    def predict(self, X):
        predictions = self._stack_predictions(X)
        most_commons = [Counter(sample).most_common(1)[0][0] for sample in predictions]
        return np.asarray(most_commons)

    def _stack_predictions(self, X):
        predictions = [model.predict(X) for model in self.estimators]
        return np.column_stack(predictions)


class MLVotingClassifier(VotingClassifier):
    def __init__(self, estimators, *, model=None, model_init=None):
        if model is None == model_init is None:
            raise ValueError(
                "One and only one between `model` and `model_init` should be supplied"
            )
        super().__init__(estimators)
        self.model = model_init() if model is None else model

    @property
    def fitted(self):
        return self.model.fitted

    def fit(self, X, y):
        predictions = self._stack_predictions(X)
        self.model.fit(predictions, y)

    def predict(self, X):
        predictions = self._stack_predictions(X)
        return self.model.predict(predictions)
