import numpy as np
from collections import Counter


class VotingClassifier:
    def __init__(self, estimators):
        self.estimators = estimators

    def predict(self, X):
        predictions = [model.predict(X) for model in self.estimators]
        stack = np.column_stack(predictions)
        most_commons = [Counter(sample).most_common(1)[0][0] for sample in stack]
        return np.asarray(most_commons)


class MLVotingClassifier:
    def __self__(self, estimators, model_init):
        self.estimators = estimators
        self.model = model_init()

    def _build_input(self, X):
        predictions = [model.predict(X) for model in self.estimators]
        return np.column_stack(predictions)

    def fit(self, X, y):
        _input = self._build_input(X)
        self.model.fit(_input, y)

    def predict(self, X):
        _input = self._build_input(X)
        return self.model.predict(_input)
