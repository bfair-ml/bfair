import numpy as np
from collections import Counter


class VotingClassifier:
    def __init__(self, estimators):
        self.estimators = estimators

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
    def __self__(self, estimators, model_init):
        super().__init__(estimators)
        self.model = model_init()

    def fit(self, X, y):
        predictions = self._stack_predictions(X)
        self.model.fit(predictions, y)

    def predict(self, X):
        predictions = self._stack_predictions(X)
        return self.model.predict(predictions)
