from collections import Counter

import numpy as np
from bfair.utils import encode_features

from sklearn.ensemble import BaggingClassifier


class VotingClassifier:
    def __init__(self, estimators):
        self.estimators = estimators

    def predict(self, X):
        predictions = [model.predict(X) for model in self.estimators]
        stack = np.column_stack(predictions)
        most_commons = [Counter(sample).most_common(1)[0][0] for sample in stack]
        return np.asarray(most_commons)


class SklearnMitigator:
    def __init__(self, *, dataset, target, metrics, n_estimators=1, encoders=None):
        self.dataset = dataset
        self.target = target
        self.metrics = metrics
        self.n_estimators = n_estimators
        self.encoders = encoders

    def __call__(self, *models):
        X, y, self.encoders = encode_features(
            self.dataset.data, target=self.target, source_encoders=self.encoders
        )

        estimators = []
        for model in models:
            bag_of_models = BaggingClassifier(
                model, n_estimators=self.n_estimators, max_samples=0.5
            )
            bag_of_models.fit(X, y)
            estimators.extend(bag_of_models.estimators_)

        voting = VotingClassifier(estimators)
        return voting
