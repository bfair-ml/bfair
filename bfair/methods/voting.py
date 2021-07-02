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
