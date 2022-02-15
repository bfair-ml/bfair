import random

import numpy as np
from bfair.methods import OverfittedVotingClassifier, VotingClassifier
from bfair.methods.voting import stack_predictions


class Model:
    def predict(self, X):
        return random.choices(["A", "B", "C"], k=X.shape[0])


estimators = [Model(), Model(), Model(), Model(), Model()]

X = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
y = ["A", "B", "C", "A"]
predictions = stack_predictions(X, estimators)

print(y)
correct = predictions == np.asarray(y)[np.newaxis].transpose()
scores = np.sum(correct, axis=0) / correct.shape[0]
print(scores)
print(predictions)

for classifier in [
    VotingClassifier(estimators, scores),
    OverfittedVotingClassifier(estimators, scores),
]:
    print("---------")

    classifier.fit(predictions, y, on_predictions=True)
    y_pred = classifier.predict(predictions, on_predictions=True)

    print(y_pred)
