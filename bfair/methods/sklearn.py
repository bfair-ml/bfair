from bfair.utils import encode_features

from sklearn.ensemble import BaggingClassifier

from .voting import VotingClassifier


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
