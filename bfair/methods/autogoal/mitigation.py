from .diversification import AutoGoalDiversifier
from .ensembling import AutoGoalEnsembler


class AutoGoalMitigator:
    def __init__(self, diversifier, ensembler):
        self.diversifier = diversifier
        self.ensembler = ensembler

    @classmethod
    def build(cls, *, input, n_classifiers: int, maximize=True, **automl_kwargs):
        diversifier = cls.build_diversifier(
            input=input,
            n_classifiers=n_classifiers,
            maximize=maximize,
            **automl_kwargs,
        )
        ensembler = cls.build_ensembler(**automl_kwargs)
        return cls(diversifier, ensembler)

    @classmethod
    def build_diversifier(
        cls, *, input, n_classifiers: int, maximize=True, **automl_kwargs
    ):
        return AutoGoalDiversifier(
            input=input,
            n_classifiers=n_classifiers,
            maximize=maximize,
            **automl_kwargs,
        )

    @classmethod
    def build_ensembler(cls, **automl_kwargs):
        raise NotImplementedError()

    def __call__(self, X, y, **search_kwargs):
        classifiers, scores = self.diversifier(X, y, **search_kwargs)
        model = self.ensembler(X, y, classifiers, scores, **search_kwargs)
        return model
