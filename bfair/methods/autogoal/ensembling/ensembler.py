from typing import List, Tuple

from autogoal.sampling import Sampler
from autogoal.search import PESearch
from bfair.utils import ClassifierWrapper

from .sampling import LogSampler, SampleModel


class AutoGoalEnsembler:
    def __init__(
        self, *, score_metric, errors="warn", allow_duplicates=False, **search_kwargs
    ):
        self.score_metric = score_metric
        self.search_kwargs = search_kwargs
        self.search_kwargs["errors"] = errors
        self.search_kwargs["allow_duplicates"] = allow_duplicates

    def __call__(
        self,
        X,
        y,
        classifiers: List[ClassifierWrapper],
        scores: List[float],
        *,
        test_on=None,
        **run_kwargs,
    ):
        return self._optimize_sampler_fn(
            X, y, classifiers, scores, test_on, **run_kwargs
        )

    def _optimize_sampler_fn(
        self,
        X,
        y,
        classifiers,
        scores,
        test_on=None,
        **run_kwargs,
    ) -> Tuple[SampleModel, float]:

        X_test, y_test = (X, y) if test_on is None else test_on
        generator, fn = self._build_generator_and_fn(
            X,
            y,
            X_test,
            y_test,
            classifiers,
            scores,
        )
        search = PESearch(
            generator_fn=generator,
            fitness_fn=fn,
            maximize=self.maximize,
            **self.search_kwargs,
        )

        best, best_fn = search.run(**run_kwargs)
        return best, best_fn

    def _build_generator_and_fn(self, X, y, X_test, y_test, classifiers, scores):
        def build_ensembler(sampler: LogSampler):
            return ...  # TODO: use sampler to build ensembler

        def generator(sampler: Sampler):
            sampler = LogSampler(sampler)
            ensembler = build_ensembler(sampler)
            return SampleModel(sampler, ensembler)

        def fn(generated: SampleModel):
            ensembler = generated.model
            ensembler.fit(X, y)
            y_pred = ensembler.predict(X_test)
            score = self.score_metric(y_test, y_pred)
            return score

        return generator, fn
