from typing import Any, Callable, List, Tuple

import numpy as np
from autogoal.kb import MatrixCategorical, Supervised, VectorCategorical
from autogoal.ml import AutoML
from autogoal.sampling import Sampler
from autogoal.search import PESearch
from bfair.methods.voting import MLVotingClassifier, VotingClassifier
from bfair.utils import ClassifierWrapper
from bfair.utils.autogoal import split_input

from .sampling import LogSampler, SampleModel


class AutoGoalEnsembler:
    def __init__(
        self,
        *,
        score_metric: Callable[[Any, Any, Any], float],
        validation_split=0.3,
        maximize=True,
        errors="warn",
        allow_duplicates=False,
        include_filter=".*",
        exclude_filter=None,
        registry=None,
        **search_kwargs,
    ):
        self.score_metric = score_metric
        self.validation_split = validation_split
        self.search_kwargs = search_kwargs
        self.search_kwargs["errors"] = errors
        self.search_kwargs["allow_duplicates"] = allow_duplicates
        self.search_kwargs["maximize"] = maximize

        self._pipeline_space = AutoML(
            input=(MatrixCategorical, Supervised[VectorCategorical]),
            output=VectorCategorical,
            include_filter=include_filter,
            exclude_filter=exclude_filter,
            registry=registry,
        ).make_pipeline_builder()

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
            X,
            y,
            classifiers,
            scores,
            test_on=test_on,
            **run_kwargs,
        )

    def _optimize_sampler_fn(
        self,
        X,
        y,
        classifiers,
        scores,
        *,
        test_on=None,
        **run_kwargs,
    ) -> Tuple[SampleModel, float]:

        if test_on is None:
            X, y, (X_test, y_test) = split_input(X, y, self.validation_split)
        else:
            X_test, y_test = test_on
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
            **self.search_kwargs,
        )

        best, best_fn = search.run(**run_kwargs)
        return best, best_fn

    def _build_generator_and_fn(self, X, y, X_test, y_test, classifiers, scores):
        named_classifiers = {str(c): c for c in classifiers}
        ensembler_types = ["voting", "learning"]

        model_generator = self._pipeline_space

        def build_ensembler(sampler: LogSampler):
            n_classifiers = sampler.discrete(
                min=min(2, len(classifiers)),
                max=len(classifiers),
                handle="n_classifiers",
            )
            names = sampler.multichoice(
                options=named_classifiers,
                k=n_classifiers,
                handle="selected_classifiers",
            )
            selected_classifiers = [named_classifiers[n] for n in names]

            ensembler_type = sampler.choice(ensembler_types, handle="ensembler_type")
            if ensembler_type == "voting":
                ensembler = VotingClassifier(selected_classifiers)
            elif ensembler_type == "learning":
                pipeline = model_generator(sampler)
                model = ClassifierWrapper(pipeline)
                ensembler = MLVotingClassifier(selected_classifiers, model=model)
            else:
                raise Exception(f"Unknown ensembler_type: {ensembler_type}")
            return ensembler

        def generator(sampler: Sampler):
            sampler = LogSampler(sampler)
            ensembler = build_ensembler(sampler)
            ensembler.fit(X, y)
            return SampleModel(sampler, ensembler)

        def fn(generated: SampleModel):
            ensembler = generated.model
            y_pred = ensembler.predict(X_test)
            score = self.score_metric(X_test, y_test, y_pred)
            return score

        return generator, fn
