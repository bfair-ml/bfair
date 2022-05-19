import argparse
import sys
from collections import OrderedDict
from itertools import product
from pathlib import Path
from typing import Any, Callable, Tuple

import bfair.metrics.disparity as disparity
from autogoal.contrib import find_classes
from autogoal.search import ConsoleLogger, PESearch
from bfair.methods import AutoGoalMitigator
from bfair.methods.autogoal.diversification import (
    build_best_performance_ranking_fd,
    build_random_but_single_best_ranking_fn,
    build_random_ranking_fn,
)
from bfair.methods.voting import (
    optimistic_oracle,
    optimistic_oracle_coverage,
    overfitted_oracle,
    overfitted_oracle_coverage,
)
from bfair.metrics import DIFFERENCE, RATIO, disagreement, double_fault_inverse
from bfair.utils import ClassifierWrapper
from numpy import argmax, argmin


def to_number(value: str):
    try:
        return int(value)
    except ValueError:
        return float(value)


def setup():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n-classifiers", type=int, default=20)
    parser.add_argument("--detriment", type=to_number, default=20)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--memory", type=int, default=2)
    parser.add_argument("--popsize", type=int, default=50)
    parser.add_argument("--selection", type=int, default=10)
    parser.add_argument("--global-timeout", type=int, default=60 * 60)
    parser.add_argument("--examples", type=int, default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--channel", default=None)
    parser.add_argument("--output", default="")
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--diversity",
        type=str,
        default="double-fault",
        choices=["double-fault", "disagreement", "shuffle", "arbitrary", "best"],
    )
    parser.add_argument(
        "--fairness",
        type=str,
        default=None,
        choices=[
            "statistical-parity",
            "equal-opportunity",
            "equalized-odds",
            "accuracy-disparity",
        ],
        action="append",
    )
    parser.add_argument(
        "--fmode",
        type=str,
        default=DIFFERENCE,
        choices=[DIFFERENCE, RATIO],
    )

    return parser.parse_args()


def run(
    *,
    load_dataset: Callable[[int], Tuple[Any, Any, Any, Any]],
    input_type,
    score_metric,
    maximize,
    args,
    title,
    protected_attributes=None,
    target_attribute=None,
    positive_target=None,
    sensor=None,
    diversifier_run_kwargs=None,
    ensembler_run_kwargs=None,
):
    path = args.output.format(title)
    if path:
        Path(path).parent.mkdir(exist_ok=True)
        output_stream = open(path, mode="a")
    else:
        output_stream = sys.stdout

    try:
        _run(
            load_dataset=load_dataset,
            input_type=input_type,
            score_metric=score_metric,
            maximize=maximize,
            protected_attributes=protected_attributes,
            target_attribute=target_attribute,
            positive_target=positive_target,
            sensor=sensor,
            args=args,
            title=title if args.title is None else f"{title}: {args.title}",
            output_stream=output_stream,
            path=path,
            diversifier_run_kwargs=diversifier_run_kwargs or {},
            ensembler_run_kwargs=ensembler_run_kwargs or {},
        )
    except Exception as e:
        print("\n", "ERROR", "\n", str(e), "\n", file=output_stream, flush=True)
    finally:
        if path:
            output_stream.close()


def _run(
    *,
    load_dataset,
    input_type,
    score_metric,
    maximize,
    protected_attributes,
    target_attribute,
    positive_target,
    sensor,
    args,
    title,
    output_stream,
    path,
    diversifier_run_kwargs,
    ensembler_run_kwargs,
):

    print(args, file=output_stream, flush=True)
    for cls in find_classes():
        print("Using: %s" % cls.__name__, file=output_stream, flush=True)

    print(f"Experiment: {title.upper()}", file=output_stream, flush=True)

    ranking_fn = None
    diversity_metric = None
    if args.diversity == "double-fault":
        diversity_metric = double_fault_inverse
    elif args.diversity == "disagreement":
        diversity_metric = disagreement
    elif args.diversity == "shuffle":
        ranking_fn = build_random_ranking_fn()
    elif args.diversity == "arbitrary":
        ranking_fn = build_random_but_single_best_ranking_fn(maximize=maximize)
    elif args.diversity == "best":
        ranking_fn = build_best_performance_ranking_fd(maximize=maximize)
    else:
        raise ValueError(f"Unknown value for diversity metric: {args.diversity}")

    args_fmetrics = {}
    if args.fairness is not None:
        try:
            args_fmetrics = {
                name: getattr(disparity, name.replace("-", "_"))
                for name in args.fairness
            }
        except AttributeError:
            raise ValueError(f"Unknown value for fairness metric: {args.fairness}")

    fairness_metrics = None
    if args_fmetrics:
        fairness_metrics = list(args_fmetrics.values())

    maximize_fmetric = args.fmode == RATIO

    X_train, y_train, X_test, y_test = load_dataset(max_examples=args.examples)

    mitigator = AutoGoalMitigator.build(
        input=input_type,
        n_classifiers=args.n_classifiers,
        detriment=args.detriment,
        score_metric=score_metric,
        diversity_metric=diversity_metric,
        fairness_metrics=fairness_metrics,
        ranking_fn=ranking_fn,
        maximize=maximize,
        maximize_fmetric=maximize_fmetric,
        protected_attributes=protected_attributes,
        target_attribute=target_attribute,
        positive_target=positive_target,
        sensor=sensor,
        metric_kwargs=dict(
            mode=args.fmode,
        ),
        # [start] AutoML args [start]
        #
        search_algorithm=PESearch,
        pop_size=args.popsize,
        search_iterations=args.iterations,
        evaluation_timeout=args.timeout,
        memory_limit=args.memory * 1024**3,
        search_timeout=args.global_timeout,
        errors="warn",
        #
        # [ end ] AutoML args [ end ]
    )

    loggers = [ConsoleLogger()]

    if args.token:
        from autogoal.contrib.telegram import TelegramLogger

        telegram = TelegramLogger(
            token=args.token,
            name=title.upper(),
            channel=args.channel,
        )
        loggers.append(telegram)

    if path:
        from bfair.utils.autogoal import FileLogger

        file_logger = FileLogger(output_path=path)
        loggers.append(file_logger)

    pipelines, scores = mitigator.diversify(
        X_train,
        y_train,
        logger=loggers,
        **diversifier_run_kwargs,
    )
    model, score = mitigator.ensemble(
        pipelines,
        scores,
        X_train,
        y_train,
        logger=loggers,
        **ensembler_run_kwargs,
    )

    for i, p in enumerate(pipelines):
        print(f"Pipeline-{i}", file=output_stream)
        print(p, file=output_stream, flush=True)

    base_models = ClassifierWrapper.wrap(pipelines)
    best_index = argmax(scores) if maximize else argmin(scores)
    best_base_model = base_models[best_index]

    print(f"Best Base Model @ {best_index}", file=output_stream, flush=True)

    models = OrderedDict()
    for i, base_model in enumerate(base_models):
        models[f"base-model-{i}"] = base_model
    models["ensemble"] = model
    models["best-base-model"] = best_base_model

    other_fmetrics = {
        name: mitigator.build_fairness_fn(
            fairness_metrics=metric,
            protected_attributes=protected_attributes,
            target_attribute=target_attribute,
            positive_target=positive_target,
            metric_kwargs=dict(
                mode=args.fmode,
            ),
            sensor=sensor,
        )
        for name, metric in args_fmetrics.items()
    }

    print("## Performance ...", file=output_stream)
    for msg in inspect(
        models,
        X_train,
        y_train,
        X_test,
        y_test,
        mitigator.score_metric,
        mitigator.fairness_metric,
        other_fmetrics=other_fmetrics,
    ):
        print(msg, file=output_stream, flush=True)

    print("## Coverage", file=output_stream)
    for caption, coverage in [
        (
            "Optimistic Oracle (score_metric)",
            optimistic_oracle(
                X_test,
                y_test,
                mitigator.score_metric,
                base_models,
            ),
        ),
        (
            "Optimistic Oracle (accuracy)",
            optimistic_oracle_coverage(
                X_test,
                y_test,
                base_models,
            ),
        ),
        (
            "Overfitted Oracle (score_metric)",
            overfitted_oracle(
                X_test,
                y_test,
                mitigator.score_metric,
                base_models,
            ),
        ),
        (
            "Overfitted Oracle (accuracy)",
            overfitted_oracle_coverage(
                X_test,
                y_test,
                base_models,
            ),
        ),
    ]:
        print(f"- {caption}: {coverage}", file=output_stream, flush=True)


def inspect(
    models: dict,
    X_train,
    y_train,
    X_test,
    y_test,
    score_metric,
    fairness_metric,
    other_fmetrics=None,
):
    collections = [(X_train, y_train, True), (X_test, y_test, False)]
    return [
        report(
            model,
            X=X,
            y=y,
            fit=fit,
            score_metric=score_metric,
            fairness_metric=fairness_metric,
            other_fmetrics=other_fmetrics or {},
            header=f"{name.upper()} @ {'TRAINING' if fit else 'TESTING'}",
        )
        for (name, model), (X, y, fit) in product(models.items(), collections)
    ]


def report(model, X, y, fit, score_metric, fairness_metric, other_fmetrics, header):
    items = []
    try:
        if fit:
            model.fit(X, y)

        y_pred = model.predict(X)

        score = score_metric(y, y_pred)
        items.append(f"Score: {score}")
        fscore = fairness_metric(X, y, y_pred)
        items.append(f"FScore: {fscore}")

        for name, f in other_fmetrics.items():
            items.append(f"{name}: {f(X, y, y_pred)}")

    except Exception as e:
        items.append(str(e))

    msg = "\n".join(items)
    return f"# {header} #\n{msg}"
