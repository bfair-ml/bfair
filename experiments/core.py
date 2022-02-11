import argparse
import sys
from collections import OrderedDict
from itertools import product

from autogoal.contrib import find_classes
from autogoal.search import PESearch, RichLogger
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
from bfair.metrics import disagreement, double_fault_inverse
from bfair.utils import ClassifierWrapper


def setup():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n-classifiers", type=int, default=20)
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
    parser.add_argument(
        "--diversity",
        type=str,
        default="double-fault",
        choices=["double-fault", "disagreement", "shuffle", "arbitrary", "best"],
    )

    return parser.parse_args()


def run(load_dataset, input_type, score_metric, maximize, args, title):
    path = args.output.format(title)
    output_stream = open(path, mode="a") if path else sys.stdout

    try:
        _run(
            load_dataset,
            input_type,
            score_metric,
            maximize,
            args,
            title,
            output_stream,
            path,
        )
    except Exception as e:
        print("\n", "ERROR", "\n", str(e), "\n", file=output_stream, flush=True)
    finally:
        if path:
            output_stream.close()


def _run(
    load_dataset, input_type, score_metric, maximize, args, title, output_stream, path
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

    X_train, y_train, X_test, y_test = load_dataset(max_examples=args.examples)

    mitigator = AutoGoalMitigator.build(
        input=input_type,
        n_classifiers=args.n_classifiers,
        detriment=20,
        diversity_metric=diversity_metric,
        ranking_fn=ranking_fn,
        maximize=maximize,
        # [start] AutoML args [start]
        #
        search_algorithm=PESearch,
        score_metric=score_metric,
        pop_size=args.popsize,
        search_iterations=args.iterations,
        evaluation_timeout=args.timeout,
        memory_limit=args.memory * 1024 ** 3,
        search_timeout=args.global_timeout,
        errors="warn",
        #
        # [ end ] AutoML args [ end ]
    )

    loggers = [RichLogger()]

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
    )
    model, score = mitigator.ensemble(
        pipelines,
        scores,
        X_train,
        y_train,
        logger=loggers,
    )

    base_models = [ClassifierWrapper(p) for p in pipelines]
    best_base_model = base_models[0]

    models = OrderedDict()
    for i, base_model in enumerate(base_models):
        models[f"base-model-{i}"] = base_model
    models["ensemble"] = model
    models["best-base-model"] = best_base_model

    print("## Performance ...", file=output_stream)
    for msg in inspect(
        models,
        X_train,
        y_train,
        X_test,
        y_test,
        mitigator.score_metric,
        mitigator.fairness_metric,
    ):
        print(msg, file=output_stream, flush=True)

    print("## Coverage", file=output_stream)
    for title, coverage in [
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
        print(f"- {title}: {coverage}", file=output_stream, flush=True)


def inspect(
    models: dict,
    X_train,
    y_train,
    X_test,
    y_test,
    score_metric,
    fairness_metric,
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
            header=f"{name.upper()} @ {'TRAINING' if fit else 'TESTING'}",
        )
        for (name, model), (X, y, fit) in product(models.items(), collections)
    ]


def report(model, X, y, fit, score_metric, fairness_metric, header):
    try:
        if fit:
            model.fit(X, y)

        y_pred = model.predict(X)
        score = score_metric(y, y_pred)
        fscore = fairness_metric(X, y, y_pred)

        msg = "\n".join(
            (
                f"Score: {score}",
                f"FScore: {fscore}",
            )
        )

    except Exception as e:
        msg = str(e)

    return f"# {header} #\n{msg}"
