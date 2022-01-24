# # Solving the HAHA challenge

# The dataset used is:

# | Dataset | URL |
# |--|--|
# | HAHA 2019 | <https://www.fing.edu.uy/inco/grupos/pln/haha/index.html#data> |

# ## Experimentation parameters
#
# This experiment was run with the following parameters:
#
# | Parameter | Value |
# |--|--|
# | Total epochs         | 1      |
# | Maximum iterations   | 10000  |
# | Timeout per pipeline | 30 min |
# | Global timeout       | -      |
# | Max RAM per pipeline | 20 GB  |
# | Population size      | 50     |
# | Selection (k-best)   | 10     |
# | Early stop           |-       |

# The experiments were run in the following hardware configurations
# (allocated indistinctively according to available resources):

# | Config | CPU | Cache | Memory | HDD |
# |--|--|--|--|--|
# | **A** | 12 core Intel Xeon Gold 6126 | 19712 KB |  191927.2MB | 999.7GB  |
# | **B** | 6 core Intel Xeon E5-1650 v3 | 15360 KB |  32045.5MB  | 2500.5GB |
# | **C** | Quad core Intel Core i7-2600 |  8192 KB |  15917.1MB  | 1480.3GB |

# !!! note
#     The hardware configuration details were extracted with `inxi -CmD` and summarized.

# ## Relevant imports


import argparse
import sys

from autogoal.contrib import find_classes
from autogoal.datasets import haha
from autogoal.kb import Sentence, Seq
from autogoal.search import ConsoleLogger, PESearch, ProgressLogger
from bfair.methods.autogoal import AutoGoalMitigator
from bfair.utils.autogoal import ClassifierWrapper
from sklearn.metrics import f1_score

# Next, we parse the command line arguments to configure the experiment.

# ## Parsing arguments

# The default values are the ones used for the experimentation reported in the paper.


parser = argparse.ArgumentParser()
parser.add_argument("--iterations", type=int, default=10000)
parser.add_argument("--timeout", type=int, default=60)
parser.add_argument("--memory", type=int, default=2)
parser.add_argument("--popsize", type=int, default=50)
parser.add_argument("--selection", type=int, default=10)
parser.add_argument("--global-timeout", type=int, default=60 * 60)
parser.add_argument("--examples", type=int, default=None)
parser.add_argument("--token", default=None)
parser.add_argument("--channel", default=None)
parser.add_argument("--output", default=None)

args = parser.parse_args()

output_stream = open(args.output, mode="a") if args.output else sys.stdout

print(args, file=output_stream, flush=True)


# The next line will print all the algorithms that AutoGOAL found
# in the `contrib` library, i.e., anything that could be potentially used
# to solve an AutoML problem.

for cls in find_classes():
    print("Using: %s" % cls.__name__, file=output_stream, flush=True)

# ## Experimentation

#  Load the HAHA dataset.

X_train, y_train, X_test, y_test = haha.load()

# Instantiate the classifier.
# Note that the input and output types here are defined to match the problem statement,
# i.e., text classification.

mitigator = AutoGoalMitigator.build(
    input=Seq[Sentence],
    n_classifiers=20,
    detriment=20,
    # [start] AutoML args [start]
    #
    search_algorithm=PESearch,
    search_iterations=args.iterations,
    score_metric=f1_score,
    errors="warn",
    pop_size=args.popsize,
    search_timeout=args.global_timeout,
    evaluation_timeout=args.timeout,
    memory_limit=args.memory * 1024 ** 3,
    #
    # [ end ] AutoML args [ end ]
)

loggers = [ProgressLogger(), ConsoleLogger()]

if args.token:
    from autogoal.contrib.telegram import TelegramLogger

    telegram = TelegramLogger(
        token=args.token,
        name=f"HAHA",
        channel=args.channel,
    )
    loggers.append(telegram)

if args.output:
    from bfair.utils.autogoal import FileLogger

    file_logger = FileLogger(output_path=args.output)
    loggers.append(file_logger)

# Finally, running the `AutoML` instance, and printing the results.

pipelines, scores = mitigator.diversify(X_train, y_train, logger=loggers)
model, score = mitigator.ensemble(pipelines, scores, X_train, y_train, logger=loggers)

best_base_model = ClassifierWrapper(pipelines[0])


def report(model, X, y, fit, header):
    try:
        if fit:
            model.fit(X, y)

        y_pred = model.predict(X)
        score = mitigator.score_metric(y, y_pred)
        fscore = mitigator.fairness_metric(X, y, y_pred)

        msg = "\n".join(
            (
                f"Score: {score}",
                f"FScore: {fscore}",
            )
        )

    except Exception as e:
        msg = str(e)

    return f"# {header} #\n{msg}"


reports = [
    report(
        best_base_model,
        X_train,
        y_train,
        fit=True,
        header="BASE @ TRAINING",
    ),
    report(
        best_base_model,
        X_test,
        y_test,
        fit=False,
        header="BASE @ TESTING",
    ),
    report(
        model,
        X_train,
        y_train,
        fit=True,
        header="ENSEMBLER @ TRAINING",
    ),
    report(
        model,
        X_test,
        y_test,
        fit=False,
        header="ENSEMBLER @ TESTING",
    ),
]


for msg in reports:
    print(msg, file=output_stream, flush=True)

if args.output:
    output_stream.close()
