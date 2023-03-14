import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from experiments.parse import get_top_ensembles, K_FSCORE
from autogoal.search import NSPESearch


class NonDominatedSorter:
    def __init__(self, maximize):
        self._maximize = maximize

    def _improves(self, a, b) -> bool:
        maximize = self._maximize
        if not isinstance(maximize, (tuple, list)):
            a, b, maximize = (a,), (b,), (maximize,)
        not_worst = all(
            (ai >= bi if m else ai <= bi) for ai, bi, m in zip(a, b, maximize)
        )
        better = any((ai > bi if m else ai < bi) for ai, bi, m in zip(a, b, maximize))
        return not_worst and better

    def non_dominated_sort(self, fns):
        fronts = [[]]
        domination_counts = [0] * len(fns)
        dominated_fns = [[] for _ in fns]

        for i, fn_i in enumerate(fns):
            for j, fn_j in enumerate(fns):
                if self._improves(fn_i, fn_j):
                    dominated_fns[i].append(j)
                elif self._improves(fn_j, fn_i):
                    domination_counts[i] += 1
            if domination_counts[i] == 0:
                fronts[0].append(i)

        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for idx in fronts[i]:
                for dominated_idx in dominated_fns[idx]:
                    domination_counts[dominated_idx] -= 1
                    if domination_counts[dominated_idx] == 0:
                        next_front.append(dominated_idx)
            i += 1
            fronts.append(next_front)

        return fronts[:-1]


PATHS = {
    20: "/home/coder/bfair/output/adult-SP/adult.0.20.txt",
    50: "/home/coder/bfair/output/adult-SP/adult.0.50.txt",
}

KEYS = ["DSP", "accuracy"]


def get_data(path, keys, tag):
    with open(path) as fd:
        text = fd.read()

    fns = [top.get(K_FSCORE) for top in get_top_ensembles(text)]

    nsorter = NonDominatedSorter(maximize=(False, True))
    fronts = nsorter.non_dominated_sort(fns)
    front_0 = [fns[i] for i in fronts[0]]

    data = [{k: v for k, v in zip(keys, values)} for values in front_0]

    df = pd.DataFrame.from_dict(data)
    df = df[df["accuracy"] >= 0.7]
    df = df[df["DSP"] < 0.15]

    df["ensemble size"] = tag

    return df


df = pd.concat([get_data(path, KEYS, tag) for tag, path in PATHS.items()])

plt.figure()
sns.lmplot(
    x="DSP", y="accuracy", hue="ensemble size", aspect=1.8, data=df, order=2
).fig.suptitle("Pareto Front")
plt.savefig("pareto-front.pdf")
