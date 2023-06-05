import argparse
from collections import defaultdict

from bfair.datasets.c2gen import load_dataset, CONTEXT
from bfair.metrics.lm.bscore import (
    BiasScore,
    FixedContext,
    InfiniteContext,
    GERDER_PAIR_ORDER,
    DM_GENDER_PAIRS,
    PENN_GENDER_PAIRS,
    WIKI_GENDER_PAIRS,
)

FIXED = "fixed"
INFINITE = "infinite"


def main(args):
    dataset = load_dataset()
    contexts = dataset.data[CONTEXT]
    text = "\n ".join(contexts).lower()

    group_words = defaultdict(set)
    for pairs in [DM_GENDER_PAIRS, PENN_GENDER_PAIRS, WIKI_GENDER_PAIRS]:
        for i, gender in enumerate(GERDER_PAIR_ORDER):
            group_words[gender].update((p[i] for p in pairs))

    bias_score = BiasScore(
        language="english",
        group_words=group_words,
        context=FixedContext() if args.context == FIXED else InfiniteContext(),
        scoring_modes=[BiasScore.S_RATIO, BiasScore.S_LOG],
    )

    scores = bias_score(text)
    for scoring_mode, (mean, stdev, scores_per_word) in scores.items():
        print(f"# {scoring_mode} ({GERDER_PAIR_ORDER})")
        print("Mean", mean)
        print("Standard Deviation", stdev)
        print("Scores per word")
        print("[")
        for score in scores_per_word:
            print(f"    {score}")
        print("]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--context",
        choices=[FIXED, INFINITE],
        default=INFINITE,
    )
    args = parser.parse_args()
    main(args)
