import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict

from bfair.datasets.c2gen import load_dataset as load_c2gen, CONTEXT
from bfair.datasets.commongen import load_dataset as load_common_gen, TARGET
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

COMMON_GEN = "common-gen"
C2GEN = "c2gen"


def main(args):
    if args.dataset == C2GEN:
        dataset = load_c2gen()
        texts = dataset.data[CONTEXT].str.lower()
    elif args.dataset == COMMON_GEN:
        dataset = load_common_gen()
        texts = dataset.all_data[TARGET].str.lower()
    elif Path(args.dataset).exists():
        dataset = pd.read_csv(args.dataset, sep="\t", usecols=["sentence"])
        texts = dataset["sentence"].dropna().str.lower()
    else:
        raise ValueError(args.dataset)

    group_words = defaultdict(set)
    for pairs in [DM_GENDER_PAIRS, PENN_GENDER_PAIRS, WIKI_GENDER_PAIRS]:
        for i, gender in enumerate(GERDER_PAIR_ORDER):
            group_words[gender].update((p[i] for p in pairs))

    bias_score = BiasScore(
        language="english",
        group_words=group_words,
        context=FixedContext() if args.context == FIXED else InfiniteContext(),
        scoring_modes=[BiasScore.S_RATIO, BiasScore.S_LOG],
        use_root=args.use_root,
    )

    scores = bias_score(texts)
    for scoring_mode, (mean, stdev, _) in scores.items():
        print(f"## {scoring_mode} ({' then '.join(GERDER_PAIR_ORDER)})")
        print("- **Mean**", mean)
        print("- **Standard Deviation**", stdev)

    print("## Scores per word")
    scores_per_word = pd.concat((df for _, _, df in scores.values()), axis=1)
    scores_per_word = scores_per_word.sort_values(by=BiasScore.S_LOG)

    if args.export_csv is not None:
        print(f"Saving scores per word to {args.export_csv}")
        scores_per_word.to_csv(args.export_csv)
    else:
        print(scores_per_word.to_markdown())


def entry_point():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--context",
        choices=[FIXED, INFINITE],
        default=INFINITE,
    )
    parser.add_argument(
        "--dataset",
        help=f"Valid options: {[COMMON_GEN, C2GEN, '<PATH>']}",
        default=C2GEN,
    )
    parser.add_argument(
        "--export-csv",
        default=None,
    )
    parser.add_argument(
        "--use-root",
        choices=["yes", "no"],
        required=True,
    )
    args = parser.parse_args()
    args.use_root = args.use_root == "yes"
    main(args)


if __name__ == "__main__":
    entry_point()
