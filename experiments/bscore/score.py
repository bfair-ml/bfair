import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict

from bfair.datasets.c2gen import load_dataset as load_c2gen, CONTEXT
from bfair.datasets.commongen import load_dataset as load_common_gen, TARGET
from bfair.datasets.victoria import load_dataset as load_victoria, OUTPUT
from bfair.metrics.lm import (
    BiasScore,
    FixedContext,
    InfiniteContext,
    ContinuousContext,
    GERDER_PAIR_ORDER,
)
from bfair.metrics.lm import EnglishGenderedWords, SpanishGenderedWords

FIXED = "fixed"
INFINITE = "infinite"
CONTINUOUS = "continuous"

COMMON_GEN = "common-gen"
C2GEN = "c2gen"
VICTORIA = "victoria"
VICTORIA_GPT35 = "victoria-GPT3.5"
VICTORIA_GPT4O = "victoria-GPT4o"
VICTORIA_LLAMA3 = "victoria-Llama3"
VICTORIA_GEMINI15 = "victoria-Gemini1.5"
VICTORIA_MISTRAL8X7B = "victoria-Mistral8x7b"


def main(args):
    if args.dataset == C2GEN:
        dataset = load_c2gen()
        texts = dataset.data[CONTEXT].str.lower()
        language = dataset.language()
    elif args.dataset == COMMON_GEN:
        dataset = load_common_gen()
        texts = dataset.all_data[TARGET].str.lower()
        language = dataset.language()
    elif args.dataset.startswith(VICTORIA):
        model = args.dataset.split("-")[1]
        dataset = load_victoria(model=model)
        texts = dataset.data[OUTPUT].str.lower()
        language = dataset.language()
    elif Path(args.dataset).exists():
        dataset = pd.read_csv(args.dataset, sep="\t", usecols=["sentence"])
        texts = dataset["sentence"].dropna().str.lower()
        language = "english"
    else:
        raise ValueError(args.dataset)

    word_handler = {
        "english": EnglishGenderedWords(),
        "spanish": SpanishGenderedWords(),
    }[language]

    group_words = word_handler.get_group_words()

    bias_score = BiasScore(
        language=language,
        group_words=group_words,
        context=(
            FixedContext()
            if args.context == FIXED
            else InfiniteContext()
            if args.context == INFINITE
            else ContinuousContext()
        ),
        scoring_modes=[BiasScore.S_RATIO, BiasScore.S_LOG],
        use_root=args.use_root,
        lower_proper_nouns=args.lower_proper_nouns,
        semantic_check=args.semantic_check,
        split_endings=args.split_endings,
    )

    scores = bias_score(texts)
    for scoring_mode, (mean, stdev, _) in scores.items():
        print(f"## {scoring_mode} [{' then '.join(GERDER_PAIR_ORDER)}]")
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
        choices=[FIXED, INFINITE, CONTINUOUS],
        required=True,
    )
    parser.add_argument(
        "--dataset",
        help="Valid options: {}".format(
            [
                COMMON_GEN,
                C2GEN,
                VICTORIA_GPT35,
                VICTORIA_GPT4O,
                VICTORIA_LLAMA3,
                VICTORIA_GEMINI15,
                VICTORIA_MISTRAL8X7B,
                "<PATH>",
            ]
        ),
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
    parser.add_argument(
        "--lower-proper-nouns",
        choices=["yes", "no"],
        required=True,
    )
    parser.add_argument(
        "--semantic-check",
        choices=["auto", "yes", "no"],
        required=True,
    )
    parser.add_argument(
        "--split-endings",
        choices=["auto", "yes", "no"],
        default="auto",
    )

    args = parser.parse_args()
    args.use_root = args.use_root == "yes"
    args.lower_proper_nouns = args.lower_proper_nouns == "yes"
    args.semantic_check = (
        None if args.semantic_check == "auto" else args.semantic_check == "yes"
    )
    args.split_endings = (
        None if args.split_endings == "auto" else args.split_endings == "yes"
    )
    main(args)


if __name__ == "__main__":
    entry_point()
