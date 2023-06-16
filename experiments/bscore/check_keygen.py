import argparse
import pandas as pd

from pathlib import Path
from statistics import mean, StatisticsError
from bfair.datasets.commongen import load_dataset as load_commongen, CONCEPTS
from bfair.datasets.c2gen import load_dataset as load_c2gen, KEYWORDS
from bfair.metrics.lm.bscore import ALL_GENDER_PAIRS

COMMON_GEN = "common-gen"
C2GEN = "c2gen"


def default_mean(x, default=0):
    try:
        return mean(x)
    except StatisticsError:
        return default


def main(args):
    path = Path(args.path)
    scores_per_word = pd.read_csv(path, index_col="words")

    if args.dataset == COMMON_GEN:
        dataset = load_commongen().all_data
        keygens = dataset[CONCEPTS]
    elif args.dataset == C2GEN:
        dataset = load_c2gen().all_data
        keygens = dataset[KEYWORDS]
    else:
        raise ValueError(args.dataset)

    in_keygens = scores_per_word.index.to_series().apply(
        lambda word: [list(words) for words in keygens if word in words]
    )

    all_gender_words = {w for pair in ALL_GENDER_PAIRS for w in pair}
    gendered_keygens = in_keygens.apply(
        lambda keygens: default_mean(
            [any(w in all_gender_words for w in words) for words in keygens]
        )
    )

    scores_per_word["gendered_keygens"] = gendered_keygens
    scores_per_word["in_keygens"] = in_keygens
    scores_per_word.to_csv(path.with_name(f"{path.stem}-check-keygen{path.suffix}"))

    print("## Keygen Matches")
    print("- Ratio of keygens:", default_mean(bool(matches) for matches in in_keygens))
    print("- Ratio of gendered keygens:", default_mean(v > 0 for v in gendered_keygens))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    parser.add_argument("--dataset", choices=[COMMON_GEN, C2GEN])
    args = parser.parse_args()
    main(args)
