import argparse
import pandas as pd

from pathlib import Path
from bfair.metrics.lm.bscore import BiasScore

def main(args):
    path = Path(args.path)
    scores_per_word = pd.read_csv(path)
    for scoring_mode in [BiasScore.S_RATIO, BiasScore.S_LOG]:
        column = scores_per_word[scoring_mode]
        print(f"## {scoring_mode}")
        print("- **Mean**", column.mean())
        print("- **Standard Deviation**", column.std())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    args = parser.parse_args()
    main(args)
