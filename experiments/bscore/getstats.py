import math
import argparse
import pandas as pd
import json
from scipy.stats import t

from pathlib import Path
from statistics import mean, stdev
from bfair.metrics.lm.bscore import BiasScore


def compute_confidence_interval(data, confidence=0.95):
    n = len(data)
    if n < 2:
        return None  # Not enough data to compute confidence interval
    mean_val = mean(data)
    sem = stdev(data) / math.sqrt(n)
    margin = sem * t.ppf((1 + confidence) / 2, n - 1)
    return margin


def main(args):
    path = Path(args.path)
    scores_per_word = pd.read_csv(path)
    results = {}

    for scoring_mode in [BiasScore.S_RATIO, BiasScore.S_LOG]:
        column = scores_per_word[scoring_mode]
        not_infinity = [s for s in column if not math.isinf(s)]
        results[scoring_mode] = {
            "mean": mean(not_infinity),
            "standard_deviation": stdev(not_infinity),
            "confidence_margin": compute_confidence_interval(not_infinity),
        }

    print(json.dumps({path.name: results}))
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump({path.name: results}, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    parser.add_argument("--output", help="Path to output JSON file")
    args = parser.parse_args()
    main(args)
