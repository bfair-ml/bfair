import argparse
import spacy
import pandas as pd

from pathlib import Path
from bfair.metrics.lm.bscore import BiasScore


def main(args):
    path = Path(args.path)
    scores_per_word = pd.read_csv(path)

    if scores_per_word.empty:
        print(f"WARNING: Empty! ... skipping ({path})")
        return

    nlp = BiasScore.get_nlp(
        args.language,
        args.semantic_model,
        args.use_legacy_semantics,
    )
    with_pos = scores_per_word["words"].apply(lambda x: nlp(x)[0].pos_ == args.pos)
    selected = scores_per_word[with_pos].set_index("words")
    selected.to_csv(path.with_name(f"{path.stem}-filtered-by-{args.pos}{path.suffix}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    parser.add_argument("--language")
    parser.add_argument("--semantic-model", choices=["yes", "no"], default=False)
    parser.add_argument(
        "--use-legacy-semantics",
        choices=["yes", "no"],
        default="no",
    )
    parser.add_argument("--pos", default="ADJ")

    args = parser.parse_args()
    args.semantic_model = args.semantic_model == "yes"
    args.semantic_model = args.semantic_model == "yes"
    args.use_legacy_semantics = args.use_legacy_semantics == "yes"
    main(args)
