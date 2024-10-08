import argparse
import pandas as pd

from pathlib import Path
from bfair.metrics.lm.bscore import BiasScore

from experiments.bscore.score import get_group_words_and_to_inspect


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

    _, professions_group_words = get_group_words_and_to_inspect(
        args.language, False, True
    )
    professions = professions_group_words.get_all_words()

    is_profession = scores_per_word["words"].apply(lambda x: x in professions)
    selected = scores_per_word[is_profession].set_index("words")
    selected.to_csv(
        path.with_name(f"{path.stem}-filtered-by-professions.{path.suffix}")
    )


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
    args = parser.parse_args()
    args.semantic_model = args.semantic_model == "yes"
    args.use_legacy_semantics = args.use_legacy_semantics == "yes"
    main(args)
