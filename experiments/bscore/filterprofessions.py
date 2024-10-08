import argparse
import pandas as pd

from pathlib import Path
from experiments.bscore.score import get_group_words_and_to_inspect


def main(args):
    path = Path(args.path)
    scores_per_word = pd.read_csv(path)

    if scores_per_word.empty:
        print(f"WARNING: Empty! ... skipping ({path})")
        return

    _, professions_group_words = get_group_words_and_to_inspect(
        args.language, False, True
    )
    professions = professions_group_words.get_all_words()

    is_profession = scores_per_word["words"].apply(lambda x: x in professions)
    is_noun = scores_per_word["pos"] == "NOUN"
    selected = scores_per_word[is_profession & is_noun].set_index("words")
    selected.to_csv(
        path.with_name(f"{path.stem}-filtered-by-professions.{path.suffix}")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    parser.add_argument("--language")
    args = parser.parse_args()
    main(args)
