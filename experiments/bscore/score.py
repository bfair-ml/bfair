import argparse
import pandas as pd
from pathlib import Path

from bfair.datasets.c2gen import load_dataset as load_c2gen, CONTEXT
from bfair.datasets.commongen import load_dataset as load_common_gen, TARGET
from bfair.datasets.victoria import load_dataset as load_victoria, OUTPUT
from bfair.datasets.ilenia import load_dataset as load_ilenia, SENTENCE, ANALYSIS
from bfair.datasets.rhopa64 import load_dataset as load_rhopa64, OUTPUT, ANNOTATIONS
from bfair.metrics.lm import (
    BiasScore,
    FixedContext,
    InfiniteContext,
    ContinualContext,
)
from bfair.metrics.lm import (
    EnglishGenderedWords,
    SpanishGenderedWords,
    DynamicGroupWords,
)
from bfair.metrics.lm.bscore import GenderMorphAnalyzer

FIXED = "fixed"
INFINITE = "infinite"
CONTINUAL = "continuous"

COMMON_GEN = "common-gen"
C2GEN = "c2gen"
VICTORIA = "victoria"
ILENIA = "ilenia"
RHOPA64 = "rhopa64"

VICTORIA_GPT35_LEADING = "victoria-GPT3.5-leading"
VICTORIA_GPT4O_LEADING = "victoria-GPT4o-leading"
VICTORIA_LLAMA3_LEADING = "victoria-Llama3-leading"
VICTORIA_GEMINI15_LEADING = "victoria-Gemini1.5-leading"
VICTORIA_MISTRAL8X7B_LEADING = "victoria-Mistral8x7b-leading"

VICTORIA_GPT35_NO_LEADING = "victoria-GPT3.5-independent"
VICTORIA_GPT4O_NO_LEADING = "victoria-GPT4o-independent"
VICTORIA_LLAMA3_NO_LEADING = "victoria-Llama3-independent"
VICTORIA_GEMINI15_NO_LEADING = "victoria-Gemini1.5-independent"
VICTORIA_MISTRAL8X7B_NO_LEADING = "victoria-Mistral8x7b-independent"


def get_group_words_and_to_inspect(
    language,
    exclude_professions,
    inspect_professions,
    include_plurals,
):
    if exclude_professions and language != "spanish":
        raise ValueError(
            f"{language.title()} language does not support professions exclusion."
        )

    if inspect_professions and exclude_professions:
        raise ValueError("For inspecting professions they cannot be excluded.")

    if inspect_professions and language != "spanish":
        raise ValueError(
            f"{language.title()} language does not support professions inspection."
        )

    if include_plurals and language != "spanish":
        raise ValueError(f"{language.title()} language does not support plural forms.")

    word_handler = {
        "english": EnglishGenderedWords(),
        "spanish": SpanishGenderedWords(
            include_professions=not exclude_professions,
            include_plurals=include_plurals,
        ),
    }.get(language)

    if word_handler is None:
        print(f"No group words defined for {language.title()} language. Ignore if using dynamic group words.")
        return None, None

    group_words = word_handler.get_group_words()

    group_words_to_inspect = (
        word_handler.get_professions_as_group_words() if inspect_professions else None
    )

    return group_words, group_words_to_inspect


def get_morph_analyzer(language, use_morph):
    if not use_morph:
        return None

    if language == "spanish":
        return GenderMorphAnalyzer(
            {
                "del": GenderMorphAnalyzer.MALE,
                "al": GenderMorphAnalyzer.MALE,
            }
        )

    raise ValueError(
        f"{language.title()} language does not support morphological analyzis."
    )


def main(args):
    group_words = None

    if args.dataset == C2GEN:
        dataset = load_c2gen()
        texts = dataset.data[CONTEXT].str.lower()
        language = dataset.language()
    elif args.dataset == COMMON_GEN:
        dataset = load_common_gen()
        texts = dataset.all_data[TARGET].str.lower()
        language = dataset.language()
    elif args.dataset.startswith(VICTORIA):
        info = args.dataset.split("-")
        if len(info) < 3:
            raise ValueError(args.dataset)
        _, model, mode = info
        dataset = load_victoria(model=model, leading=mode == "leading")
        texts = dataset.data[OUTPUT].str.lower()
        language = dataset.language()
    elif args.dataset.startswith(ILENIA):
        _, *params = args.dataset.split("|")
        kwargs = {
            name: value for param in params for name, value in (param.split(":"),)
        }
        dataset = load_ilenia(**kwargs)
        texts = dataset.data[SENTENCE]
        language = dataset.language
        group_words = (
            DynamicGroupWords(texts, dataset.data[ANALYSIS], ["male", "female"])
            if dataset.annotated
            else None
        )
    elif args.dataset.startswith(RHOPA64):
        _, *params = args.dataset.split("|")
        kwargs = {
            name: value for param in params for name, value in (param.split(":"),)
        }
        dataset = load_rhopa64(**kwargs)
        texts = dataset.data[OUTPUT].explode()
        language = dataset.language
        group_words = (
            DynamicGroupWords(texts, dataset.data[ANNOTATIONS].explode(), ["male", "female"])
            if dataset.annotated
            else None
        )

    elif Path(args.dataset).exists():
        dataset = pd.read_csv(args.dataset, sep="\t", usecols=["sentence"])
        texts = dataset["sentence"].dropna().str.lower()
        language = "english"
    else:
        raise ValueError(args.dataset)

    _group_words, group_words_to_inspect = get_group_words_and_to_inspect(
        language,
        args.exclude_professions,
        args.inspect_professions,
        args.include_plurals,
    )
    if group_words is None:
        group_words = _group_words
    else:
        print("Using precomputed group words.")

    morph_analyzer = get_morph_analyzer(language, args.use_morph)

    bias_score = BiasScore(
        language=language,
        group_words=group_words,
        context=(
            FixedContext()
            if args.context == FIXED
            else InfiniteContext()
            if args.context == INFINITE
            else ContinualContext()
        ),
        scoring_modes=[BiasScore.S_RATIO, BiasScore.S_LOG],
        use_root=args.use_root,
        merge_paragraphs=dataset.annotated,
        lower_proper_nouns=args.lower_proper_nouns,
        semantic_check=args.semantic_check,
        use_legacy_semantics=args.use_legacy_semantics,
        split_endings=args.split_endings,
        group_words_to_inspect=group_words_to_inspect,
        morph_analyzer=morph_analyzer,
    )

    scores, simple_scores = bias_score(texts)

    print("## Simple Scores")
    print(simple_scores)

    for scoring_mode, (mean, stdev, _) in scores.items():
        print(f"## {scoring_mode} [{' then '.join(group_words.groups())}]")
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
        choices=[FIXED, INFINITE, CONTINUAL],
        required=True,
    )
    parser.add_argument(
        "--dataset",
        help="Valid options: {}".format(
            [
                COMMON_GEN,
                C2GEN,
                VICTORIA_GPT35_LEADING,
                VICTORIA_GPT4O_LEADING,
                VICTORIA_LLAMA3_LEADING,
                VICTORIA_GEMINI15_LEADING,
                VICTORIA_MISTRAL8X7B_LEADING,
                VICTORIA_GPT35_NO_LEADING,
                VICTORIA_GPT4O_NO_LEADING,
                VICTORIA_LLAMA3_NO_LEADING,
                VICTORIA_GEMINI15_NO_LEADING,
                VICTORIA_MISTRAL8X7B_NO_LEADING,
                ILENIA,
                RHOPA64,
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
        "--use-legacy-semantics",
        choices=["auto", "yes", "no"],
        default="no",
    )
    parser.add_argument(
        "--split-endings",
        choices=["auto", "yes", "no"],
        default="auto",
    )
    parser.add_argument(
        "--exclude-professions",
        choices=["yes", "no"],
        default="no",
    )
    parser.add_argument(
        "--inspect-professions",
        choices=["yes", "no"],
        default="no",
    )
    parser.add_argument(
        "--use-morph",
        choices=["yes", "no"],
        required=True,
    )
    parser.add_argument(
        "--include-plurals",
        choices=["yes", "no"],
        default="no",
    )

    args = parser.parse_args()
    args.use_root = args.use_root == "yes"
    args.lower_proper_nouns = args.lower_proper_nouns == "yes"
    args.exclude_professions = args.exclude_professions == "yes"
    args.inspect_professions = args.inspect_professions == "yes"
    args.use_morph = args.use_morph == "yes"
    args.include_plurals = args.include_plurals == "yes"

    args.semantic_check = (
        None if args.semantic_check == "auto" else args.semantic_check == "yes"
    )
    args.use_legacy_semantics = (
        None
        if args.use_legacy_semantics == "auto"
        else args.use_legacy_semantics == "yes"
    )
    args.split_endings = (
        None if args.split_endings == "auto" else args.split_endings == "yes"
    )
    main(args)


if __name__ == "__main__":
    entry_point()
