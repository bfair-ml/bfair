import math
import spacy
import numpy as np
import pandas as pd

from typing import Dict, Set
from collections import defaultdict
from itertools import repeat
from functools import partial
from statistics import mean, stdev
from tqdm import tqdm

from nltk import ngrams, word_tokenize
from nltk.corpus import stopwords

MALE = "male"
FEMALE = "female"

GERDER_PAIR_ORDER = [MALE, FEMALE]

DM_GENDER_PAIRS = [
    ("actor", "actress"),
    ("boy", "girl"),
    ("boyfriend", "girlfriend"),
    ("boys", "girls"),
    ("father", "mother"),
    ("fathers", "mothers"),
    ("gentleman", "lady"),
    ("gentlemen", "ladies"),
    ("grandson", "granddaughter"),
    ("he", "she"),
    ("him", "her"),
    ("his", "her"),
    ("husbands", "wives"),
    ("kings", "queens"),
    ("male", "female"),
    ("males", "females"),
    ("man", "woman"),
    ("men", "women"),
    ("prince", "princess"),
    ("son", "daughter"),
    ("sons", "daughters"),
    ("spokesman", "spokeswoman"),
    ("stepfather", "stepmother"),
    ("uncle", "aunt"),
    ("husband", "wife"),
    ("king", "queen"),
    ("brother", "sister"),
    ("brothers", "sisters"),
]

PENN_GENDER_PAIRS = [
    ("actor", "actress"),
    ("boy", "girl"),
    ("father", "mother"),
    ("he", "she"),
    ("him", "her"),
    ("his", "her"),
    ("male", "female"),
    ("man", "woman"),
    ("men", "women"),
    ("son", "daughter"),
    ("sons", "daughters"),
    ("spokesman", "spokeswoman"),
    ("husband", "wife"),
    ("king", "queen"),
    ("brother", "sister"),
]

WIKI_GENDER_PAIRS = [
    ("actor", "actress"),
    ("boy", "girl"),
    ("boyfriend", "girlfriend"),
    ("boys", "girls"),
    ("father", "mother"),
    ("fathers", "mothers"),
    ("gentleman", "lady"),
    ("gentlemen", "ladies"),
    ("grandson", "granddaughter"),
    ("he", "she"),
    ("hero", "heroine"),
    ("him", "her"),
    ("his", "her"),
    ("husband", "wife"),
    ("husbands", "wives"),
    ("king", "queen"),
    ("kings", "queens"),
    ("male", "female"),
    ("males", "females"),
    ("man", "woman"),
    ("men", "women"),
    ("mr.", "mrs."),
    ("prince", "princess"),
    ("son", "daughter"),
    ("sons", "daughters"),
    ("spokesman", "spokeswoman"),
    ("stepfather", "stepmother"),
    ("uncle", "aunt"),
    # ("wife", "husband"),
    ("king", "queen"),
]

ALL_GENDER_PAIRS = list(
    {
        pair
        for pairs in [DM_GENDER_PAIRS, PENN_GENDER_PAIRS, WIKI_GENDER_PAIRS]
        for pair in pairs
    }
)


class EnglishGenderedWords:
    def __init__(
        self,
        sources=(DM_GENDER_PAIRS, PENN_GENDER_PAIRS, WIKI_GENDER_PAIRS),
        order=GERDER_PAIR_ORDER,
    ):
        group_words = defaultdict(set)
        for pairs in sources:
            for i, gender in enumerate(order):
                group_words[gender].update((p[i] for p in pairs))
        self.male = group_words[MALE]
        self.female = group_words[FEMALE]

    def get_male_words(self):
        return self.male

    def get_female_words(self):
        return self.female


class FixedContext:
    def __init__(self, window_size=10):
        self.window_size = window_size

    def __call__(self, tokens):
        for window in ngrams(
            tokens,
            2 * self.window_size + 1,
            pad_left=True,
            pad_right=True,
        ):
            middle = self.window_size
            center = window[middle]

            if center is None:
                continue

            get_context = partial(self.weighted, window, middle)
            yield center, get_context

    def weighted(self, window, middle):
        window = (word for word in window if word is not None)
        return zip(window, repeat(1))


class InfiniteContext(FixedContext):
    def __init__(self):
        super().__init__(window_size=200)
        self.step = 0.95

    def weighted(self, window, middle):
        for i, word in enumerate(window):
            if word is None:
                continue
            distance = abs(middle - i)
            weight = pow(self.step, distance)
            yield word, weight


class BiasScore:
    LANGUAGE2MODEL = {
        "english": "en_core_web_sm",
        "spanish": "es_core_news_sm",
    }

    S_RATIO = "count_disparity"
    S_LOG = "log_score"

    def __init__(
        self,
        *,
        language,
        group_words: Dict[str, Set],
        context,
        scoring_modes,
        use_root,
        tokenizer=None,
        remove_stopwords=True,
        remove_groupwords=True,
        merge_paragraphs=False,
    ):
        self.language = language
        self.group_words = group_words
        self.context = context
        self.scoring_modes = scoring_modes
        self.remove_stopwords = remove_stopwords
        self.remove_groupwords = remove_groupwords
        self.merge_paragraphs = merge_paragraphs
        self.all_group_words = {w for words in group_words.values() for w in words}
        self.tokenizer = (
            self._get_default_tokenizer(language, use_root, self.all_group_words)
            if tokenizer is None
            else tokenizer
        )
        self.stopwords = stopwords.words(language) if remove_stopwords else None

    @classmethod
    def _get_default_tokenizer(cls, language, use_root, relevant):
        if not use_root:
            return partial(word_tokenize, language=language)

        if language not in cls.LANGUAGE2MODEL:
            raise ValueError(language)

        nlp = spacy.load(cls.LANGUAGE2MODEL[language])

        def tokenizer(text):
            return [
                (
                    token.text
                    if token.pos_ == "PROPN"
                    else token.lower_
                    if token.pos_ in ("PRON", "DET") or token.lower_ in relevant
                    else token.lemma_.lower()
                )
                for token in nlp(text)
            ]

        return tokenizer

    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        if not self.merge_paragraphs:
            texts = [paragraph for text in texts for paragraph in text.splitlines()]

        word2counts = defaultdict(lambda: defaultdict(int))
        for text in tqdm(texts, desc="Computing counts"):
            self._aggregate_counts(
                word2counts,
                self.get_counts(
                    text=text,
                    group_words=self.group_words,
                    context=self.context,
                    tokenizer=self.tokenizer,
                ),
            )

        if self.remove_stopwords or self.remove_groupwords:
            word2counts = self.drop_words(
                word2counts,
                remove_stopwords=self.remove_stopwords,
                stopwords=self.stopwords,
                remove_groupwords=self.remove_groupwords,
                all_group_words=self.all_group_words,
            )
        return self.compute_scores(
            word2counts, self.group_words.keys(), self.scoring_modes
        )

    @classmethod
    def _aggregate_counts(cls, total_counts, new_counts):
        for word, counts in new_counts.items():
            total_count = total_counts[word]
            for group, count in counts.items():
                total_count[group] += count

    @classmethod
    def get_counts(
        cls,
        *,
        text,
        group_words: Dict[str, Set],
        context,
        tokenizer=str.split,
    ):
        word2counts = defaultdict(lambda: defaultdict(int))
        tokens = tokenizer(text)
        for center, get_context in context(tokens):
            for group, words in group_words.items():
                if center not in words:
                    continue
                for word, weight in get_context():
                    word2counts[word][group] += weight

        return word2counts

    @classmethod
    def drop_words(
        cls,
        word2counts,
        *,
        remove_stopwords,
        stopwords,
        remove_groupwords,
        all_group_words,
    ):
        return {
            word: counts
            for word, counts in tqdm(word2counts.items(), desc="Dropping words")
            if not (
                remove_stopwords
                and word in stopwords
                or remove_groupwords
                and word in all_group_words
            )
        }

    @classmethod
    def compute_scores(cls, word2counts, groups, scoring_modes):
        result = {}
        for scoring_mode in scoring_modes:
            scores = [
                (
                    word,
                    cls.compute_score_for_word(word, word2counts, groups, scoring_mode),
                )
                for word in tqdm(word2counts, desc=f"Scoring words ({scoring_mode})")
            ]
            not_infinity = [s for _, s in scores if not math.isinf(s)]
            result[scoring_mode] = (
                mean(not_infinity),
                stdev(not_infinity),
                pd.DataFrame.from_records(
                    scores,
                    columns=["words", scoring_mode],
                ).set_index("words"),
            )

        for group in groups:
            scoring_mode = f"count ({group})"
            not_infinity = [
                c[group] for c in word2counts.values() if not math.isinf(c[group])
            ]
            result[scoring_mode] = (
                mean(not_infinity),
                stdev(not_infinity),
                pd.DataFrame.from_records(
                    [(word, counts[group]) for word, counts in word2counts.items()],
                    columns=["words", scoring_mode],
                ).set_index("words"),
            )
        return result

    @classmethod
    def compute_score_for_word(cls, word, word2counts, groups, scoring_mode):
        if cls.S_RATIO == scoring_mode:
            return cls.compute_count_disparity_for_word(word, word2counts, groups)
        elif cls.S_LOG == scoring_mode:
            return cls.compute_log_score_for_word(word, word2counts, groups)
        else:
            raise ValueError(scoring_mode)

    @classmethod
    def compute_count_disparity_for_word(cls, word, word2counts, groups):
        counts = word2counts[word]
        if len(counts) != len(groups):
            return 1

        min_value = min(counts.values())
        max_value = max(counts.values())
        return 1 - min_value / max_value if max_value else 1

    @classmethod
    def compute_log_score_for_word(cls, word, word2counts, groups):
        if len(groups) != 2:
            raise ValueError("Only usable with two classes.")
        first, second, *_ = groups
        prob_first = cls.compute_word_prob(word, first, word2counts)
        prob_second = cls.compute_word_prob(word, second, word2counts)

        if prob_first and prob_second:
            return np.log2(prob_first / prob_second)  # I am pretty sure it is log 2
        elif prob_first:
            return float("+inf")
        elif prob_second:
            return float("-inf")
        else:
            return 0

    @classmethod
    def compute_word_prob(cls, word, group, word2counts):
        count_w_g = word2counts[word][group]
        total_count_w_g = sum(counts[group] for counts in word2counts.values())

        numerator = count_w_g / total_count_w_g if total_count_w_g else 0
        return numerator
