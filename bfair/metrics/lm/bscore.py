import numpy as np

from typing import Dict, Set
from collections import defaultdict
from itertools import repeat
from functools import partial
from statistics import mean, stdev

from nltk import ngrams
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
    ("wife", "husband"),
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
    ("wife", "husband"),
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
    ("wife", "husband"),
    ("king", "queen"),
]


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

    S_RATIO = "count_ratio"
    S_LOG = "log_score"

    def __init__(
        self,
        *,
        language,
        group_words: Dict[str, Set],
        context,
        scoring_modes,
        tokenizer=str.split,
        remove_stopwords=True,
        remove_groupwords=True,
    ):
        self.language = language
        self.group_words = group_words
        self.context = context
        self.scoring_modes = scoring_modes
        self.tokenizer = tokenizer
        self.remove_stopwords = remove_stopwords
        self.remove_groupwords = remove_groupwords

        self.stopwords = stopwords.words(language) if remove_stopwords else None
        self.all_group_words = {w for words in group_words.values() for w in words}

    def __call__(self, text):
        word2counts = self.get_counts(
            text=text,
            group_words=self.group_words,
            context=self.context,
            tokenizer=self.tokenizer,
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
            for word, counts in word2counts.items()
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
                cls.compute_score_for_word(word, word2counts, groups, scoring_mode)
                for word in word2counts
            ]
            result[scoring_mode] = (
                mean(scores),
                stdev(scores),
                list(zip(word2counts, scores)),
            )
        return result

    @classmethod
    def compute_score_for_word(cls, word, word2counts, groups, scoring_mode):
        if cls.S_RATIO == scoring_mode:
            return cls.compute_count_ratio_for_word(word, word2counts, groups)
        elif cls.S_LOG == scoring_mode:
            return cls.compute_log_score_for_word(word, word2counts, groups)
        else:
            raise ValueError(scoring_mode)

    @classmethod
    def compute_count_ratio_for_word(cls, word, word2counts, groups):
        counts = word2counts[word]
        if len(counts) != len(groups):
            return 0

        min_value = min(counts.values())
        max_value = max(counts.values())
        return 1 - min_value / max_value if max_value else 0

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
            return 1
        elif prob_second:
            return -1
        else:
            return 0

    @classmethod
    def compute_word_prob(cls, word, group, word2counts):
        count_w_g = word2counts[word][group]
        total_count_w_g = sum(counts[group] for counts in word2counts.values())

        numerator = count_w_g / total_count_w_g if total_count_w_g else 0
        return numerator
