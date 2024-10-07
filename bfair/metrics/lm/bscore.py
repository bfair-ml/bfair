import math
import spacy
import numpy as np
import pandas as pd

from typing import Dict, Set, Sequence, Callable
from collections import defaultdict
from itertools import repeat
from functools import partial
from statistics import mean, stdev
from tqdm import tqdm

from nltk import ngrams, word_tokenize
from nltk.corpus import stopwords

from bfair.metrics.lm.words import GroupWords
from bfair.metrics.lm.semantics import (
    ITokenChecker,
    PersonCheckerForSpanish,
    DummyChecker,
    get_model_with_trf_vectors,
)
from bfair.metrics.lm.endings import spanish_split_gender_endings


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
    def __init__(self, step=0.95):
        super().__init__(window_size=200)
        self.step = step

    def weighted(self, window, middle):
        for i, word in enumerate(window):
            if word is None:
                continue
            if i == middle:  # to achieve (..., 0.95, 1, 1, 1, 0.95, ...)
                yield word, 1
            else:
                distance = abs(middle - i)
                weight = pow(self.step, distance - 1)
                yield word, weight


class ContinuousContext(InfiniteContext):
    DISRUPTION_PENALIZATION = {
        punct: penalization
        for symbols, penalization in (
            (".¿?¡!", 0.25),
            (";«»()[]", 0.50),
            (("...",), 0.75),
            (",:-•", 0.75),
            ("/", 0.90),
        )
        for punct in symbols
    }

    def __init__(self, step=0.95, disruption_penalization=None):
        super().__init__(step)
        self.disruption_penalization = (
            self.DISRUPTION_PENALIZATION
            if disruption_penalization is None
            else disruption_penalization
        )

    def weighted(self, window, middle):
        yield from self._sequential_weighting(reversed(window[:middle]))
        yield from self._sequential_weighting(window[middle:])

    def _sequential_weighting(self, sequence):
        weight = 1
        for word in sequence:
            if word is None:
                break
            yield word, weight
            try:
                weight *= self.disruption_penalization[word]
            except KeyError:
                weight *= self.step


class BiasScore:
    LANGUAGE2MODEL = {
        "english": "en_core_web_sm",
        "spanish": "es_core_news_sm",
    }

    LANGUAGE2SEMANTIC = {
        "spanish": "es_dep_news_trf",
    }

    LANGUAGE2ENDINGS = {"spanish": spanish_split_gender_endings}

    S_RATIO = "count_disparity"
    S_LOG = "log_score"

    DISCRETE = 0.05

    EPSILON = np.finfo(float).eps

    def __init__(
        self,
        *,
        language,
        group_words: GroupWords,
        context,
        scoring_modes,
        use_root,
        tokenizer=None,
        remove_stopwords=True,
        remove_groupwords=True,
        merge_paragraphs=False,
        lower_proper_nouns=False,
        semantic_check=None,
        use_legacy_semantics=None,
        split_endings=None,
        group_words_to_inspect: GroupWords = None,
    ):
        self.language = language
        self.group_words = group_words
        self.context = context
        self.scoring_modes = scoring_modes
        self.remove_stopwords = remove_stopwords
        self.remove_groupwords = remove_groupwords
        self.merge_paragraphs = merge_paragraphs

        if semantic_check is None:
            semantic_check = language in self.LANGUAGE2SEMANTIC
        if use_legacy_semantics is None:
            use_legacy_semantics = language not in self.LANGUAGE2SEMANTIC

        nlp = self.get_nlp(language, semantic_check, use_legacy_semantics)

        preprocessings = []
        if split_endings is None:
            split_endings = language in self.LANGUAGE2ENDINGS
        if split_endings:
            splitter = self._get_endings_splitter(language)
            preprocessings.append(splitter)

        self.tokenizer = (
            self._get_default_tokenizer(
                nlp,
                use_root,
                lower_proper_nouns,
                group_words,
                preprocessings,
            )
            if tokenizer is None
            else tokenizer
        )
        self.person_checker = (
            self._get_person_checker(language, nlp)
            if semantic_check
            else DummyChecker()
        )
        self.lemmatizer = lambda word: nlp(word)[0].lemma_

        self.stopwords = stopwords.words(language) if remove_stopwords else None
        self.group_words_to_inspect = group_words_to_inspect

    @classmethod
    def get_nlp(cls, language, semantic_check, use_legacy_semantics):
        if semantic_check and use_legacy_semantics:
            print("[BFAIR ⚠️] Using legacy semantics.")
        source = (
            cls.LANGUAGE2SEMANTIC
            if semantic_check and not use_legacy_semantics
            else cls.LANGUAGE2MODEL
        )
        try:
            model_name = source[language]
            if model_name == "es_dep_news_trf":
                return get_model_with_trf_vectors(model_name)
            return spacy.load(model_name)
        except KeyError:
            raise ValueError(f"Not supported: {language}(semantic={semantic_check})")

    @classmethod
    def _get_person_checker(cls, language: str, nlp) -> ITokenChecker:
        if language == "spanish":
            return PersonCheckerForSpanish(nlp)
        raise ValueError(
            f"{language.title()} does not currently support semantic checking."
        )

    @classmethod
    def _get_endings_splitter(cls, language: str) -> Callable[[str], str]:
        try:
            return cls.LANGUAGE2ENDINGS[language]
        except KeyError:
            raise ValueError(
                f"{language.title()} does not currently support split endings."
            )

    @classmethod
    def _get_default_tokenizer(
        cls,
        nlp,
        use_root,
        lower_proper_nouns,
        group_words: GroupWords,
        preprocessings: Sequence[Callable[[str], str]],
    ):
        def tokenizer(text):
            for preprocessor in preprocessings:
                text = preprocessor(text)

            return [
                (
                    # TEXT
                    token.text
                    if (
                        token.pos_ == "PROPN"
                        and (not lower_proper_nouns or group_words.includes(token.text))
                    )
                    else token.lower_
                    if (
                        not use_root
                        or token.pos_ in ("PRON", "DET")
                        or group_words.includes(token.lower_, token.pos_)
                        or group_words.includes(token.lemma_.lower(), token.pos_)
                    )
                    else token.lemma_.lower(),
                    # POS
                    token.pos_,
                    # TOKEN
                    token,
                )
                for token in nlp(text)
                if not token.is_space
            ]

        return tokenizer

    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        if not self.merge_paragraphs:
            texts = [paragraph for text in texts for paragraph in text.splitlines()]

        word2counts = defaultdict(lambda: defaultdict(int))
        word2discrete = defaultdict(lambda: defaultdict(int))
        word2matches = defaultdict(lambda: defaultdict(set))
        word2occurrence = defaultdict(int)

        for text in tqdm(texts, desc="Computing counts"):
            (
                local_word2counts,
                local_word2discrete,
                local_word2match,
                local_word2occurrence,
            ) = self.get_counts(
                text=text,
                group_words=self.group_words,
                context=self.context,
                tokenizer=self.tokenizer,
                person_checker=self.person_checker,
            )

            self._aggregate_counts(word2counts, local_word2counts)
            self._aggregate_counts(word2discrete, local_word2discrete)
            self._aggregate_matches(word2matches, local_word2match)
            self._aggregate_numbers(word2occurrence, local_word2occurrence)

        if self.remove_stopwords or self.remove_groupwords:
            word2counts = self.drop_words(
                word2counts,
                remove_stopwords=self.remove_stopwords,
                stopwords=self.stopwords,
                remove_groupwords=self.remove_groupwords,
                group_words=self.group_words,
                group_words_to_inspect=self.group_words_to_inspect,
            )
            word2discrete = {word: word2discrete[word] for word in word2counts}
            word2matches = {word: word2matches[word] for word in word2counts}
            word2occurrence = {word: word2occurrence[word] for word in word2counts}

        if self.group_words_to_inspect is not None:
            (
                word2counts,
                word2discrete,
                word2matches,
                word2occurrence,
            ) = self.merge_words_to_inspect(
                word2counts,
                word2discrete,
                word2matches,
                word2occurrence,
                group_words_to_inspect=self.group_words_to_inspect,
                lemmatizer=self.lemmatizer,
            )

        return self.compute_scores(
            word2counts,
            word2discrete,
            word2matches,
            word2occurrence,
            self.group_words.groups(),
            self.scoring_modes,
        )

    @classmethod
    def _aggregate_counts(cls, total_counts, new_counts):
        for word, counts in new_counts.items():
            total_count = total_counts[word]
            for group, count in counts.items():
                total_count[group] += count

    @classmethod
    def _aggregate_matches(cls, total, new):
        for word, collection in new.items():
            total_matches = total[word]
            for group, matches in collection.items():
                total_matches[group].update(matches)

    @classmethod
    def _aggregate_numbers(cls, total, new):
        for word, number in new.items():
            total[word] += number

    @classmethod
    def get_counts(
        cls,
        *,
        text,
        group_words: GroupWords,
        context,
        tokenizer,
        person_checker=DummyChecker(),
    ):
        word2counts = defaultdict(lambda: defaultdict(int))
        word2discrete = defaultdict(lambda: defaultdict(int))
        word2matches = defaultdict(lambda: defaultdict(set))
        word2occurrence = defaultdict(int)

        tokens = tokenizer(text)
        for (center, center_pos, center_token), get_context in context(tokens):
            word2occurrence[center] += 1
            for group, words in group_words.items():
                if not words.includes(
                    center, center_pos
                ) or not person_checker.check_token(center_token):
                    continue

                for (word, word_pos, word_token), weight in get_context():
                    if word_pos in ("PUNCT", "NUM", "SYM") or word.isdigit():
                        continue

                    word2counts[word][group] += weight

                    if weight > cls.DISCRETE:
                        word2discrete[word][group] += 1

                        offset = 3
                        min_token = min(center_token, word_token, key=lambda x: x.i)
                        max_token = max(center_token, word_token, key=lambda x: x.i)
                        min_word, max_word = (
                            (center, word)
                            if center_token == min_token
                            else (word, center)
                        )

                        if center_token.i == word_token.i:
                            core = f"«{center_token.text}->{center}»"
                        else:
                            core = "[[{}->{}]] {} [[{}->{}]]".format(
                                # WORD or CENTER
                                min_token.text,
                                # LEX
                                min_word,
                                # MIDDLE
                                min_token.doc[min_token.i + 1 : max_token.i]
                                if max_token.i > min_token.i + 1
                                else "",
                                # WORD or CENTER
                                max_token.text,
                                # LEX
                                max_word,
                            )

                        highlighted = "{}{} {} {}{}".format(
                            # START
                            "... " if min_token.i - offset > 0 else "",
                            # PRE
                            min_token.doc[max(min_token.i - offset, 0) : min_token.i]
                            if min_token.i > 0
                            else "",
                            # CORE
                            core,
                            # POST
                            max_token.doc[max_token.i + 1 : max_token.i + offset]
                            if len(max_token.doc) > max_token.i + 1
                            else "",
                            # END
                            " ..." if max_token.i + offset < len(max_token.doc) else "",
                        )

                        word2matches[word][group].add(highlighted)

        return word2counts, word2discrete, word2matches, word2occurrence

    @classmethod
    def drop_words(
        cls,
        word2counts,
        *,
        remove_stopwords,
        stopwords,
        remove_groupwords,
        group_words: GroupWords,
        group_words_to_inspect: GroupWords,
    ):
        return {
            word: counts
            for word, counts in tqdm(word2counts.items(), desc="Dropping words")
            if not (
                remove_stopwords
                and word in stopwords
                or remove_groupwords
                and group_words.includes(word)
                and (
                    group_words_to_inspect is None
                    or not group_words_to_inspect.includes(word)
                )
            )
        }

    @classmethod
    def merge_words_to_inspect(
        cls,
        word2counts,
        word2discrete,
        word2matches,
        word2occurrence,
        *,
        group_words_to_inspect: GroupWords,
        lemmatizer,
    ):
        merged_word2counts = defaultdict(lambda: defaultdict(int))
        merged_word2discrete = defaultdict(lambda: defaultdict(int))
        merged_word2matches = defaultdict(lambda: defaultdict(set))
        merged_word2occurrence = defaultdict(int)

        for word in word2counts:
            if not group_words_to_inspect.includes(word):
                merged_word2counts[word] = word2counts[word]
                merged_word2discrete[word] = word2discrete[word]
                merged_word2matches[word] = word2matches[word]
                merged_word2occurrence[word] = word2occurrence[word]
            else:
                lemma = lemmatizer(word)
                cls.combine_by_sum(merged_word2counts[lemma], word2counts[word])
                cls.combine_by_sum(merged_word2discrete[lemma], word2discrete[word])
                cls.combine_by_or(merged_word2matches[lemma], word2matches[word])
                merged_word2occurrence[lemma] += word2occurrence[word]

        return (
            merged_word2counts,
            merged_word2discrete,
            merged_word2matches,
            merged_word2occurrence,
        )

    @classmethod
    def combine_by_sum(cls, merged: defaultdict, new: defaultdict):
        for gender, count in new.items():
            merged[gender] += count

    @classmethod
    def combine_by_or(cls, merged: defaultdict, new: defaultdict):
        for gender, count in new.items():
            merged[gender] |= count

    @classmethod
    def compute_scores(
        cls,
        word2counts,
        word2discrete,
        word2matches,
        word2occurrence,
        groups,
        scoring_modes,
    ):
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

        for group in groups:
            scoring_mode = f"discrete[{cls.DISCRETE}] ({group})"
            not_infinity = [
                c[group] for c in word2discrete.values() if not math.isinf(c[group])
            ]
            result[scoring_mode] = (
                mean(not_infinity),
                stdev(not_infinity),
                pd.DataFrame.from_records(
                    [(word, counts[group]) for word, counts in word2discrete.items()],
                    columns=["words", scoring_mode],
                ).set_index("words"),
            )

        result["global-count"] = (
            mean(word2occurrence.values()),
            stdev(word2occurrence.values()),
            pd.DataFrame.from_records(
                [(word, count) for word, count in word2occurrence.items()],
                columns=["words", "global-count"],
            ).set_index("words"),
        )

        for group in groups:
            scoring_mode = f"matches ({group})"
            result[scoring_mode] = (
                None,
                None,
                pd.DataFrame.from_records(
                    [(word, matches[group]) for word, matches in word2matches.items()],
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
        prob_first = cls.compute_word_prob(word, first, word2counts) + cls.EPSILON
        prob_second = cls.compute_word_prob(word, second, word2counts) + cls.EPSILON

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
        # denominator = ... # Compared to Co-Occurrence Bias score: count(group) / sum(count(non-seed-words))
        return numerator
