import gensim.downloader as api
from gensim.models import KeyedVectors
from bfair.sensors.text.embedding.word import WordEmbedding, register_embedding
from bfair.utils.spacy import get_model
from bfair.envs import DEBIASED_WORD2VEC_PATH


class GensimEmbedding(WordEmbedding):
    def __init__(self):
        self.wv = None

    def __getitem__(self, word: str):
        return self.wv[word]

    def similarity(self, word1: str, word2: str):
        try:
            return self.wv.similarity(word1, word2)
        except KeyError:
            return 0

    def get_ready(self):
        if self.wv is None:
            self.wv = self._load_wv()
        return self

    def _load_wv(self):
        raise NotImplementedError()


class GensimPretrainedEmbedding(GensimEmbedding):
    def __init__(self, name):
        """Creates a wrapper for any gensim model.
        - `name`: Any model name that is listed [here](https://github.com/RaRe-Technologies/gensim-data).
        """
        self.name = name
        super().__init__()

    def _load_wv(self):
        model = api.load(self.name)
        return model.wv


@register_embedding("english", "word2vec")
class Word2VecEmbedding(GensimPretrainedEmbedding):
    def __init__(self):
        super().__init__("word2vec-google-news-300")


@register_embedding("english", "word2vec-debiased")
class DebiasedWord2VecEmbedding(GensimEmbedding):
    def _load_wv(self):
        return KeyedVectors.load_word2vec_format(
            DEBIASED_WORD2VEC_PATH,
            binary=True,
        )


class SpacyEmbedding(WordEmbedding):
    def __init__(self, model_name=None, language=None, add_transformer_vectors=None):
        if model_name is None and language is None:
            raise ValueError("Provide `model_name` or `language`.")

        self.model_name = model_name
        self.language = language
        self.add_transformer_vectors = add_transformer_vectors
        self.nlp = None

    def similarity(self, word1: str, word2: str):
        token1 = self.nlp(word1)[0]
        token2 = self.nlp(word2)[0]
        return token1.similarity(token2)

    def __getitem__(self, word: str):
        token = self.nlp(word)[0]
        return token.vector

    def get_ready(self):
        self.nlp = get_model(
            model_name=self.model_name,
            language=self.language,
            add_transformer_vectors=self.add_transformer_vectors,
        )
        return self


@register_embedding("english", "spacy-sm")
class SpacyEnglishSmEmbedding(SpacyEmbedding):
    def __init__(self):
        super().__init__("en_core_web_sm")

    def get_ready(self):
        print("[BFAIR ⚠️] This model only have syntantic vectors (not semantic).")
        return super().get_ready()


@register_embedding("english", "spacy-md")
class SpacyEnglishMdEmbedding(SpacyEmbedding):
    def __init__(self):
        super().__init__("en_core_web_md")


@register_embedding("english", "spacy-lg")
class SpacyEnglishLgEmbedding(SpacyEmbedding):
    def __init__(self):
        super().__init__("en_core_web_lg")


@register_embedding("english", "spacy-trf")
class SpacyEnglishTrfEmbedding(SpacyEmbedding):
    def __init__(self):
        super().__init__("en_core_web_trf", add_transformer_vectors=True)


@register_embedding("spanish", "spacy-sm")
class SpacySpanishSmEmbedding(SpacyEmbedding):
    def __init__(self):
        super().__init__("es_core_news_sm")

    def get_ready(self):
        print("[BFAIR ⚠️] This model only have syntantic vectors (not semantic).")
        return super().get_ready()


@register_embedding("spanish", "spacy-md")
class SpacySpanishMdEmbedding(SpacyEmbedding):
    def __init__(self):
        super().__init__("es_core_news_md")


@register_embedding("spanish", "spacy-lg")
class SpacySpanishLgEmbedding(SpacyEmbedding):
    def __init__(self):
        super().__init__("es_core_news_lg")


@register_embedding("spanish", "spacy-trf")
class SpacySpanishTrfEmbedding(SpacyEmbedding):
    def __init__(self):
        super().__init__("es_dep_news_trf", add_transformer_vectors=True)

@register_embedding("valencian", "spacy-sm")
@register_embedding("catalan", "spacy-sm")
class SpacyCatalanSmEmbedding(SpacyEmbedding):
    def __init__(self):
        super().__init__("ca_core_news_sm")

    def get_ready(self):
        print("[BFAIR ⚠️] This model only have syntantic vectors (not semantic).")
        return super().get_ready()
    
@register_embedding("valencian", "spacy-md")
@register_embedding("catalan", "spacy-md")
class SpacyCatalanMdEmbedding(SpacyEmbedding):
    def __init__(self):
        super().__init__("ca_core_news_md")

@register_embedding("valencian", "spacy-lg")
@register_embedding("catalan", "spacy-lg")
class SpacyCatalanLgEmbedding(SpacyEmbedding):
    def __init__(self):
        super().__init__("ca_core_news_lg")

@register_embedding("valencian", "spacy-trf")
@register_embedding("catalan", "spacy-trf")
class SpacyCatalanTrfEmbedding(SpacyEmbedding):
    def __init__(self):
        super().__init__("ca_dep_news_trf", add_transformer_vectors=True)