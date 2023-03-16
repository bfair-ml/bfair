import gensim.downloader as api
from gensim.models import KeyedVectors
from bfair.sensors.embedding.word import WordEmbedding, register_embedding

DEBIASED_WORD2VEC_PATH = "embeddings/GoogleNews-vectors-negative300-hard-debiased.bin"


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
