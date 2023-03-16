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
        return self.wv.similarity(word1, word2)


class GensimPretrainedEmbedding(GensimEmbedding):
    def __init__(self, name):
        """Creates a wrapper for any gensim model.
        - `name`: Any model name that is listed [here](https://github.com/RaRe-Technologies/gensim-data).
        """
        self.name = name
        super().__init__()

    def get_ready(self):
        model = api.load(self.name)
        self.wv = model.wv
        return self


@register_embedding("english", "word2vec")
class Word2VecEmbedding(GensimPretrainedEmbedding):
    def __init__(self):
        super().__init__("word2vec-google-news-300")


@register_embedding("english", "word2vec-debiased")
class DebiasedWord2VecEmbedding(GensimEmbedding):
    def get_ready(self):
        self.wv = KeyedVectors.load_word2vec_format(
            DEBIASED_WORD2VEC_PATH,
            binary=True,
        )
        return self
