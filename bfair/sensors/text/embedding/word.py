class EmbeddingLoader:
    _register = {}

    @classmethod
    def load(cls, *, language: str, source: str) -> "WordEmbedding":
        try:
            embedding = cls._register[language, source]
            return embedding.get_ready()
        except KeyError:
            raise KeyError(
                f"No embedding is registered for language `{language}` and source `{source}`."
            )

    @classmethod
    def register(cls, embedding: "WordEmbedding", *, language: str, source: str):
        cls._register[language, source] = embedding
        return cls


class WordEmbedding:
    def similarity(self, word1: str, word2: str):
        raise NotImplementedError()

    def u_similarity(self, word1: str, word2: str):
        similarity = self.similarity(word1, word2)
        return abs(similarity)

    def __getitem__(self, word: str):
        raise NotImplementedError()

    def get_ready(self):
        return self


def register_embedding(language: str, source: str):
    """Useful for registering `WordEmdeddings` using the default initialization.

    Example:
    ```python
    @register_embedding("english", "word2vec")
    class Word2Vec(WordEmbedding):
        # ...
    ```
    """

    def do_registration(embedding_init):
        instance = embedding_init()
        EmbeddingLoader.register(instance, language=language, source=source)
        return embedding_init

    return do_registration
