from nltk.tokenize import sent_tokenize, word_tokenize


class Tokenizer:
    def __call__(self, string):
        raise NotImplementedError()


class TextTokenizer(Tokenizer):
    def __init__(
        self,
        plain=True,
        language="english",
    ):
        self.plain = plain
        self.language = language

    def __call__(self, text):
        sentences = sent_tokenize(text, self.language)
        words_per_sentence = [word_tokenize(sentence) for sentence in sentences]
        return (
            [word for words in words_per_sentence for word in words]
            if self.plain
            else words_per_sentence
        )


class TextSplitter(Tokenizer):
    def __init__(self, language="english"):
        self.language = language

    def __call__(self, text):
        return sent_tokenize(text, self.language)


class SentenceTokenizer(Tokenizer):
    def __init__(self, language="english"):
        self.language = language

    def __call__(self, sentence):
        return word_tokenize(sentence, self.language)
