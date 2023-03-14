from nltk.tokenize import sent_tokenize, word_tokenize


class TextTokenizer:
    def __init__(self, plain=True):
        self.plain = plain

    def __call__(self, text):
        sentences = sent_tokenize(text)
        words_per_sentence = [word_tokenize(sentence) for sentence in sentences]
        return (
            [word for words in words_per_sentence for word in words]
            if self.plain
            else words_per_sentence
        )


class SentenceTokenizer:
    def __call__(self, sentence):
        return word_tokenize(sentence)
