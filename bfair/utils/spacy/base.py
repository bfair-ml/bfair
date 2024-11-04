import spacy
from bfair.utils.spacy.trf_vecs import get_model_with_trf_vectors

LANGUAGE2MODEL = {
    "english": "en_core_web_sm",
    "spanish": "es_core_news_sm",
}


def get_model(*, language, model_name=None, add_transformer_vectors=False):
    if model_name is None:
        try:
            model_name = LANGUAGE2MODEL[language]
        except KeyError:
            raise ValueError(f"Language not supported: {language}.")

        if add_transformer_vectors:
            return get_model_with_trf_vectors(model_name)

        return spacy.load(model_name)
