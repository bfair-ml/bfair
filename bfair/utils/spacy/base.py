import spacy


try:
    from .trf_vecs import get_model_with_trf_vectors
except:
    print(
        "[BFAIR ⚠️]: Failed to load the semantic module.",
        "You can safely ignore this message if the module is not required for your current use.",
    )
    # MOCK
    def get_model_with_trf_vectors(model_name):
        raise Exception(
            "Attempting to use the semantic module, but it was not loaded correctly."
        )


LANGUAGE2MODEL = {
    "english": "en_core_web_sm",
    "spanish": "es_core_news_sm",
    "valencian": "ca_core_news_sm",
    "catalan": "ca_core_news_sm",
}


def get_model(*, model_name=None, language=None, add_transformer_vectors=False, remote_url=None):
    if remote_url is not None and add_transformer_vectors:
        raise ValueError("Remote models with transformer vectors are not supported.")

    if model_name is None:
        try:
            model_name = LANGUAGE2MODEL[language]
        except KeyError:
            raise ValueError(f"Language not supported: {language}.")

    if remote_url is not None:
        from rspacy import RemoteSpacy
        return RemoteSpacy(remote_url)

    return (
        get_model_with_trf_vectors(model_name)
        if add_transformer_vectors
        else spacy.load(model_name)
    )
