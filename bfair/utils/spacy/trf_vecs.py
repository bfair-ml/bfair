import spacy
import numpy as np

from spacy.language import Language
from spacy.tokens import Doc


@Language.factory("trf_vectors")
class TrfContextualVectors:
    """
    Spacy pipeline which add transformer vectors to each token based on user hooks.

    https://spacy.io/usage/processing-pipelines#custom-components-user-hooks
    https://github.com/explosion/spaCy/discussions/6511
    """

    def __init__(self, nlp: Language, name: str):
        self.name = name
        Doc.set_extension("trf_token_vecs", default=None)

    def __call__(self, sdoc):
        # inject hooks from this class into the pipeline
        if type(sdoc) == str:
            sdoc = self._nlp(sdoc)

        if not sdoc:
            return sdoc

        # pre-calculate all vectors for every token:

        # calculate groups for spacy token boundaries in the trf vectors
        vec_idx_splits = np.cumsum(sdoc._.trf_data.align.lengths)
        # get transformer vectors and reshape them into one large continous tensor
        trf_vecs = sdoc._.trf_data.tensors[0].reshape(-1, 768)
        # calculate mapping groups from spacy tokens to transformer vector indices
        vec_idxs = np.split(sdoc._.trf_data.align.dataXd, vec_idx_splits)

        # take sum of mapped transformer vector indices for spacy vectors
        vecs = np.stack([trf_vecs[idx].sum(0) for idx in vec_idxs[:-1]])
        sdoc._.trf_token_vecs = vecs

        sdoc.user_token_hooks["vector"] = self.vector
        # sdoc.user_span_hooks["vector"] = self.vector
        # sdoc.user_hooks["vector"] = self.vector
        sdoc.user_token_hooks["has_vector"] = self.has_vector
        sdoc.user_token_hooks["similarity"] = self.similarity
        # sdoc.user_span_hooks["similarity"] = self.similarity
        # sdoc.user_hooks["similarity"] = self.similarity
        return sdoc

    def vector(self, token):
        return token.doc._.trf_token_vecs[token.i]

    def has_vector(self, token):
        return True

    def similarity(self, token, other):  # JP
        x = token.vector
        y = other.vector
        return 1 - np.dot(x, y) / (np.dot(x, x) ** 0.5 * np.dot(y, y) ** 0.5)


def get_model_with_trf_vectors(model_name):
    """
    `model_name`: transformer-based model.
    """
    nlp = spacy.load(model_name)
    nlp.add_pipe("trf_vectors")
    return nlp
