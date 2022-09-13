from src.alignment.alignment import alignment_score
from sentence_transformers import SentenceTransformer
import nltk

nltk.download('punkt')

encoder = SentenceTransformer('distiluse-base-multilingual-cased-v1')

encoder_en = SentenceTransformer('stsb-distilbert-base')
encoder_es = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es')


def get_global_alignment(sentences1, sentences2):
    """
    Args:
        sentences1: list of sentences/short paragraphs in original language
        sentences2: list of sentences/short paragraphs in target language

    Returns:

    """
    # Multilingual embeddings
    en_embeddings = encoder.encode(sentences1)
    es_embeddings = encoder.encode(sentences2)

    # Alternatively, use different embeddings separately trained
    # (e.g. encoder_en / encoder_es)
    """
    en_embeddings = encoder_en.encode(sentences1)
    es_embeddings = encoder_es.encode(sentences2)
    """
    return alignment_score(en_embeddings, es_embeddings)