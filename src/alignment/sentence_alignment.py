from src.alignment.word_alignment import word_alignment, anchors_alignment
from src.util.word_embeddings import multilingual_cos_sim
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
import numpy as np


encoder = SentenceTransformer('distiluse-base-multilingual-cased-v1')
print("Embeddings loaded")


def sentence_alignment(s1, s2):
    """
    Args:
        s1: sentence 1 (in original language)
        s2: sentence 2 (in target language)

    Returns:
        sentence alignment
    """
    tuples = get_word_correspondence_align(s1, s2, threshold=anchors_alignment)
    words1 = word_tokenize(s1)
    scores = [score for (_, _, score) in tuples]
    return sum(scores)/len(words1)


def get_word_correspondence_align(s1, s2, threshold=0.3):
    """
    Args:
        s1:
        s2:
        threshold: similarity threshold, if metric used is multilingual cos sim
                    OR alignment of anchors if alignment is used

    Returns:
        Correspondence of word: List of tuples [(w, correspondent word, score)]
    """
    words1 = word_tokenize(s1)
    words2 = word_tokenize(s2)
    word_tuples = []
    taken = []
    for w1 in words1:
        max_score = 0
        max_word = ""
        for w2 in words2:
            if w2 not in taken:
                score = word_alignment(w1, w2)
                """
                Alternatively: use multilingual cos sim
                """
                # score = multilingual_cos_sim(w1, w2)
                if score > max_score:
                    max_score = score
                    max_word = w2

        if max_score > threshold:
            word_tuples.append((w1, max_word, max_score))
            taken.append(max_word)
    return word_tuples


if __name__ == '__main__':
    en_sentence = "What do you mean"
    es_sentence = "Que quieres decir"
    print(sentence_alignment(en_sentence, es_sentence))