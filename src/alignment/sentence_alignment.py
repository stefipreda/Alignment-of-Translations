from src.alignment.word_alignment import word_alignment, multilingual_cos_sim
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from nltk.tokenize import word_tokenize
import numpy as np


encoder = SentenceTransformer('distiluse-base-multilingual-cased-v1')
print("Embeddings loaded")


def sentence_alignment(s1, s2, bidirectional=True):
    tuples = get_word_correspondance_align(s1, s2)
    print(tuples)
    words1 = word_tokenize(s1)
    scores = [score for (_, _,score) in tuples]
    return sum(scores)/len(words1)

def get_word_correspondance_align(s1, s2, threshold=0.3):
    words1 = word_tokenize(s1)
    words2 = word_tokenize(s2)
    word_pairs = []
    taken = []
    for w1 in words1:
        max_score = 0
        max_word = ""
        for w2 in words2:
            if w2 not in taken:
                score = multilingual_cos_sim(w1, w2)
                if score > max_score:
                    max_score = score
                    max_word = w2

        if max_score > threshold:
            word_pairs.append((w1, max_word, max_score))
            taken.append(max_word)
    return word_pairs


if __name__ == '__main__':
    en_sentence = "What do you mean"
    es_sentence = "Que quieres decir"
    print(sentence_alignment(en_sentence, es_sentence))