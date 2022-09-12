from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import numpy as np
from scipy.stats import spearmanr
from nltk.tokenize import word_tokenize
import random

from src.alignment.alignment import alignment_score

encoder = SentenceTransformer('distiluse-base-multilingual-cased-v1')
print("Embeddings loaded")


def read_anchor_words(file='../../data/en_es_dictionary.txt'):
    en_words = []
    es_words = []
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            words = word_tokenize(line)
            es_words.append(words[0])
            en_words.append(words[1])

    return en_words, es_words


def closest_word(word):
    w_emb = encoder.encode(word)
    max_sim = 0
    word = ""
    for i, emb in enumerate(anchors_es_embs):
        cos = cos_sim(w_emb, emb)
        if cos > max_sim:
            max_sim = cos
            word = anchors_es[i]
    return word


def multilingual_cos_sim(w_en, w_es):
    en_emb = encoder.encode(w_en)
    es_emb = encoder.encode(w_es)
    return cos_sim(en_emb, es_emb)


def word_alignment(w_en, w_es):
    en_emb = encoder.encode(w_en)
    es_emb = encoder.encode(w_es)
    en_emb = np.expand_dims(en_emb, axis=0)
    es_emb = np.expand_dims(es_emb, axis=0)
    en_embs = np.append(anchors_en_embs, en_emb, axis=0)
    es_embs = np.append(anchors_es_embs, es_emb, axis=0)
    score = alignment_score(en_embs, es_embs)
    return score


anchors_en, anchors_es = read_anchor_words()

#train_indices = random.sample(range(0, len(anchors_en)), 100)
train_indices = range(300, 400)
train_words_en = [anchors_en[i] for i in train_indices]
train_words_es = [anchors_es[i] for i in train_indices]
anchors_en_embs = encoder.encode(train_words_en)
anchors_es_embs = encoder.encode(train_words_es)


"""
for round in range(5):
    print("Round {}".format(round))

    train_indices = random.sample(range(0, len(anchors_en)), 100)
    train_words_en = [anchors_en[i] for i in train_indices]
    train_words_es = [anchors_es[i] for i in train_indices]
    anchors_en_embs = encoder.encode(train_words_en)
    anchors_es_embs = encoder.encode(train_words_es)
    print(len(anchors_es_embs))


    print(alignment_score(anchors_en_embs, anchors_es_embs))

    #print(word_alignment("dog", "perro"))
    #print(word_alignment("dog", "cancion"))

    test_indices = random.sample(range(0, len(anchors_en)), 50)

    test_words_en = [anchors_en[i] for i in test_indices]
    test_words_es = [anchors_es[i] for i in test_indices]


    test_words_en_embs = encoder.encode(test_words_en)
    test_words_es_embs = encoder.encode(test_words_es)

    cos_sim_test_scores = [cos_sim(test_words_en_embs[i], test_words_es_embs[i]) for i in range(len(test_words_es_embs))]

    alignment_test_scores = [word_alignment(test_words_en[i], test_words_es[i]) for i in range(len(test_words_en))]

    print(spearmanr(alignment_test_scores, cos_sim_test_scores))


for N in [10, 50, 100, 200, 500]:
    algns = []
    print("N = {}".format(N))
    for _ in range(10):
        train_indices = random.sample(range(0, len(anchors_en)), N)
        train_words_en = [anchors_en[i] for i in train_indices]
        train_words_es = [anchors_es[i] for i in train_indices]
        anchors_en_embs = encoder.encode(train_words_en)
        anchors_es_embs = encoder.encode(train_words_es)

        algns.append(alignment_score(anchors_en_embs, anchors_es_embs))

    print(sum(algns) / len(algns))
"""
