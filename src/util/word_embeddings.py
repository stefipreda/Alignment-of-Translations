from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformer

from gensim.models import Word2Vec, KeyedVectors
import numpy as np

"""
English vectors
"""
english_embeddings = KeyedVectors.\
    load_word2vec_format('../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin', binary=True)

"""
Spanish Vectors
"""
spanish_embeddings = {}

with open('../input/pretrained-word-vectors-for-spanish/SBW-vectors-300-min5.txt') as f:
    f.readline()
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, dtype='float32', sep=" ")

        spanish_embeddings[word.lower()] = coefs

        
"""
Multilingual encoder
"""
encoder = SentenceTransformer('distiluse-base-multilingual-cased-v1')
print("Embeddings loaded")


"""
Functions to get embeddings
"""


def get_emb_en(word):
    word = word.lower()
    if word in english_embeddings:
        emb = english_embeddings[word]
        emb = np.expand_dims(emb, axis=0)
        return emb
    else:
        raise Exception("Sorry, no EN embedding for this word!")
    # Alternatively, multilingual encoder:
    # return encoder.encode(word)


def get_emb_es(word):
    word = word.lower()
    if word in spanish_embeddings:
        emb = spanish_embeddings[word]
        emb = np.expand_dims(emb, axis=0)
        return emb
    else:
        raise Exception("Sorry, no ES embedding for this word!")
    # Alternatively, multilingual encoder:
    return encoder.encode(word)


def get_embs_en(words):
    embs_en = [get_emb_en(w) for w in words if w in english_embeddings]
    return embs_en
    # Alternatively, multilingual encoder:
    #return encoder.encode(words)


def get_embs_es(words):
    embs_es = [get_emb_es(w) for w in words if w in spanish_embeddings]
    return embs_es
    # Alternatively, multilingual encoder:
    # return encoder.encode(words)

    
"""
Multilingual functionality
"""


def multilingual_cos_sim(w_en, w_es):
    en_emb = encoder.encode(w_en)
    es_emb = encoder.encode(w_es)
    return cos_sim(en_emb, es_emb)


