from datasets import load_dataset
from src.alignment.alignment import alignment_score
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import numpy as np
from scipy.stats import spearmanr
from numpy import dot
from numpy.linalg import norm
from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm
import random
import nltk

nltk.download('punkt')

encoder = SentenceTransformer('distiluse-base-multilingual-cased-v1')

encoder_en = SentenceTransformer('stsb-distilbert-base')
encoder_es = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es')



def get_global_alignment(sentences1, sentences2):
    en_embeddings = encoder_en.encode(sentences1)
    es_embeddings = encoder_es.encode(sentences2)
    return alignment_score(en_embeddings, es_embeddings)