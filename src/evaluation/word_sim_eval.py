import pandas as pd
from scipy.stats import spearmanr as SRank
import numpy as np
from src.alignment.word_alignment import word_alignment, multilingual_cos_sim

sim_file = "../../data/en_es_word_similarity.csv"
df_sim = pd.read_csv(sim_file)

words_en = df_sim['ENG']
words_es = df_sim['SPA']
sim_scores = df_sim['score']

cos_sim_scores = [multilingual_cos_sim(words_en[i], words_es[i]) for i in range(1000)]
align_scores = [word_alignment(words_en[i], words_es[i]) for i in range(1000)]

print(SRank(align_scores, list(sim_scores[:100])))
print(SRank(cos_sim_scores, list(sim_scores[100])))
print(SRank(align_scores, cos_sim_scores))
