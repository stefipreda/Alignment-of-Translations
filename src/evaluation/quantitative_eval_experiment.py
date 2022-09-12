from src.alignment.global_alignment import get_global_alignment
from src.alignment.sentence_alignment import sentence_alignment
from datasets import load_dataset
import random
import  numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    dataset_en = load_dataset("stsb_multi_mt", name="en", split="train")
    dataset_es = load_dataset("stsb_multi_mt", name="es", split="train")

    a_tr = []
    a_rel = []
    a_rand = []
    for i in tqdm(range(10)):
        en_indices = random.sample(range(0, len(dataset_en)), 100)
        #sentences_en = dataset_en['sentence1'][:100]
        sentences_en = [dataset_en[i]['sentence1'] for i in en_indices]


        #translated_sentences_es = dataset_es['sentence1'][:100]
        translated_sentences_es = [dataset_es[i]['sentence1'] for i in en_indices]

        #rel_sentences_es = dataset_es['sentence2'][:100]
        rel_sentences_es = [dataset_es[i]['sentence2'] for i in en_indices]

        random_indices = random.sample(range(0, len(dataset_en)), 100)
        random_sentences_es = [dataset_es[i]['sentence2'] for i in random_indices]

        """
        Global alignment
        """
        alignment_translated = get_global_alignment(sentences_en, translated_sentences_es)
        a_tr.append(alignment_translated)
        alignment_related = get_global_alignment(sentences_en, rel_sentences_es)
        a_rel.append(alignment_related)
        alignment_random = get_global_alignment(sentences_en, random_sentences_es)
        a_rand.append(alignment_random)

    print(np.mean(a_tr))
    print(np.mean(a_rel))
    print(np.mean(a_rand))

    """
    Sentence level alignment
    """


    scores_translated = []
    scores_rel = []
    scores_random = []

    count_rel_better = 0
    count_random_better = 0

    for i in tqdm(range(1000)):
        score_translated = sentence_alignment(sentences_en[i], translated_sentences_es[i])
        score_rel = sentence_alignment(sentences_en[i], rel_sentences_es[i])
        score_random = sentence_alignment(sentences_en[i], random_sentences_es[i])

        if score_random > score_rel + 0.005:
            count_random_better += 1

        if score_random > score_translated + 0.005:
            count_rel_better += 1

        scores_translated.append(score_translated)
        scores_rel.append(score_rel)
        scores_random.append(score_random)

    scores_random = np.array(scores_random)
    scores_random = scores_random[~np.isnan(scores_random)]

    print(count_random_better)
    print(count_rel_better)
    print("Means")
    print(np.mean(scores_translated))
    print(np.mean(scores_rel))
    print(np.mean(scores_random))
    print("Medians")
    print(np.median(scores_translated))
    print(np.median(scores_rel))
    print(np.median(scores_random))
