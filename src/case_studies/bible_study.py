from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from src.alignment.alignment import alignment_score
from scipy.stats import spearmanr as SRank
import lang2vec.lang2vec as l2v
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import operator


def cos_sim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def plot_alignment_score(scores, languages):
    y_pos = np.arange(len(languages))

    zipped_lists = zip(scores, languages)
    sorted_pairs = sorted(zipped_lists, key=operator.itemgetter(0), reverse=True)
    sorted_scores = [score for (score, _) in sorted_pairs]
    sorted_languages = [lang for (_, lang) in sorted_pairs]

    plt.bar(y_pos, sorted_scores, align='center', alpha=0.5)
    plt.xticks(y_pos, sorted_languages)
    plt.ylabel("Alignment score with English")
    plt.show()


if __name__ == '__main__':
    languages = ['en', 'ar', 'fr', 'it', 'es', 'pt', 'nl', 'de', 'fi', 'hu', 'ru', 'pl', 'tr', 'ko', 'jpn']
    languages_en = ['en']

    """
    Lang2Vec features
    """
    syntax_features = l2v.get_features(languages, 'syntax_knn', header=True)
    phonological_features = l2v.get_features(languages, 'phonology_knn', header=True)
    fam_features = l2v.get_features(languages, 'fam', header=True)

    """
    Lang2Vec distances
    """

    syntax_dists = [cos_sim(syntax_features[l1], syntax_features[l2]) for l1 in languages_en for l2 in languages]
    phonological_dists = [cos_sim(phonological_features[l1], phonological_features[l2])
                          for l1 in languages_en for l2 in languages]
    fam_dists = [np.dot(fam_features[l1], fam_features[l2]) for l1 in languages_en for l2 in languages]

    """Sentence Embeddings"""
    encoder = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    print("Encoder loaded")

    align_scores = []

    for i, ll1 in enumerate(languages_en):
        for j, ll2 in enumerate(languages):

            if ll2 < ll1:
                l1 = ll2
                l2 = ll1
            else:
                l1 = ll1
                l2 = ll2

            if l1 == l2:
                align_scores.append(1.0)
            else:
                if l2 == 'jpn':
                    l2 = 'jap'
                if l1 == 'jpn':
                    l1 = 'jap'
                print("{} - {}".format(l1, l2))
                dataset = load_dataset("bible_para", lang1=l1, lang2=l2)["train"]['translation']
                en_sentences = [dataset[i][l1] for i in range(len(dataset))]
                translated_sentences = [dataset[i][l2] for i in range(len(dataset))]
                """
                Global alignment - first 200 paragraphs
                """
                en_embeddings = encoder.encode(en_sentences[:500])
                translated_embeddings = encoder.encode(translated_sentences[:500])
                score = alignment_score(en_embeddings, translated_embeddings)
                print(score)
                align_scores.append(score)

    # Nicely plot the results
    plot_alignment_score(align_scores, languages)

    print("Syntax:")
    print(SRank(align_scores, syntax_dists))
    print("Phonological:")
    print(SRank(align_scores, phonological_dists))
    print("Fam:")
    print(SRank(align_scores, fam_dists))
