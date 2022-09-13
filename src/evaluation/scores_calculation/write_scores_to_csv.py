from datasets import load_dataset
from src.alignment.global_alignment import  get_global_alignment
import random

import nltk

nltk.download('punkt')

if __name__ == '__main__':
    en_sentence = "But it was something that had to be risked."
    es_sentence = "Pero había que intentarlo"

    books_en_es = load_dataset("opus_books", "en-es")
    books_en_es = books_en_es['train']
    translations = books_en_es['translation']
    """
    Load first book - Jane Austen : Sense and Sensibility
    """
    first_book_len = 4495  # 500
    sampled_indices = random.sample(range(0, first_book_len), 500)
    en_sentences = [translations[i]['en'] for i in range(1, first_book_len)]
    es_sentences = [translations[i]['es'] for i in range(1, first_book_len)]

    score = get_global_alignment(en_sentences, es_sentences)
    print("Global alignment Jane Austen : {}".format(score))

    # Global alignment of Jane Austen

    en_sentences = [translations[i]['en'] for i in sampled_indices]
    es_sentences = [translations[i]['es'] for i in sampled_indices]

    score = get_global_alignment(en_sentences, es_sentences)
    print("Global alignment Jane Austen - random: {}".format(score))

    kafka_begin = 48625
    kafka_end = 49260
    poe_begin = 49265
    poe_end = 49545

    """
    Load Kafka : Metamorphosis
    """

    en_kafka_sentences = [translations[i]['en'] for i in range(kafka_begin, kafka_end)]
    es_kafka_sentences = [translations[i]['es'] for i in range(kafka_begin, kafka_end)]

    score = get_global_alignment(en_kafka_sentences, es_kafka_sentences)
    print("Global alignment Kafka : {}".format(score))

    """
    Load Edgar Allan Poe : The Fall of the House of Usher
    """

    en_poe_sentences = [translations[i]['en'] for i in range(poe_begin, poe_end)]
    es_poe_sentences = [translations[i]['es'] for i in range(poe_begin, poe_end)]

    score = get_global_alignment(en_poe_sentences, es_poe_sentences)
    print("Global alignment Poe : {}".format(score))

    """
    Load European Constitution
    """
    const_en_es = load_dataset("opus_euconst", "en-es")
    const_en_es = const_en_es['train']
    translations = const_en_es['translation']

    en_sentences = [translations[i]['en'] for i in range(500)]
    es_sentences = [translations[i]['es'] for i in range(500)]

    score = get_global_alignment(en_sentences, es_sentences)
    print("Global alignment European Const : {}".format(score))

    """
    Load ECDC - Medical
    """

    medical_en_es = load_dataset("qanastek/ECDC", "en-es", split='train')
    translations = medical_en_es['translation']
    en_sentences = [translations[i]['en'] for i in range(500)]
    es_sentences = [translations[i]['es'] for i in range(500)]

    score = get_global_alignment(en_sentences, es_sentences)
    print("Global alignment Medical : {}".format(score))

    """
    Load random sample of books dataset
    """
    translations = books_en_es['translation']
    sampled_indices = random.sample(range(0, len(translations)), 500)
    en_sentences = [translations[i]['en'] for i in sampled_indices]
    es_sentences = [translations[i]['es'] for i in sampled_indices]

    score = get_global_alignment(en_sentences, es_sentences)
    print("Global alignment random books : {}".format(score))


    for i in range(len(en_sentences)):
        if "Source:" in en_sentences[i]:
            print(i)
            print(en_sentences[i + 1])

    scores = []
    with open("medical_scores.txt", 'w') as f:
        for i in tqdm(range(len(en_sentences))):
            score = get_word_level_alignment(en_sentences[i], es_sentences[i])
            if not np.isnan(score):
                scores.append(score)

            f.write(en_sentences[i])
            f.write("\n")
            f.write(es_sentences[i])
            f.write("\n")
            f.write(str(score))
            f.write("\n")

    print(sum(scores)/len(scores))
