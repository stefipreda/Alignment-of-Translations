from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
import spacy
import pandas as pd

nlp = spacy.load('en_core_web_sm')


def length(sentence):
    """
    :param sentence:
    :return: number of words
    """
    words = word_tokenize(sentence)
    return len(words)


def word_polysemy(word):
    return len(wn.synsets(word))


def DP(sentence, verbose=True):
    """
    :param sentence:
    :return: degree of polysemy
    """
    words = word_tokenize(sentence)
    if verbose:
        for w in words:
            print(word_polysemy(w))

    return sum([word_polysemy(w) for w in words]) / len(words)


def SC(sentence):
    """
    :param sentence:
    :return: structural complexity
    """
    doc = nlp(sentence)
    dep_lengths = []
    for token in doc:
        if token.dep_ != 'punct' and token.dep_ != 'ROOT':
            """
            Dependency length = difference of indices between the tokens connected by a dependency
            """
            dl = abs(token.i - token.head.i)
            dep_lengths.append(dl)
    print(dep_lengths)
    return sum(dep_lengths)/len(dep_lengths)


if __name__ == '__main__':
    #df = pd.read_csv("td_en_es.csv")
    print(DP("The man deposited the money in the bank"))