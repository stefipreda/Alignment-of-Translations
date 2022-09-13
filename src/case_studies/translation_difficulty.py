from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
import spacy

# Import spacy pipeline
nlp = spacy.load('en_core_web_sm')


def length(sentence):
    """
    Args:
        sentence:
    Returns:
        number of words
    """
    words = word_tokenize(sentence)
    return len(words)


def word_polysemy(word):
    """
    Args:
        word:

    Returns:
        number of senses
    """
    return len(wn.synsets(word))


def DP(sentence, verbose=True):
    """
    Args:
        sentence:
        verbose: True for printing polysemy of each word

    Returns:
        degree of polysemy
    """
    words = word_tokenize(sentence)
    if verbose:
        for w in words:
            print(word_polysemy(w))

    return sum([word_polysemy(w) for w in words]) / len(words)


def SC(sentence):
    """
    Args:
        sentence:

    Returns:
        structural complexity of sentence
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

    print(DP("The man deposited the money in the bank"))
    print(SC("The plants you gave me last time you saw me are growing beautifully"))