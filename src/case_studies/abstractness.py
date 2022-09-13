import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
conc_scores = {}

# encoder = SentenceTransformer('distiluse-base-multilingual-cased-v1')

# Read abstract scores from csv file
df = pd.read_csv("../../data/abstract_scores.csv")
df_scores = df[['Word', 'Conc.M']]
df_scores.dropna()
words = []
for _, row in df_scores.iterrows():
    word = str(row['Word'])
    score = row['Conc.M']
    if " " not in word:
        conc_scores[word] = score
        words.append(word)


def concreteness(sentence):
    """
    Args:
        sentence:

    Returns:
        a concreteness rating between 0 to 5
    """
    words = word_tokenize(sentence)
    words_found = []
    for word in words:
        if word in conc_scores:
            words_found.append(word)
        else:
            """
            Look up its lemma
            """
            lemma = lemmatizer.lemmatize(word)
            if lemma in conc_scores:
                words_found.append(lemma)
            else:
                """
                Look up its stem
                """
                stem = stemmer.stem(word)
                if stem in conc_scores:
                    words_found.append(stem)

    if len(words_found) == 0:
        return -1
    score = sum([conc_scores[w] for w in words_found]) / len(words_found)
    return score


if __name__ == '__main__':
    print(concreteness("The family of Dashwood had long been settled in Sussex."))
