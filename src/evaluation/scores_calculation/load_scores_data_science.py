import pandas as pd
from scipy.stats import pearsonr as SRank
import numpy as np

if __name__ == '__main__':
    """
    Build dataframe based on sentence_scores.txt, kafka_scores, poe_scores, const_scores.
    """

    en_sentences = []
    es_sentences = []
    alignment_scores = []
    with open("medical_scores.txt", 'r') as f:
        lines = f.readlines()
        idx = 0
        for line in lines:
            if idx % 3 == 0:
                en_sentences.append(str(line))
            elif idx % 3 == 1:
                es_sentences.append(str(line))
            else:
                alignment_scores.append(float(line))
            idx += 1

    for i in range(len(en_sentences)):
        if alignment_scores[i] > 0.9:
            print(alignment_scores[i])
            print(en_sentences[i])
            print(es_sentences[i])
        if alignment_scores[i] < 0.4:
            print(alignment_scores[i])
            print(en_sentences[i])
            print(es_sentences[i])

    dp_scores = []
    sc_scores = []
    conc_scores = []
    lens = []
    for en_sentence in en_sentences:
        lens.append(length(en_sentence))
        dp_scores.append(DP(en_sentence))
        sc_scores.append(SC(en_sentence))
        conc_scores.append(concreteness(en_sentence))

    d = {'Alignment': alignment_scores, 'Len': lens, 'DP': dp_scores,
         'SC': sc_scores, 'Concreteness': conc_scores}
    df = pd.DataFrame(data=d)
    print(df.corr())
    df.to_csv("medical.csv", index=False)


    """
    Load & Data Science
    """
    df_medical = pd.read_csv("medical.csv")
    # df_austen = pd.read_csv("jane_austen.csv")
    # df_poe = pd.read_csv("poe.csv")
    # df_law = pd.read_csv("const.csv")
    # df = pd.concat([df_medical, df_law, df_austen, df_poe])

    df = df_medical

    df = df[df['Concreteness'] != -1.0]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()
    # print(5 - df['Concreteness'].mean())
    # print(5 - df['Concreteness'].median())
    print(df.corr())
    print(SRank(df['Alignment'], df['Concreteness']))

