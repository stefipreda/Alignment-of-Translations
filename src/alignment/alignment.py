from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
import numpy as np
from scipy.stats import spearmanr
from numpy import dot
from numpy.linalg import norm
from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm
import random

import nltk

def alignment_score(x, y, f=None):
    """Compute the alignment score between two set of points.
  ​
      Arguments:
          x: A set of points.
              shape=(n,d)
          y: A set of points.
              shape=(n,d)
          f (optional): A kernel function that computes the similarity
              or dissimilarity between two vectors. The function must
              accept two matrices with shape=(m,d).
  ​
      Returns:
          corr: The alignment score between the two sets of points.
  ​
    """
    n = x.shape[0]
    if y.shape[0] != n:
        raise ValueError(
            "The argument `x` and `y` must have the same number of rows."
        )

    # Determine upper triangular pairwise distance.
    d_x = pdist_triu(x)
    #print(d_x)
    d_y = pdist_triu(y)
    #print(d_y)
    corr, pval = spearmanr(d_x, d_y)
    return corr


def pdist_triu(x, f=None):
    """
      Pairwise distance.

      Arguments:
          x: A set of points.
              shape=(n,d)
          f (optional): A kernel function that computes the similarity
              or dissimilarity between two vectors. The function must
              accept two matrices with shape=(m,d).
  ​
      Returns:
          Upper triangular pairwise distances in "unrolled" form.
  ​
    """
    n = x.shape[0]
    if f is None:
        # Use Euclidean distance.
        """
        def f(x, y):
            return np.sqrt(np.sum((x - y) ** 2, axis=1))

        """
        # Use cosine similarity instead
        def f(x, y):
            n = x.shape[0]
            similarities = [np.squeeze(cos_sim([x[i]], [y[i]])) for i in range(n)]
            return np.array(similarities)


    # Determine indices of upper triangular matrix (not including
    # diagonal elements).
    idx_upper = np.triu_indices(n, 1)

    return f(x[idx_upper[0]], x[idx_upper[1]])
