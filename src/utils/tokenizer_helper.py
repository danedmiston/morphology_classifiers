import string
import re
import numpy as np

def clean_sentence(sentence):
    # Takes raw sentence and inserts spaces between words and punctuation
    #sentence = re.sub("([.,!?()])", " \1 ", sentence)
    sentence = sentence.translate(sentence.maketrans({key: " {0} ".format(key)
                                                      for key in "([.,!?()])"}))
    sentence = re.sub("\s{2,}", " ", sentence)
    return(sentence)


def collapse_columns(distribution, indices):
    # Collapses discrete distribution of M values into distribution with N values, N<M
    n_cols = len(indices)
    C = np.zeros((len(distribution), n_cols), int)
    for column in range(len(C.T)):
        C[indices[column], column] = 1
    return(distribution.dot(C))
