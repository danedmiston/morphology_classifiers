import numpy as np
from sklearn.preprocessing import normalize
from scipy.special import softmax



def renormalize_attention(matrix):
    # Given an attention matrix, 0's the diagonal and re-normalizes 
    intermed = matrix - np.diag(matrix.diagonal())
    row_sums = np.linalg.norm(intermed, ord=1, axis=1)
    normed = intermed / row_sums[:, np.newaxis]
    return(normed)

