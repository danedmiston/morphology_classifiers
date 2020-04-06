import numpy as np
import pandas as pd
from utils.loader import *
from utils.book_keeping import *
from utils.viz import *
import pickle


class Agree():
    def __init__(self, language, feature="Number"):
        self.language = language
        self.feature = feature


        self.attentions = load_attentions(self.language, self.feature)


    def renormalize_attention(self, matrix):
        """
        matrix: n x n tensor --- Attention matrix for a sentence 

        Helper function;
        Given an attention matrix, 0's the diagonal and re-normalizes
        """
        intermed = matrix - np.diag(matrix.diagonal())
        row_sums = np.linalg.norm(intermed, ord=1, axis=1)
        normed = intermed / row_sums[:, np.newaxis]
        return(normed)

        
    def chi_squared(self, ids, distribution, in_agree, renormalize=True):
        """
        ids: (int) --- Location of words participating in agree
        distribution: tensor --- Probability distribution, tensor.shape == 1 x sent-length
        in_agree: bool --- Only matters if renormalize=True; 
        renormalize: bool --- Whether or not to 0 the diagonal
        """
        if renormalize:
            if in_agree:
                agree_expectation = (len(ids)-1) / (len(distribution)-1)
            else:
                agree_expectation = (len(ids)) / (len(distribution)-1)
        else:
            agree_expectation = len(ids) / len(distribution)
        out_expectation = 1 - agree_expectation
        # Numerators
        agree_score_num = (distribution[[prob for prob in ids]].sum() - agree_expectation)**2
        out_score_num = (distribution[[prob for prob in range(len(distribution))
                                       if prob not in ids]].sum() - out_expectation)**2
        agree_score = agree_score_num / agree_expectation
        out_score = out_score_num / out_expectation
        return(agree_score + out_score)

    def evaluate_att_matrix_chi2(self, ids, att_matrix, renormalize=True):
        if renormalize:
            att_matrix = self.renormalize_attention(att_matrix)
        agree_scores = []
        out_scores = []
        agree_set = att_matrix[[dist for dist in ids]]
        out_set = att_matrix[[dist for dist in range(len(att_matrix)) if dist not in ids]]
        for row in agree_set:
            agree_scores.append(self.chi_squared(ids, row, in_agree=True, renormalize=renormalize))
        for row in out_set:
            out_scores.append(self.chi_squared(ids, row, in_agree=False, renormalize=renormalize))
        agree_avg = np.mean(agree_scores)
        out_avg = np.mean(out_scores)
        return(agree_avg, out_avg)


    def run_agree_experiment(self, renormalize=True):
        scores_agree = np.empty((12,12))
        scores_out = np.empty((12,12))
        scores_aoo = np.empty((12,12))
        for layer in range(12):
            print("Processing layer {0}".format(layer+1))
            for head in range(12):
                print("Processing head {0}".format(head+1))
                temp_scores_agree = []
                temp_scores_out = []
                for ex in self.attentions:
                    ids = [int(item) for item in ex[0]]
                    att_matrix = ex[2][layer][head]
                    agree, out = self.evaluate_att_matrix_chi2(ids,
                                                               att_matrix,
                                                               renormalize=renormalize)
                    temp_scores_agree.append(agree)
                    temp_scores_out.append(out)
                scores_agree[layer, head] = np.nanmean(temp_scores_agree)
                scores_out[layer, head] = np.nanmean(temp_scores_out)
        scores_agree = pd.DataFrame(scores_agree, index=["Layer="+str(i) for i in range(1,13)],
                                    columns=["Head="+str(i) for i in range(1,13)])
        scores_out = pd.DataFrame(scores_out, index=["Layer="+str(i) for i in range(1,13)],
                                  columns=["Head="+str(i) for i in range(1,13)])
        scores = {"agree" : scores_agree, "out" : scores_out}
        #viz_layers_heads(scores_agree.values)
        #viz_layers_heads(scores_out.values)
        pickle.dump(scores, open(Results + self.language + "_agree.p", "wb"))
        return(scores)





    
