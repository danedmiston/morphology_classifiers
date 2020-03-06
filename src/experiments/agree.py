import pandas as pd
import numpy as np
from utils.loader import *
from utils.book_keeping import *
from conll.lexicon import Lexicon
from utils.viz import *
from utils.attention import *
import pickle

np.seterr('raise')

class Agree():
    def __init__(self, language, feature):
        self.language = language
        self.feature = feature


        self.lexicon = Lexicon(self.language)
        self.attentions = load_attentions(self.language, self.feature)


    def score_distribution(self, ids, distribution, in_agree):
        """
        Scores a single distribution; helper function to evaluat_distributions
        Compares how much attention on agree set relative to how much is 
        expected at random

        args:
        ids: (int) --- tuple of ints; locations of words in agree relation
        distribution: np.array --- array of size (len(sentence.split()), 1), 
        one row in a heatmap
        in_agree: bool --- if True, then random_expectation should be len(ids)-1, 
        otherwise, should be len(ids)
        """
        if in_agree:
            random_expectation = (len(ids)-1)/(len(distribution)-1)
            agree_score = distribution[[prob for prob in ids]].sum() / random_expectation
        else:
            random_expectation = len(ids)/(len(distribution)-1)
            agree_score = distribution[[prob for prob in ids]].sum() / random_expectation
        return(agree_score)      
        
        
    def evaluate_heatmap(self, ids, heatmap):
        # Scores an entire heatmap
        attentions = renormalize_attention(heatmap) # Sets heatmap[i,i]=0 and renormalizes
        agree_scores = []
        out_scores = []
        agree_set = attentions[[dist for dist in ids]]
        out_set = attentions[[dist for dist in range(len(attentions)) if dist not in ids]]
        for row in agree_set:
            agree_scores.append(self.score_distribution(ids, row, in_agree=True))
        for row in out_set:
            out_scores.append(self.score_distribution(ids, row, in_agree=False))
        agree_avg = np.mean(agree_scores)
        out_avg = np.mean(out_scores)
        return(agree_avg/out_avg)


    
    def evaluate_heatmap_random(self, ids, heatmap):
        # Creates normalized random heatmap of same size as heatmap 
        attentions = renormalize_attention(np.random.random_sample([len(heatmap), len(heatmap)]))
        agree_scores = []
        out_scores = []
        agree_set = attentions[[dist for dist in ids]]
        out_set = attentions[[dist for dist in range(len(attentions)) if dist not in ids]]
        for row in agree_set:
            agree_scores.append(self.score_distribution(ids, row, in_agree=True))
        for row in out_set:
            out_scores.append(self.score_distribution(ids, row, in_agree=False))
        agree_avg = np.mean(agree_scores)
        out_avg = np.mean(out_scores)
        return(agree_avg/out_avg)
    
    def evaluate_head(self, head):
        scores = []
        for layer in range(12):
            for _, row in self.attentions[layer].iterrows():
                agree = row["Agree"]
                ids = row["IDs"]
                heatmap = row["Attentions"][head]
                score = self.evaluate_heatmap(ids, heatmap)
                scores.append(score)
        avg_score = np.mean(scores)
        return(avg_score)

    def evaluate_random(self, head):
        # Sets a random baseline. Suitable for head or layer
        scores_random = []
        for layer in range(12):
            for _, row in self.attentions[layer].iterrows():
                agree = row["Agree"]
                ids = row["IDs"]
                heatmap = row["Attentions"][head]
                score = self.evaluate_heatmap_random(ids, heatmap)
                scores_random.append(score)
        avg_score = np.mean(scores_random)
        return(avg_score)

    def evaluate_layer(self, layer):
        scores = []
        for head in range(12):
            for _, row in self.attentions[layer].iterrows():
                agree = row["Agree"]
                ids = row["IDs"]
                heatmap = row["Attentions"][head]
                score = self.evaluate_heatmap(ids, heatmap)
                scores.append(score)
        avg_score = np.mean(scores)
        return(avg_score)

    def run_agree_experiment(self, output=True):
        scores = np.empty((12,12))
        for layer in range(12):
            print("Processing layer", layer+1)
            for head in range(12):
                temp_scores = []
                for _, row in self.attentions[layer].iterrows():
                    ids = row["IDs"]
                    heatmap = row["Attentions"][head]
                    score = self.evaluate_heatmap(ids,heatmap)
                    temp_scores.append(score)
                scores[layer, head] = np.mean(temp_scores)
        scores = pd.DataFrame(scores, index=["Layer="+str(i) for i in range(1,13)],
                              columns=["Head="+str(i) for i in range(1,13)])
        if output:
            pickle.dump(scores, open(Agree_Results[self.language] + self.feature + ".p", "wb"))
        viz_layers_heads(scores.values)
        return(scores)
                              
                    



    def chi_squared(self, ids, distribution, renormalize, in_agree):
        """
        ids: (int) --- Location of words participating in agree
        distribution: np.array --- Probability distribution
        renormalize: bool --- Whether or not to cancel out diagonal and renormalize
        in_agree: bool --- Only matters if renormalize=True; have to calculate agree
        expectation for in_agree and out differently if renormalize is true.
        """
        if renormalize:
            if in_agree:
                agree_expectation = (len(ids)-1) / (len(distribution)-1)
            else:
                agree_expectation = (len(ids)) / (len(distribution)-1)
        else:
            agree_expectation = len(ids) / len(distribution)
        out_expectation = 1-agree_expectation
        # Numerators
        agree_score_num = (distribution[[prob for prob in ids]].sum() - agree_expectation)**2 
        out_score_num = (distribution[[prob for prob in range(len(distribution))
                                       if prob not in ids]].sum() - out_expectation)**2
        agree_score = agree_score_num / agree_expectation
        out_score = out_score_num / out_expectation
        if out_expectation == 0:
            Exception("Divided by 0")
        return(agree_score + out_score)

    def evaluate_heatmap_chi2(self, ids, heatmap, renormalize=True):
        if renormalize:
            heatmap = renormalize_attention(heatmap)
        agree_scores = []
        out_scores = []
        agree_set = heatmap[[dist for dist in ids]]
        out_set = heatmap[[dist for dist in range(len(heatmap)) if dist not in ids]]
        for row in agree_set:
            agree_scores.append(self.chi_squared(ids, row, renormalize=renormalize, in_agree=True))
        for row in out_set:
            out_scores.append(self.chi_squared(ids, row, renormalize=renormalize, in_agree=False))
        agree_avg = np.mean(agree_scores)
        out_avg = np.mean(out_scores)
        # Return agree average, out average, and agree/out
        # Scores can be significant with lots of focus on agree set
        # or lots of focus not on agree set.
        return(agree_avg, out_avg, agree_avg/out_avg)

    def average_chi2_heatmap(self, ids, heatmap, renormalize=False):
        scores = []
        for dist in range(len(heatmap)):
            scores.append(self.chi_squared(ids,
                                           heatmap[dist],
                                           renormalize=renormalize,
                                           in_agree=False))
        return(np.mean(scores))

    def run_chi2_experiment(self, output=True, renormalize=True):
        scores_agree = np.empty((12,12))
        scores_out = np.empty((12,12))
        scores_aoo = np.empty((12,12))
        for layer in range(12):
            print("Processing layer", layer+1)
            for head in range(12):
                temp_scores_agree = []
                temp_scores_out = []
                temp_scores_aoo = []
                for _, row in self.attentions[layer].iterrows():
                    ids = row["IDs"]
                    heatmap = row["Attentions"][head]
                    try:
                        agree, out, aoo = self.evaluate_heatmap_chi2(ids,
                                                                     heatmap,
                                                                     renormalize=renormalize)
                        temp_scores_agree.append(agree)
                        temp_scores_out.append(out)
                        temp_scores_aoo.append(aoo)
                    except:
                        continue
                scores_agree[layer, head] = np.mean(temp_scores_agree)
                scores_out[layer, head] = np.mean(temp_scores_out)
                scores_aoo[layer,head] = np.mean(temp_scores_aoo) 
        scores_agree = pd.DataFrame(scores_agree, index=["Layer="+str(i) for i in range(1,13)],
                                    columns=["Head="+str(i) for i in range(1,13)])
        scores_out = pd.DataFrame(scores_out, index=["Layer="+str(i) for i in range(1,13)],
                                  columns=["Head="+str(i) for i in range(1,13)])
        scores_aoo = pd.DataFrame(scores_aoo, index=["Layer="+str(i) for i in range(1,13)],
                                  columns=["Head="+str(i) for i in range(1,13)])
        scores = {"agree":scores_agree, "out":scores_out, "agree_over_out":scores_aoo}
        if output:
            if renormalize:
                pickle.dump(scores, open(Agree_Results[self.language] + "Chi2_renormalized.p", "wb"))
            else:
                pickle.dump(scores, open(Agree_Results[self.language] + "Chi2.p", "wb"))
        viz_layers_heads(scores_agree.values)
        viz_layers_heads(scores_out.values)
        viz_layers_heads(scores_aoo.values)
        return(scores)

    def run_chi2_average_experiment(self, output=True, renormalize=True):
        scores = np.empty((12,12))
        for layer in range(12):
            print("Processing layer", layer+1)
            for head in range(12):
                temp_scores = []
                for _, row in self.attentions[layer].iterrows():
                    ids = row["IDs"]
                    heatmap = row["Attentions"][head]
                    avg = self.average_chi2_heatmap(ids, heatmap, renormalize=renormalize)
                    temp_scores.append(avg)
                scores[layer, head] = np.mean(temp_scores)
        scores = pd.DataFrame(scores, index=["Layer="+str(i) for i in range(1,13)],
                              columns=["Head="+str(i) for i in range(1,13)])
        if output:
            pickle.dump(scores, open(Agree_Results[self.language] + "Avg_Chi2.p", "wb"))
        viz_layers_heads(scores.values)
        return(scores)
    
