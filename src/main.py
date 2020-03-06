# This will be the main file for use later
from argparse import ArgumentParser
from utils.loader import *
from experiments.classify import * 
from experiments.agree import *
from utils.book_keeping import *
import numpy as np
import pandas as pd
import warnings
import pickle
from scipy.stats import spearmanr, pearsonr

warnings.simplefilter("ignore")


parser = ArgumentParser()
parser.add_argument("--method", type=str, default="cluster")
parser.add_argument("--task", type=str, default="layers")
args = parser.parse_args()


def test_layers_classify(method="cluster"):
    scores = {}
    for language in Languages:
        print("Processing", language)
        layer_scores = []
        for layer in range(13):
            print("Processing layer", layer)
            feature_scores = []
            for feature in Features[language]:
                print("Processing feature", feature)
                if method == "cluster":
                    clusterer = Clusterer(language)
                    score, _ = clusterer.classify(layer, feature)
                elif method == "linear":
                    linear = Linear(language)
                    score, _ = linear.classify(layer, feature)
                elif method == "nonlinear":
                    score = 0
                    nonlinear = NonLinear(language)
                    score, _ = nonlinear.classify(layer, feature)
                feature_scores.append(score)
            layer_scores.append(np.mean(feature_scores))
        layer_scores.append(np.mean(layer_scores))
        scores[language] = layer_scores
        print("\n\n")
    df = pd.DataFrame.from_dict(scores)
    df.index = [i for i in range(13)] + ["Average"]
    pickle.dump(df, open(Tables+method+"_layer_scores.p", "wb"))
    return(df)

def test_features_classify(method="cluster"):
    scores = {}
    for language in Languages:
        print("Processing", language)
        feature_scores = []
        for feature in Features["German"]: # German has all relevant features
            if feature not in Features[language]:
                feature_scores.append(None)
            else:
                print("Processing feature", feature)
                layer_scores = []
                for layer in range(13):
                    print("Processing layer", layer)
                    if method == "cluster":
                        clusterer = Clusterer(language)
                        score, _ = clusterer.classify(layer, feature)
                    elif method == "linear":
                        linear = Linear(language)
                        score, _ = linear.classify(layer, feature)
                    elif method == "nonlinear":
                        nonlinear = NonLinear(language)
                        score, _ = nonlinear.classify(layer, feature)
                    layer_scores.append(score)
                feature_scores.append(np.mean(layer_scores))
        feature_scores.append(np.mean([score for score in feature_scores if score != None]))
        scores[language] = feature_scores
        print("\n\n")
    df = pd.DataFrame.from_dict(scores)
    df.index = [feature for feature in Features["German"]] + ["Average"]
    pickle.dump(df, open(Tables+method+"_feature_scores.p", "wb"))
    return(df)

def test_ambiguity_correlations(method="cluster"):
    # Tests the correlation between performance as measured by average F1 score across
    # layers for each feature, and how ambiguous that feature is for the given langauge
    language_scores = {}
    language_scores["percent"] = {}
    language_scores["length"] = {}
    for language in Languages:
        print("Processing language", language)
        avg_f1s = []
        pct_ambiguous_list = []
        feature_lengths = []
        for feature in Features[language]:
            print("Processing feature", feature)
            if method == "cluster":
                classifier = Clusterer(language)
            elif method == "linear":
                classifier = Linear(language)
            elif method == "nonlinear":
                classifier = NonLinear(language)
            breakdown = classifier.ambiguity_breakdown(feature)
            pct_ambiguous = 1 - breakdown[1]
            f1s = []
            amb_scores_list = []
            for i in range(13):
                f1, amb_scores = classifier.classify(i, feature)
                f1s.append(f1)
                amb_scores_list.append(amb_scores)
            possible_values = len(classifier.features[feature])
            avg_amb_scores = []
            avg_f1 = np.mean(f1s) 
            for i in range(possible_values):
                try:
                    scores = np.stack([item[i][1] for item in amb_scores_list], axis=0)
                    avg_ambiguity_score = np.mean(scores)
                    avg_amb_scores.append(avg_ambiguity_score)
                except:
                    continue
            avg_f1s.append(avg_f1)
            pct_ambiguous_list.append(pct_ambiguous)
            feature_lengths.append(len(breakdown))
        language_scores["percent"][language] = (spearmanr(avg_f1s, pct_ambiguous_list),
                                                pearsonr(avg_f1s, pct_ambiguous_list))
        language_scores["length"][language] = (spearmanr(avg_f1s, feature_lengths),
                                               pearsonr(avg_f1s, feature_lengths))
    df_perc = pd.DataFrame.from_dict(language_scores["percent"])
    df_perc.index = ["Spearman Correlation", "Pearson Correlation"]
    df_len = pd.DataFrame.from_dict(language_scores["length"])
    df_len.index = ["Spearman Correlation", "Pearson Correlation"]
    pickle.dump((df_perc, df_len), open(Tables+method+"_ambiguity_correlations.p", "wb"))
    return((df_perc, df_len))

def ambiguity_per_layer(language, feature, method="linear"):
    # Used to produce Figure 2
    if method == "cluster":
        classifier = Cluster(language)
    elif method == "linear":
        classifier = Linear(language)
    elif method == "nonlinear":
        classifier = NonLinear(language)
    f1s = []
    amb_scores_list = []
    for layer in range(13):
        f1, amb_scores = classifier.classify(layer, feature)
        f1s.append(f1)
        amb_scores_list.append(amb_scores)
    possible_values = len(classifier.features[feature])
    score_per_amb = {}
    for i in range(possible_values):
        try:
            score_per_amb[i+1] = np.stack([item[i][1] for item in amb_scores_list], axis=0)
        except:
            continue
    df = pd.DataFrame.from_dict(score_per_amb)
    return(df)


if __name__ == "__main__":
    print(ambiguity_per_layer("German", "Case"))


    #print(test_ambiguity_correlations(method="linear"))
    
    #if args.task == "layers":
    #    print(test_layers_classify(args.method))
    #elif args.task == "features":
    #    print(test_features_classify(args.method))
