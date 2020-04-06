from argparse import ArgumentParser
import pandas as pd

from conll.conll import *
from utils.book_keeping import *
from embedder.embedder import Transformer
from experiments.classify import *
from experiments.agree import *

parser = ArgumentParser()
parser.add_argument("--random", type=bool, default=False)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--method", type=str, default="Linear")
parser.add_argument("--language", type=str, default="German")
parser.add_argument("--feature", type=str, default="Case")
parser.add_argument("--build_datasets_classify", type=bool, default=False)
parser.add_argument("--build_datasets_agree", type=bool, default=False)
parser.add_argument("--calculate_ambiguities", type=bool, default=False)
parser.add_argument("--calculate_statistics", type=bool, default=False)
parser.add_argument("--embed_datasets_classify", type=bool, default=False)
parser.add_argument("--embed_datasets_agree", type=bool, default=False)
parser.add_argument("--test_features_classify", type=bool, default=False)
parser.add_argument("--test_layers_classify", type=bool, default=False)
parser.add_argument("--test_ambiguity_correlation", type=bool, default=False)
parser.add_argument("--test_ambiguity_per_layer", type=bool, default=False)
parser.add_argument("--test_agree", type=bool, default=False)

def build_datasets_classify():
    print("Creating and saving datasets for classify tasks. This may take some time...")
    for language in Languages:
        print("Processing {0}".format(language))
        conll = CONLL(language)
        for feature in Features[language]:
            try:
                conll.create_dataset_classify(feature, n_examples=750, dump=True)
            except:
                # Data not sufficient for 750 examples in every case
                try:
                    conll.create_dataset_classify(feature, n_examples=381, dump=True)
                except:
                    conll.create_dataset_classify(feature, n_examples=249, dump=True)
    print("Classificaiton datasets successfully created and are now ready to be embedded.")

def build_datasets_agree():
    print("Creating and saving datasets for agree tasks...")
    for language in Languages:
        if language == "English":
            print("Processing English")
            conll = English_CONLL()
        elif language == "French":
            print("Processing French")
            conll = French_CONLL()
        elif language == "German":
            print("Processing German")
            conll = German_CONLL()
        try:
            conll.create_dataset_agree(n_examples=2000, dump=True)
        except:
            # Data not sufficient for 2000 examples in every case
            conll.create_dataset_agree(n_examples=1521, dump=True)
    print("Agree datasets successfully created and are now ready to be embedded.")

def calculate_ambiguities():
    print("Calculating ambiguous/syncretic forms.")
    print("This requires repeated calls to a lexicon, and so may take some time.")
    for language in Languages:
        print("Processing {0}".format(language))
        conll = CONLL(language)
        for feature in Features[language]:
            print("Processing {0}".format(feature))
            conll.calculate_ambiguities(feature, dump=True)

def calculate_statistics():
    print("Calculating ambiguity and feature-length statistics")
    language_stats = {}
    language_stats["percent"] = {}
    language_stats["length"] = {}
    for language in Languages:
        print("Processing {0}".format(language))
        classifier = Linear(language)
        pct, leng = classifier.calculate_statistics()
        language_stats["percent"][language] = [pct]
        language_stats["length"][language] = [leng]
    df_perc = pd.DataFrame.from_dict(language_stats["percent"])
    df_len = pd.DataFrame.from_dict(language_stats["length"])
    final = pd.concat([df_perc, df_len], axis=0)
    final.index = ["Pct. Ambiguous", "Avg. Feat-len"]
    pickle.dump(final, open(Results + "statistics.p", "wb"))
    return(final)


def embed_datasets_classify(random=False, device="cuda"):
    """
    random: bool --- Whether or not to initialize random weights; done to 
    create random baselines in paper
    """
    print("Embedding classification datasets. This may take some time if on cpu...")
    for language in Languages:
        transformer = Transformer(language, random, device)
        print("Embedding datasets for {0}".format(language))
        for feature in Features[language]:
            print("Embedding dataset for {0}".format(feature))
            transformer.embed_dataset_classify(feature, batch_size=32, dump=True)
    print("All classification datasets successfully embedded.")

def embed_datasets_agree(random=False, device="cuda"):
    """
    random: bool --- Whether or not to initialize random weights;
    could be done to create random baselines, but this wasn't done in the paper
    """
    print("Embedding agree datasets. This may take some time...")
    for language in ["English", "French", "German"]:
        transformer = Transformer(language, random, device)
        print("Embedding dataset for {0}".format(language))
        transformer.embed_dataset_agree(feature="Number", batch_size=32, dump=True)
    print("All agree datasets successfully embedded.")

def test_features_classify(random=False, device="cuda"):
    """
    random: bool --- Whether to use pre-trained model or randomly initialized one

    Will reproduce results in Table 3---results may differ slightly due to random
    sampling of examples. Set random=True to reproduce results in Table 8.
    """
    results = []
    for method in Methods:
        print("Performing method {0}".format(method))
        scores = {}
        for language in Languages:
            if method == "Cluster":
                classifier = Clusterer(language, random=random)
            elif method == "Linear":
                classifier = Linear(language, random=random)
            elif method == "NonLinear":
                classifier = NonLinear(language, random=random, device=device)
            print("Processing {0}".format(language))
            feature_scores = []
            for feature in Features["German"]: # German has all relevant features
                if feature not in Features[language]:
                    feature_scores.append(None)
                else:
                    print("Processing feature {0}".format(feature))
                    layer_scores = []
                    for layer in range(13):
                        print("Processing layer {0}".format(layer))
                        score, _ = classifier.classify(layer, feature)
                        layer_scores.append(score)
                    feature_scores.append(np.mean(layer_scores))
            feature_scores.append(np.mean([score for score in feature_scores if score != None]))
            scores[language] = feature_scores
            print("\n\n")
        df = pd.DataFrame.from_dict(scores)
        df.index = [feature for feature in Features["German"]] + ["Average"]
        results.append(df)
    results = [pd.concat([df, df.mean(axis=1)], axis=1) for df in results]
    for df in results:
        df.columns = list(results[0].columns)[:-1] + ["Average"]
    final = pd.concat([df[col] for col in results[0].columns for df in results], axis=1).round(2)
    if random:
        pickle.dump(final, open(Results + "feature_scores_random.p", "wb"))
    else:
        pickle.dump(final, open(Results + "feature_scores.p", "wb"))
    return(results)

def test_layers_classify(random=False, device="cuda"):
    """
    random: bool --- Whether or not to use pre-trained model or randomly 
    initialized one
    
    Will reproduce results in Table 9---results may differ slightly due
    to random sampling of examples. Set random=True to reproduce results
    in Table 10.
    """
    results = []
    for method in Methods:
        print("Performing method {0}".format(method))
        scores = {}
        for language in Languages:
            if method == "Cluster":
                classifier = Clusterer(language, random=random)
            elif method == "Linear":
                classifier = Linear(language, random=random)
            elif method == "NonLinear":
                classifier = NonLinear(language, random=random, device=device)
            print("Processing {0}".format(language))
            layer_scores = []
            for layer in range(13):
                print("Processing layer {0}".format(layer))
                feature_scores = []
                for feature in Features[language]:
                    print("Processing feature {0}".format(feature))
                    score, _ = classifier.classify(layer, feature)
                    feature_scores.append(score)
                layer_scores.append(np.mean(feature_scores))
            layer_scores.append(np.mean(layer_scores))
            scores[language] = layer_scores
            print("\n\n")
        df = pd.DataFrame.from_dict(scores)
        df.index = [i for i in range(13)] + ["Average"]
        results.append(df)
    results = [pd.concat([df, df.mean(axis=1)], axis=1) for df in results]
    for df in results:
        df.columns = list(results[0].columns)[:-1] + ["Average"]
    final = pd.concat([df[col] for col in results[0].columns for df in results], axis=1).round(2)
    if random:
        pickle.dump(final, open(Results + "layer_scores_random.p", "wb"))
    else:
        pickle.dump(final, open(Results + "layer_scores.p", "wb"))
    return(final)

def test_ambiguity_correlation(method="Linear", random=False):
    language_scores = {}
    language_scores["percent"] = {}
    language_scores["length"] = {}
    for language in Languages:
        if method == "Cluster":
            classifier = Clusterer(language, random=random)
        elif method == "Linear":
            classifier = Linear(language, random=random)
        elif method == "NonLinear":
            classifier = NonLinear(language, random=random)
        print("Processing {0}".format(language))
        results = classifier.ambiguity_correlation()
        language_scores["percent"][language] = results["percent"]
        language_scores["length"][language] = results["length"]
    df_perc = pd.DataFrame.from_dict(language_scores["percent"])
    df_len = pd.DataFrame.from_dict(language_scores["length"])
    dfs = [df_perc, df_len]
    final = pd.concat([df[col] for col in df_perc.columns for df in dfs], axis=1)
    final.index = ["Spearman Correlation", "Pearson Correlation"]
    if random:
        pickle.dump(final, open(Results + "ambiguity_correlation_random.p", "wb"))
    else:
        pickle.dump(final, open(Results + "ambiguity_correlation.p", "wb"))
    return(final)

def test_ambiguity_per_layer(language, feature, method="Linear"):
    """
    language: str --- Which language to test
    feature: str --- Which feature to test
    method: str --- Which method of classification to use

    Used to produce Figure 2
    """
    if method == "Cluster":
        classifier = Clusterer(language)
    elif method == "Linear":
        classifier = Linear(language)
    elif method == "NonLinear":
        classifier = NonLinear(language)
    f1s = []
    amb_scores_list = []
    for layer in range(13):
        print("Processing layer {0}".format(layer))
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
    pickle.dump(df, open(Results + "amb_per_layer_" + language + "_" + feature + ".p", "wb"))
    return(df)

def test_agree():
    """
    Used to produce Tables 5-7 and 11-13
    """
    for language in ["English", "French", "German"]:
        agree = Agree(language)
        print("Processing language {0}".format(language))
        agree.run_agree_experiment()
    
    

if __name__ == "__main__":

    args = parser.parse_args()

    if args.build_datasets_classify:
        build_datasets_classify()
    if args.build_datasets_agree:
        build_datasets_agree()
    if args.calculate_ambiguities:
        calculate_ambiguities()
    if args.calculate_statistics:
        calculate_statistics()
    if args.embed_datasets_classify:
        embed_datasets_classify(random=args.random, device=args.device)
    if args.embed_datasets_agree:
        embed_datasets_agree(random=args.random, device=args.device)
    if args.test_features_classify:
        test_features_classify(random=args.random, device=args.device)
    if args.test_layers_classify:
        test_layers_classify(random=args.random, device=args.device)
    if args.test_ambiguity_correlation:
        test_ambiguity_correlation(method=args.method, random=args.random)
    if args.test_ambiguity_per_layer:
        test_ambiguity_per_layer(language=args.language, feature=args.feature, method=args.method)
    if args.test_agree:
        test_agree()
