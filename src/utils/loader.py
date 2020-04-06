import pickle
from .book_keeping import *
import pandas as pd
from embedder.dataset import *

def load_examples_classify(language="German", feature="Case"):
    dataset = ClassificationDataset(language, feature)
    return(dataset)

def load_examples_attention(language="German", feature="Number"):
    dataset = AgreeDataset(language, feature)
    return(dataset)

def load_ambiguities(language="German", feature="Number"):
    addr = Ambiguity[language] + feature + ".p"
    return(pickle.load(open(addr, "rb")))

def load_vectors(language="German", feature="Case", random=False):
    if random:
        addr = "../Datasets/Vectors/" + language + "/" + feature + "_random.p"
    else:
        addr = "../Datasets/Vectors/" + language + "/" + feature + ".p"
    return(pickle.load(open(addr, "rb")))

def load_attentions(language="German", feature="Number"):
    addr = "../Datasets/Attentions/" + language + "/" + feature + ".p"
    return(pickle.load(open(addr, "rb")))

def load_lexicon(language="German"):
    addr = Lexicons[language]
    lexicon = pd.read_csv(addr, sep="\t", error_bad_lines=False, warn_bad_lines=False, header=None)
    lexicon.drop([0,1,4,5,7], axis=1, inplace=True)
    lexicon.columns = ["Word", "Lemma", "Features"]
    return(lexicon)
