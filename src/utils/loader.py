import pickle
from .book_keeping import *
import pandas as pd


def load_examples_classify(language="German", feature="Case"):
    addr = "../Datasets/Examples_Classify/" + language + "/" + feature + ".p"
    return(pickle.load(open(addr, "rb")))

def load_examples_attention(language="German", feature="Number"):
    addr = "../Datasets/Examples_Agree/" + language + "/" + feature + ".p"
    return(pickle.load(open(addr, "rb")))

def load_point_clouds(language="German", feature="Case"):
    addr = "../Datasets/Point_Clouds/" + language + "/" + feature + ".p"
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