from torch.utils.data import Dataset
from ast import literal_eval as parse

from utils.book_keeping import *

class ClassificationDataset(Dataset):
    def __init__(self, language, feature):
        self.language = language
        self.feature = feature

        self.examples = []

        with open(Examples_Classify[self.language] + self.feature + ".tsv", "r") as fin:
            for line in fin:
                word, loc, sent, val = line.split("\t")
                self.examples.append((word, int(loc), sent, val.strip()))

    def __len__(self):
        return(len(self.examples))

    def __getitem__(self, idx):
        return(self.examples[idx])


class AgreeDataset(Dataset):
    def __init__(self, language, feature="Number"):
        self.language = language
        self.feature = feature

        self.examples = []

        with open(Examples_Agree[self.language] + self.feature + ".tsv", "r") as fin:
            for line in fin:
                agree, ids, sent, val = line.split("\t")
                self.examples.append((parse(agree), parse(ids), sent, val.strip()))

    def __len__(self):
        return(len(self.examples))

    def __getitem__(self, idx):
        return(self.examples[idx])
        
