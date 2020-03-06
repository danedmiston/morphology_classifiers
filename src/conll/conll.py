import pyconll as pc
from utils.book_keeping import * 
from utils.tokenizer_helper import *
import pandas as pd
import pickle

class CONLL():
    def __init__(self, language):
        self.language = language

        self.conlls = []

        print(CoNLL[self.language])
        for conll in CoNLL[self.language]:
            self.conlls.append(pc.load_from_file(conll))

        self.features = Features[self.language]


    def create_dataset_classify(self, feature, n_examples=750, dump=True):
        """
        This function creates a dataframe with four columns
        The columns are [Word, ID, Sentence, Value]
        It records a Word in position ID, marked for the feature with Value in Sentence 
        This will be used to make a point-cloud feature set
        
        Args:
        feature: str --- Some feature in Features[self.language]
        n_examples: int --- How many examples per feature value
        dump: bool --- If True, dump dataframe into ../Datasets/Examples/self.language
        """
        examples = []
        for dataset in self.conlls:
            for sentence in dataset:
                cleaned = clean_sentence(sentence.text)
                for token in sentence:
                    if feature in token.feats:
                        # Subtract 1 from token.id for 0-based indexing
                        try:
                            examples.append((token.form, int(token.id)-1,
                                             cleaned, token.feats[feature]))
                        except:
                            continue
        df = pd.DataFrame(examples, columns=["Word", "ID", "Sentence", "Value"])
        values = []
        for value in self.features[feature]:
            values.append(df[df["Value"] == value].sample(n=n_examples))
            #values.append(df[df["Value"] == value])
        final_results = pd.concat(values)
        if dump:
            pickle.dump(final_results, open(Examples_Classify[self.language] + feature + ".p", "wb"))
        return(final_results)


    def create_dataset_agree(self, feature="Number", n_examples=2000, dump=True):
        """
        Similar to above, but capturing sets of words which agree

        Args:
        feature: str --- this is superfluous for now; I will only be doing
        one type of agree per language. e.g. For English I will only be
        doing agreement in number between subjects and verbs/aux
        """
        examples = []
        for dataset in self.conlls:
            for sentence in dataset:
                agree, ids, sentence, value = self.retrieve_agree(sentence, feature)
                if len(agree) > 0:
                    examples.append((agree, ids, sentence.text, value))
        df = pd.DataFrame(examples, columns=["Agree", "IDs", "Sentence", "Value"])
        df = df.sample(n=n_examples)
        if dump:
            pickle.dump(df, open(Examples_Agree[self.language] + feature + ".p", "wb"))
        return(df)
    
    
class English_CONLL(CONLL):
    def __init__(self):
        super().__init__("English")

        self.POS = {"Det" : ["DET"],
                    "Noun" : ["PRON", "PROPN", "NOUN"],
                    "Verb" : ["AUX", "VERB"]}
    
    def retrieve_agree(self, sentence, feature):
        # Takes a pyconll sentence object and returns Agree sets
        # Only captures subj-verb agreement
        pairs = []
        ids = []
        values = []
        nouns = [token for token in sentence
                 if token.upos in self.POS["Noun"]
                 and feature in token.feats]
        verbs = [token for token in sentence
                 if token.upos in self.POS["Verb"]
                 and feature in token.feats]
        for noun in nouns:
            for verb in verbs:
                if noun.head == verb.id or noun.head == verb.head: 
                    if noun.deprel == "nsubj":
                        try:
                            if verb.feats["VerbForm"]=={'Fin'}:
                                value = noun.feats[feature]
                                # Have to subtract 1 from id for 0-based indexing
                                pairs.append((noun._form,
                                              verb._form))
                                ids.append((int(noun.id)-1, int(verb.id)-1))
                                values.append(value)
                        except:
                            continue
        # Only want sentences with one matching agree relationship 
        if len(pairs) == 1:
            return(pairs[0], ids[0], sentence, values[0])
        else:
            return([], [], sentence, [])

    """
    def retrieve_agree_alternative(self, sentence, feature):
        # Takes a pyconll sentence object and returns Agree sets
        # Only captures subj-verb agreement
        matches = []
        ids = []
        values = []
        dets = [token for token in sentence
                if token.upos in self.POS["Det"]
                and feature in token.feats]
        nouns = [token for token in sentence
                 if token.upos in self.POS["Noun"]
                 and feature in token.feats]
        verbs = [token for token in sentence
                 if token.upos in self.POS["Verb"]
                 and feature in token.feats]
        for det in dets:
            for noun in nouns:
                for verb in verbs:
                    if det.head == noun.id and (noun.head == verb.head or noun.head == verb.id): 
                        if noun.deprel == "nsubj":
                            try:
                                if verb.feats["VerbForm"]=={'Fin'}:
                                    value = noun.feats[feature]
                                    # Have to subtract 1 from id for 0-based indexing
                                    matches.append((det._form,
                                                    noun._form,
                                                    verb._form))
                                    ids.append((int(det.id)-1, int(noun.id)-1, int(verb.id)-1))
                                    values.append(value)
                            except:
                                continue
        # Only want sentences with one matching agree relationship 
        if len(matches) == 1:
            return(matches[0], ids[0], sentence, values[0])
        else:
            return([], [], sentence, [])
    """

        

        
        
class German_CONLL(CONLL):
    def __init__(self):
        super().__init__("German")

        self.POS = {"Det" : ["DET"],
                    "Adj" : ["ADJ"],
                    "Noun" : ["NOUN"],
                    "Verb" : ["VERB", "AUX"]}

    def retrieve_agree(self, sentence, feature):
        # For now, only going to implement subj-verb agreement for number,
        # where subj is of form DET-ADJ-NOUN
        # Code could no doubt be optimized, but only needs to be run once and is fast enough
        matches = []
        ids = []
        values = []
        dets = [token for token in sentence
                if token.upos in self.POS["Det"]
                and feature in token.feats]
        adjs = [token for token in sentence
                if token.upos in self.POS["Adj"]
                and feature in token.feats]
        nouns = [token for token in sentence
                 if token.upos in self.POS["Noun"]
                 and feature in token.feats]
        verbs = [token for token in sentence
                 if token.upos in self.POS["Verb"]
                 and feature in token.feats]
        for det in dets:
            for adj in adjs:
                for noun in nouns:
                    for verb in verbs:
                        if det.head == adj.head == noun.id and noun.head == verb.id:
                            if det.feats[feature] == adj.feats[feature] == noun.feats[feature] == verb.feats[feature]:
                                if noun.deprel == "nsubj":
                                    value = det.feats[feature]
                                    matches.append((det._form, adj._form, noun._form, verb._form))
                                    ids.append((int(det.id)-1, int(adj.id)-1,int(noun.id)-1,int(verb.id)-1))
                                    values.append(value)
        if len(matches) == 1:
            return(matches[0], ids[0], sentence, values[0])
        else:
            return([], [], sentence, [])

class French_CONLL(CONLL):
    def __init__(self):
        super().__init__("French")

        self.POS = {"Det" : ["DET"],
                    "Adj" : ["ADJ"],
                    "Noun" : ["NOUN"],
                    "Verb" : ["AUX", "VERB"]}
        
    def retrieve_agree(self, sentence, feature):
        # Will do same as German; subj-verb agreement, where subj of form: DET-ADJ-NOUN
        matches = []
        ids = []
        values = []
        dets = [token for token in sentence
                if token.upos in self.POS["Det"]
                and feature in token.feats]
        adjs = [token for token in sentence
                if token.upos in self.POS["Adj"]
                and feature in token.feats]
        nouns = [token for token in sentence
                 if token.upos in self.POS["Noun"]
                 and feature in token.feats]
        verbs = [token for token in sentence
                 if token.upos in self.POS["Verb"]
                 and feature in token.feats]
        for det in dets:
            for adj in adjs:
                for noun in nouns:
                    for verb in verbs:
                        if det.head == adj.head == noun.id and noun.head == verb.id:
                            if det.feats[feature] == adj.feats[feature] == noun.feats[feature] == verb.feats[feature]:
                                if noun.deprel == "nsubj":
                                    value = det.feats[feature]
                                    matches.append((det._form, adj._form, noun._form, verb._form))
                                    ids.append((int(det.id)-1, int(adj.id)-1,int(noun.id)-1,int(verb.id)-1))
                                    values.append(value)
        if len(matches) == 1:
            return(matches[0], ids[0], sentence, values[0])
        else:
            return([], [], sentence, [])

        
class Russian_CONLL(CONLL):
    def __init__(self):
        super().__init__("Russian")

        # TO DO

class Spanish_CONLL(CONLL):
    def __init__(self):
        super().__init__("Spanish")

        # TO DO
