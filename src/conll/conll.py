import pyconll as pc
import pandas as pd
import pickle
from tqdm import tqdm

from utils.book_keeping import *
from utils.tokenizer_wrapper import *
from utils.loader import *
from .lexicon import Lexicon

class CONLL():
    def __init__(self, language):
        self.language = language

        self.conlls = []

        for conll in CoNLL[self.language]:
            self.conlls.append(pc.load_from_file(conll))

        self.features = Features[self.language]

        self.tokenizer = My_Tokenizer(self.language)
        self.lexicon = Lexicon(self.language)
        
    def create_dataset_classify(self, feature, n_examples=750, dump=True):
        """
        Function creates a dataframe with four columns
        Columns are [Word, ID, Sentence, Value]
        Records a Word in position ID in Sentence, marked for feature with Value

        Args:
        feature: str --- Some feature in Features[self.language]
        n_examples: int --- How many examples per feature value
        dump: bool --- If True, dump examples as .txt file to be used as Torch Dataset later
        """
        examples = []
        for dataset in self.conlls:
            for sentence in dataset:
                text = " ".join([token.form for token in sentence])
                text = self.tokenizer.prepare_sentence(text)
                for token in sentence:
                    if feature in token.feats:
                        try:
                            examples.append((token.form,
                                             int(token.id),
                                             text,
                                             list(token.feats[feature])[0]))
                        except:
                            continue
        df = pd.DataFrame(examples, columns=["Word", "ID", "Sentence", "Value"])
        values = []
        for value in self.features[feature]:
            values.append(df[df["Value"] == value].sample(n=n_examples))
        final_results = pd.concat(values)
        if dump:
            with open(Examples_Classify[self.language] + feature + ".tsv", "w") as fout:
                for _, row in final_results.iterrows():
                    line = "\t".join([str(row[col]) for col in final_results.columns])
                    fout.write(line + "\n")
        return(final_results)

    def create_dataset_agree(self, n_examples=2000, dump=True):
        """
        Similar to above, but capturing sets of words which agree
        Only concerned with subj-verb agreement for Number feature
        """
        examples = []
        for dataset in self.conlls:
            for sentence in dataset:
                agree, ids, sentence, value = self.retrieve_agree(sentence)
                text = " ".join([token.form for token in sentence])
                text = self.tokenizer.prepare_sentence(text)
                if len(agree) > 0:
                    examples.append((agree, ids, text, list(value)[0]))
        df = pd.DataFrame(examples, columns=["Agree", "IDs", "Sentence", "Value"])
        df = df.sample(n=n_examples)
        if dump:
            with open(Examples_Agree[self.language]+ "Number.tsv", "w") as fout:
                for _, row in df.iterrows():
                    line = "\t".join([str(row[col]) for col in df.columns])
                    fout.write(line + "\n")
        return(df)

    def calculate_ambiguities(self, feature, dump=False):
        dataset = load_examples_classify(self.language, feature)
        ambiguities = {}
        for example in tqdm(dataset):
            word = example[0]
            if word not in ambiguities:
                ambiguities[word] = self.lexicon.values_for_word(word, feature)
        if dump:
            pickle.dump(ambiguities, open(Ambiguity[self.language] + feature + ".p", "wb"))
        return(ambiguities)

        
    

class English_CONLL(CONLL):
    def __init__(self):
        super().__init__("English")

        self.POS = {"Det" : ["DET"],
                    "Noun" : ["PRON", "PROPN", "NOUN"],
                    "Verb" : ["AUX", "VERB"]}
    
    def retrieve_agree(self, sentence):
        # Takes a pyconll sentence object and returns Agree sets
        # Only captures subj-verb agreement
        pairs = []
        ids = []
        values = []
        nouns = [token for token in sentence
                 if token.upos in self.POS["Noun"]
                 and "Number" in token.feats]
        verbs = [token for token in sentence
                 if token.upos in self.POS["Verb"]
                 and "Number" in token.feats]
        for noun in nouns:
            for verb in verbs:
                if noun.head == verb.id or noun.head == verb.head: 
                    if noun.deprel == "nsubj":
                        try:
                            if verb.feats["VerbForm"]=={'Fin'}:
                                value = noun.feats["Number"]
                                pairs.append((noun._form,
                                              verb._form))
                                ids.append((int(noun.id), int(verb.id)))
                                values.append(value)
                        except:
                            continue
        # Only want sentences with one matching agree relationship 
        if len(pairs) == 1:
            return(pairs[0], ids[0], sentence, values[0])
        else:
            return([], [], sentence, [])        


        
class German_CONLL(CONLL):
    def __init__(self):
        super().__init__("German")

        self.POS = {"Det" : ["DET"],
                    "Adj" : ["ADJ"],
                    "Noun" : ["NOUN"],
                    "Verb" : ["VERB", "AUX"]}

    def retrieve_agree(self, sentence):
        # For now, only going to implement subj-verb agreement for number,
        # where subj is of form DET-ADJ-NOUN
        # Code could no doubt be optimized, but only needs to be run once and is fast enough
        matches = []
        ids = []
        values = []
        dets = [token for token in sentence
                if token.upos in self.POS["Det"]
                and "Number" in token.feats]
        adjs = [token for token in sentence
                if token.upos in self.POS["Adj"]
                and "Number" in token.feats]
        nouns = [token for token in sentence
                 if token.upos in self.POS["Noun"]
                 and "Number" in token.feats]
        verbs = [token for token in sentence
                 if token.upos in self.POS["Verb"]
                 and "Number" in token.feats]
        for det in dets:
            for adj in adjs:
                for noun in nouns:
                    for verb in verbs:
                        if det.head == adj.head == noun.id and noun.head == verb.id:
                            if det.feats["Number"] == adj.feats["Number"] == noun.feats["Number"] == verb.feats["Number"]:
                                if noun.deprel == "nsubj":
                                    value = det.feats["Number"]
                                    matches.append((det._form, adj._form, noun._form, verb._form))
                                    ids.append((int(det.id), int(adj.id), int(noun.id), int(verb.id)))
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
        
    def retrieve_agree(self, sentence):
        # Will do same as German; subj-verb agreement, where subj of form: DET-ADJ-NOUN
        matches = []
        ids = []
        values = []
        dets = [token for token in sentence
                if token.upos in self.POS["Det"]
                and "Number" in token.feats]
        adjs = [token for token in sentence
                if token.upos in self.POS["Adj"]
                and "Number" in token.feats]
        nouns = [token for token in sentence
                 if token.upos in self.POS["Noun"]
                 and "Number" in token.feats]
        verbs = [token for token in sentence
                 if token.upos in self.POS["Verb"]
                 and "Number" in token.feats]
        for det in dets:
            for adj in adjs:
                for noun in nouns:
                    for verb in verbs:
                        if det.head == adj.head == noun.id and noun.head == verb.id:
                            if det.feats["Number"] == adj.feats["Number"] == noun.feats["Number"] == verb.feats["Number"]:
                                if noun.deprel == "nsubj":
                                    value = det.feats["Number"]
                                    matches.append((det._form, adj._form, noun._form, verb._form))
                                    ids.append((det.id, adj.id, noun.id, verb.id))
                                    values.append(value)
        if len(matches) == 1:
            return(matches[0], ids[0], sentence, values[0])
        else:
            return([], [], sentence, [])
