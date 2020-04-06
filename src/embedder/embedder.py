import torch
from transformers import BertModel, CamembertModel
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from conll.lexicon import *
from utils.book_keeping import *
from utils.tokenizer_wrapper import *
from .dataset import *

class Transformer():
    def __init__(self, language, random=False, device="cuda"):
        self.language = language
        self.random = random
        self.device = device

        if self.language == "French":
            model = CamembertModel
        else:
            model = BertModel

        self.model = model.from_pretrained(Transformers[self.language],
                                           output_hidden_states=True,
                                           output_attentions=True)
        self.tokenizer = My_Tokenizer(self.language)
        self.lexicon = Lexicon(self.language)

        if self.random:
            self.model = model(config=self.model.config)

        self.model.to(self.device)
            
    def embed_dataset_classify(self, feature, batch_size=32, dump=False):
        """
        feature: str --- Feature for which to embed dataset. Takes examples from
        files in Data/Vectors/language and creates (X,y) datasets for classification
        batch_size: int --- Batch size
        dump: bool --- Whether or not to save the output
        """
        dataset = ClassificationDataset(self.language, feature)
        iterator = DataLoader(dataset, batch_size=batch_size)
        examples = []
        for batch in iterator:
            words, locs, sents, vals = batch
            
            tokenized = self.tokenizer.tokenizer.batch_encode_plus(list(sents),
                                                                   add_special_tokens=False,
                                                                   pad_to_max_length=True,
                                                                   return_tensors="pt")
            input_ids = tokenized["input_ids"].to(self.device)

            if input_ids.shape[1] > 250:
                tokenized = self.tokenizer.tokenizer.batch_encode_plus(list(sents),
                                                                       add_special_tokens=False,
                                                                       max_length = 250,
                                                                       pad_to_max_length=True,
                                                                       return_tensors="pt")
                input_ids = tokenized["input_ids"].to(self.device)

            attention_masks = tokenized["attention_mask"].to(self.device)


            with torch.no_grad():
                outputs = torch.stack(self.model(input_ids,
                                                 attention_mask=attention_masks)[2],
                                      dim=1)

            batch_examples = []
            # Loop over examples in batch---add good ones
            for i in range(len(outputs)): 
                # Collect indices for word of interest
                word, indices = self.tokenizer.word_ids_to_token_ids(sents[i])[locs[i]]
                # Makes sure that collecting the right indices---BertTokenizer sometimes
                # doesn't agree with CoNLL tokenization
                if words[i] == word:
                    X = torch.mean(outputs[i, :, indices[0]:indices[1], :], dim=1).to("cpu")
                    y = vals[i]
                    batch_examples.append((word, X, y))
                else:
                    pass

            examples += batch_examples

        if dump:
            if self.random:
                pickle.dump(examples, open(Vectors[self.language] + feature + "_random.p", "wb"))
            else:
                pickle.dump(examples, open(Vectors[self.language] + feature + ".p", "wb"))

        print("Successfully embedded {0} out of {1} possible examples.".format(len(examples),
                                                                               len(dataset)))
        return(examples)
                    

    def embed_dataset_agree(self, feature="Number", batch_size=32, dump=False):
        """
        feature: str --- Per paper, only implemented for Eng,Fr,De for Number
        batch_size: int --- Batch size
        dump: bool --- Whether or not to save output
        """
        dataset = AgreeDataset(self.language, feature)
        iterator = DataLoader(dataset, batch_size=batch_size)
        examples = []

        for batch in iterator:
            agrees, ids, sents, vals = batch

            tokenized = self.tokenizer.tokenizer.batch_encode_plus(list(sents),
                                                                   add_special_tokens=False,
                                                                   pad_to_max_length=True,
                                                                   return_tensors="pt")
            input_ids = tokenized["input_ids"].to(self.device)

            if input_ids.shape[1] > 250:
                tokenized = self.tokenizer.tokenizer.batch_encode_plus(list(sents),
                                                                       add_special_tokens=False,
                                                                       max_length = 250,
                                                                       pad_to_max_length=True,
                                                                       return_tensors="pt")
                input_ids = tokenized["input_ids"].to(self.device)

            attention_masks = tokenized["attention_mask"].to(self.device)

            
            with torch.no_grad():
                outputs = torch.stack(self.model(input_ids,
                                                 attention_mask=attention_masks)[3],
                                      dim=1)

            batch_examples = []
            #Loop over examples in batch---add good ones
            for i in range(len(outputs)):
                #[(str, (int,int))]
                words_to_tokens_map = self.tokenizer.word_ids_to_token_ids(sents[i]) 
                words = [item[0] for item in words_to_tokens_map]
                indices = [item[1] for item in words_to_tokens_map]
                # Ensure alignment between BERT tokenizer and conll ordering
                if all([agrees[j][i]==words[int(ids[j][i])] for j in range(len(agrees))]):
                    attns = torch.empty(12, 12, len(indices), len(indices))
                    for j in range(12):
                        for k in range(12):
                            X = self.process_attention_matrix(outputs[i][j][k], indices)
                            attns[j][k] = X
                    agree_set = (ids[0][i], ids[1][i])
                    sentence = sents[i]
                    batch_examples.append((agree_set, sentence, attns))
                    
                else:
                    pass

            examples += batch_examples
            print("Processed batch")
            
        if dump:
            if self.random:
                pickle.dump(examples, open(Attentions[self.language] + feature + "_random.p", "wb"))
            else:
                pickle.dump(examples, open(Attentions[self.language] + feature + ".p", "wb"))

        print("Successfully embedded {0} out of {1} possible examples.".format(len(examples),
                                                                               len(dataset)))
        return(examples)


    def process_attention_matrix(self, distributions, indices):
        """
        Takes an attention matrix of n x n, where n is num of tokens, and reshapes it
        to m x m, where m is number of words. Averages token distributions to get 
        word distributions, then sums token probabilities in columns to get word
        probability

        distribution: tensor --- An n x n attention output where n = sequence length
        indices: [(int,int)] --- Inidices showing where each word starts and ends
        """
        temp = torch.stack([torch.mean(distributions[index[0]:index[1]], dim=0)
                            for index in indices], dim=0)
        col_collapser = self.column_collapser(temp, indices)
        return(torch.matmul(temp.double(), col_collapser.double()))
        
        
    def column_collapser(self, dists, indices):
        """
        Helper function for self.process_attention_matrix

        dists: tensor --- temporary attn matrix of size m x n, rows already collapsed
        indices: [(int, int)] --- Start and end index for each word
        """
        n_cols = len(indices) # 
        C = np.zeros((dists.shape[-1], n_cols), int)
        for column in range(len(C.T)):
            start = indices[column][0]
            end = indices[column][1]
            C[start : end, column] = 1
        return(torch.tensor(C).to(self.device))
