from transformers import BertTokenizer, CamembertTokenizer
from .book_keeping import *
import re
from string import punctuation

class My_Tokenizer():
    def __init__(self, language):
        self.language = language
        
        if self.language == "French":
            self.tokenizer = CamembertTokenizer.from_pretrained(Transformers[self.language])
        else:
            self.tokenizer = BertTokenizer.from_pretrained(Transformers[self.language])

    def prepare_sentence(self, sentence):
        return(self.tokenizer.cls_token + " " + sentence + " " + self.tokenizer.sep_token)

    def word_ids_to_token_ids(self, sentence):
        if sentence.split()[0] != self.tokenizer.cls_token:
            sentence = self.prepare_sentence(sentence)
        words = sentence.split()
        mapping = []
        counter = 0
        for i in range(len(words)):
            num_tokens = len(self.tokenizer.tokenize(words[i]))
            mapping.append((words[i], (counter, counter+num_tokens)))
            counter += num_tokens
        return(mapping)
