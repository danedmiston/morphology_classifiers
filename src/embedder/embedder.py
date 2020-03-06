from transformers import BertModel, BertTokenizer, CamembertModel, CamembertTokenizer
from utils.book_keeping import *
from utils.tokenizer_helper import *
from utils.loader import *
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from conll.lexicon import *
import pickle
from utils.viz import *

class Transformer():
    def __init__(self):
        # Will be initialized in subclass
        self.language = None
        self.model = None
        self.tokenizer = None

    def retokenize_words(self, tokenized):
        # This will be overwritten by subclass's implementation
        return(tokenized)


    def tokenize_by_word(self, tokenized):
        word_tokenized = self.retokenize_words(tokenized)
        return(word_tokenized)

    def embed(self, sentence):
        cleaned = clean_sentence(sentence)
        tokenized = self.tokenizer.tokenize(cleaned)
        word_tokenized = self.tokenize_by_word(tokenized)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized)
        tokens_tensor = torch.tensor([indexed_tokens])

        with torch.no_grad():
            outputs = self.model(tokens_tensor)
            hidden_states = torch.cat(outputs[2]) # layer x dim x sent-length
            attentions = torch.cat(outputs[3]) # layer x attn-head x sent-length x sent-length

        # This bit organizes hidden layer representations
        embeddings = [(word[0], torch.mean(hidden_states[:, word[1]], dim=1, keepdim=True).squeeze())
                      for word in word_tokenized]
        words = [pd.DataFrame.from_dict({entry[0] : [entry[1][i].numpy() for i in range(13)]},
                                        orient="index",
                                        columns = [i for i in range(13)]) #Layer representations
                 for entry in embeddings]
        representations = pd.concat(words)

        # This bit organizes attention representations

        indices = [item[1] for item in word_tokenized]
        word_attentions = np.empty((12,12,len(word_tokenized), len(word_tokenized)))
        for i in range(12):
            for j in range(12):
                temp = [torch.mean(attentions[i, j, word[1]], dim=0, keepdim=True).squeeze().numpy()
                        for word in word_tokenized]
                temp = [collapse_columns(item,indices) for item in temp]
                temp = np.stack(temp, axis=0)
                word_attentions[i,j] = temp
        
        return(representations, word_attentions)


    def extract_vectors(self, word, ID, sentence):
        """
        Extracts the contextualized embedding of word given sentence

        Args:
        word: str --- Word to be embedded
        ID: int --- Place of word in sentence
        sentence: str --- Sentence which provides context for word
        """
        assert(sentence.split()[ID] == word)
        representations, _ = self.embed(sentence)
        return(representations.iloc[ID])

    
    def embed_dataset_vectors(self, feature):
        """
        Takes a dataset in the Examples folder, which is a dataframe of [Word, ID, Sentence, Value],
        and returns/saves dataframes (one for each layer) of shape [Word, Ambiguity, Vector, Class], 
        where Vector is the contextualized Word vector of the word in the context of the sentence, 
        and Ambiguity is how ambiguous a certain entry is w.r.t. the relevant feature.
        
        Args:
        feature: str --- Some string in Features[self.language]
        """
        examples = load_examples_classify(self.language, feature)

        
        layers = {}
        for layer in range(13):
            layers[layer] = []

        for id_num, row in tqdm(examples.iterrows()):
            try:
                word = row["Word"]
                # How ambiguous is the word w.r.t. the feature
                ambiguity = self.lexicon.values_for_word(word, feature)
                reps = self.extract_vectors(word=word, ID=row["ID"], sentence=row["Sentence"])
                value = row.Value
                for layer in range(13):
                    layers[layer].append((word, ambiguity, reps[layer], value))
            except:
                continue                    
        df = {}
        for layer in range(13):
            df[layer] = pd.DataFrame.from_records(layers[layer],
                                                  columns=["Word", "Ambiguity", "Vector", "Class"])

        print(len(examples))
        print(len(df[0]))
        pickle.dump(df, open(Point_Clouds[self.language] + feature + ".p", "wb"))


    def extract_attention(self, agree, IDs, sentence):
        """
        Given a dataframe entry from Examples_Agree, extract the attention representations,
        and provide information necessary for building dataset for experimentation
        """
        for i in range(len(agree)):
            assert(agree[i] == sentence.split()[IDs[i]])
        _, attns = self.embed(sentence)
        return(attns)

    def embed_dataset_attention(self, feature="Number"):
        """
        See function embed_dataset_vectors above for description
        """
        examples = load_examples_attention(self.language, feature)

        layers = {}
        for layer in range(12):
            layers[layer] = []

        for id_num, row in tqdm(examples.iterrows()):
            try:
                agree = row["Agree"]
                ids = row["IDs"]
                sentence = row["Sentence"]
                attns = self.extract_attention(agree, ids, sentence)
                value = row["Value"]
                for layer in range(12):
                    layers[layer].append((agree, ids, attns[layer], sentence, value))
            except:
                continue
        df = {}
        for layer in range(12):
            df[layer] = pd.DataFrame.from_records(layers[layer],
                                                  columns=["Agree", "IDs", "Attentions",
                                                           "Sentence", "Value"])
        print(len(examples))
        print(len(df[0]))
        pickle.dump(df, open(Attentions[self.language] + feature + ".p", "wb"))
                

        
    
class German(Transformer):
    def __init__(self):
        super().__init__()

        self.language = "German"

        self.model = BertModel.from_pretrained(Transformers["German"],
                                               output_hidden_states=True,
                                               output_attentions=True)
        self.tokenizer = BertTokenizer.from_pretrained(Transformers["German"])

        self.lexicon = Lexicon(self.language)
        
    def retokenize_words(self, tokenized):
        # Needs to be implemented on a model-by-model basis, as each tokenizer is different
        words = []
        for i in range(len(tokenized)):
            if tokenized[i][:2] != "##":
                if i != 0:
                    words.append(("".join([item[0] for item in word]), [item[1] for item in word]))
                word = []
                word.append((tokenized[i], i))
            else:
                word.append((tokenized[i][2:], i))
        words.append(("".join([item[0] for item in word]), [item[1] for item in word]))
        return(words)
        

class Russian(Transformer):
    def __init__(self):
        super().__init__()

        self.language = "Russian"

        self.model = BertModel.from_pretrained(Transformers["Russian"],
                                               output_hidden_states=True,
                                               output_attentions=True)
        self.tokenizer = BertTokenizer.from_pretrained(Transformers["Russian"])

        self.lexicon = Lexicon(self.language)

    def retokenize_words(self, tokenized):
        # Needs to be implemented on a model-by-model basis, as each tokenizer is different
        # It just so happens, German's and Russian's tokenizers act identically
        words = []
        for i in range(len(tokenized)):
            if tokenized[i][:2] != "##":
                if i != 0:
                    words.append(("".join([item[0] for item in word]), [item[1] for item in word]))
                word = []
                word.append((tokenized[i], i))
            else:
                word.append((tokenized[i][2:], i))
        words.append(("".join([item[0] for item in word]), [item[1] for item in word]))
        return(words)


class Spanish(Transformer):
    def __init__(self):
        super().__init__()

        self.language = "Spanish"

        self.model = BertModel.from_pretrained(Transformers["Spanish"],
                                               output_hidden_states=True,
                                               output_attentions=True)
        self.tokenizer = BertTokenizer.from_pretrained(Transformers["Spanish"])

        self.lexicon = Lexicon(self.language)

    def retokenize_words(self, tokenized):
        # Needs to be implemented on a model-by-model basis, as each tokenizer is different
        # It just so happens, Spanish's tokenizer in this case ALSO works identically
        words = []
        for i in range(len(tokenized)):
            if tokenized[i][:2] != "##":
                if i != 0:
                    words.append(("".join([item[0] for item in word]), [item[1] for item in word]))
                word = []
                word.append((tokenized[i], i))
            else:
                word.append((tokenized[i][2:], i))
        words.append(("".join([item[0] for item in word]), [item[1] for item in word]))
        return(words)

    
class English(Transformer):
    def __init__(self):
        super().__init__()

        self.language = "English"

        self.model = BertModel.from_pretrained(Transformers["English"],
                                               output_hidden_states=True,
                                               output_attentions=True)
        self.tokenizer = BertTokenizer.from_pretrained(Transformers["English"])

        self.lexicon = Lexicon(self.language)

    def retokenize_words(self, tokenized):
        # Needs to be implemented on a model-by-model basis, as each tokenizer is different
        # Apparently, all the tokenizers are the same, since they are all Bert-base...
        words = []
        for i in range(len(tokenized)):
            if tokenized[i][:2] != "##":
                if i != 0:
                    words.append(("".join([item[0] for item in word]), [item[1] for item in word]))
                word = []
                word.append((tokenized[i], i))
            else:
                word.append((tokenized[i][2:], i))
        words.append(("".join([item[0] for item in word]), [item[1] for item in word]))
        return(words)


class French(Transformer):
    def __init__(self):
        super().__init__()

        self.language = "French"

        self.model = CamembertModel.from_pretrained(Transformers["French"],
                                                    output_hidden_states=True,
                                                    output_attentions=True)
        self.tokenizer = CamembertTokenizer.from_pretrained(Transformers["French"])

        self.lexicon = Lexicon(self.language)

    def retokenize_words(self, tokenized):
        # A different tokenizer!
        words = []
        for i in range(len(tokenized)):
            if tokenized[i][0] == "▁":
                if i != 0:
                    words.append(("".join([item[0] for item in word]), [item[1] for item in word]))
                word = []
                word.append((tokenized[i][1:], i))
            else:
                word.append((tokenized[i], i))
        words.append(("".join([item[0] for item in word]), [item[1] for item in word]))
        return(words)
        

        
