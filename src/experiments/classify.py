from sklearn.metrics import f1_score
import numpy as np
from sklearn.linear_model import LogisticRegression
from conll.lexicon import Lexicon
from utils.loader import *
from sklearn.cluster import *
from utils.book_keeping import *
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier
from skorch.callbacks import Checkpoint, EarlyStopping, LRScheduler
from tqdm import tqdm
import pickle
import warnings
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings("ignore")

class Classifier():
    def __init__(self, language):
        self.language = language

        self.lexicon = Lexicon(self.language)
        self.features = Features[self.language]

        
    def f2l(self, feature):
        return(feat2label(self.language, feature)[0])
    
    def classify(self):
        # To be over-ridden at sub-class level
        pass

    def evaluate_by_ambiguity(self, quadruples, feature):
        """
        Takes a list of quadruples of the form [(word, ambiguity, y_hat, y)]
        Partitions dataset by how ambiguous words are w.r.t. relevant feature

        Tests accuracy of these subsets. We want to see if performance is
        correlated with how ambiguous (i.e. syncretic) words are for a feature.
        """
        scores = []
        possible_values = len(Features[self.language][feature])
        for i in range(1, possible_values+1):
            if i == 1:
                # Some entries not listed for a given feature; treat these
                # as ambiguity=1
                subset = [item for item in quadruples if item[1]==0 or item[1]==1]
            else:
                subset = [item for item in quadruples if item[1]==i]
            if len(subset) > 0:
                y_hat = [item[2] for item in subset]
                y = [item[3] for item in subset]
                scores.append((i, f1_score(y, y_hat, average="weighted")))
        return(scores)

    def ambiguity_breakdown(self, feature):
        # Determines ambiguity/syncretism statistics for a feature
        breakdown = {}
        values = []
        ambiguities = load_ambiguities(self.language, feature)
        dataset = load_examples_classify(self.language, feature)
        possible_values = len(self.features[feature])
        for example in dataset:
            word = example[0]
            values.append(len(ambiguities[word]))
        for i in range(1, possible_values+1):
            if i == 1:
                temp = len([item for item in values
                            if item==0 or item==1])
                breakdown[i] = temp/len(dataset)
            else:
                temp = len([item for item in values
                            if item==i])
                breakdown[i] = temp/len(dataset)
        return(breakdown)

    def ambiguity_correlation(self):
        # Determines how correlated ambiguity of features is with performance on classification
        scores = {}
        scores["percent"] = {}
        scores["length"] = {}
        avg_f1s = []
        pct_ambiguous_list = []
        feature_lengths = []
        for feature in self.features:
            print("Processing feature {0}".format(feature))
            breakdown = self.ambiguity_breakdown(feature)
            pct_ambiguous = 1 - breakdown[1]
            pct_ambiguous_list.append(pct_ambiguous)
            feature_lengths.append(len(breakdown))
            f1s = []
            for i in range(13):
                print("Processing layer {0}".format(i))
                f1, _ = self.classify(i, feature)
                f1s.append(f1)
            avg_f1s.append(np.mean(f1s))
        scores["percent"] = (spearmanr(avg_f1s, pct_ambiguous_list),
                             pearsonr(avg_f1s, pct_ambiguous_list))
        scores["length"] = (spearmanr(avg_f1s, feature_lengths),
                            pearsonr(avg_f1s, feature_lengths))
        return(scores)

    def calculate_statistics(self):
        feature_lengths = []
        counter = 0
        ambiguous = 0
        for feature in self.features:
            feature_lengths.append(len(self.features[feature]))
            data = load_vectors(language=self.language, feature=feature, random=self.random)
            examples = [item for item in data if all(np.isfinite(item[1][0].numpy()))]
            ambiguities = load_ambiguities(self.language, feature)
            counter += len(examples)
            for ex in examples:
                if len(ambiguities[ex[0]]) > 1:
                    ambiguous += 1
        return(ambiguous/counter, np.mean(feature_lengths))
                    
            
class Clusterer(Classifier):
    def __init__(self, language, method=KMeans, random=False):
        super().__init__(language)
        self.method = method
        self.random = random

    def prepare_datasets(self, layer, feature):
        data = load_vectors(language=self.language, feature=feature, random=self.random)
        examples = [item for item in data if all(np.isfinite(item[1][layer].numpy()))]
        return(examples)
        
    def classify(self, layer, feature, average="weighted"):
        # This first part deals with calculating f1 score
        num_classes = len(self.features[feature])
        clustering = self.method(n_clusters=num_classes, random_state=42)
        data = self.prepare_datasets(layer, feature)
        vectors = np.stack([item[1][layer].numpy() for item in data], axis=0)
        ground_truth = np.stack([self.f2l(feature)[item[2]] for item in data], axis=0)
        clustering.fit(vectors)
        f1 = f1_score(ground_truth, clustering.labels_, average=average)

        # Second part calculates accuracy for words which have different
        # levels of ambiguity. Used to calculate correlation between
        # accuracy and how ambiguous words are
        ambiguities = load_ambiguities(self.language, feature)
        words = [item[0] for item in data]
        ambis = [len(ambiguities[word]) for word in words]
        quads = [(words[i], ambis[i], clustering.labels_[i], ground_truth[i])
                 for i in range(len(words))]
        amb_scores = self.evaluate_by_ambiguity(quads, feature)
        return(f1, amb_scores)


class Linear(Classifier):
    def __init__(self, language, random=False):
        super().__init__(language)
        self.random = random

    def prepare_datasets(self, layer, feature):
        data = load_vectors(language=self.language, feature=feature, random=self.random)
        data = [item for item in data if all(np.isfinite(item[1][layer].numpy()))]
        
        train_set = {}
        test_set = {}

        predictors = [(item[0], item[1][layer].numpy()) for item in data]
        ground_truth = np.stack([self.f2l(feature)[item[2]] for item in data], axis=0)
        X_train, X_test, y_train, y_test = train_test_split(predictors, ground_truth,
                                                            test_size=0.15, random_state=42)

        train_set["X"] = X_train
        train_set["y"] = y_train
        test_set["X"] = X_test
        test_set["y"] = y_test

        return(train_set, test_set)

    def classify(self, layer, feature, average="weighted"):
        train_set, test_set = self.prepare_datasets(layer, feature)
        # Training
        X_train = np.stack([item[1] for item in train_set["X"]], axis=0)
        y_train = train_set["y"]
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X_train, y_train)
        # Testing
        X_test = np.stack([item[1] for item in test_set["X"]], axis=0)
        y_test = test_set["y"]
        y_hat = clf.predict(X_test)
        f1 = f1_score(y_test, y_hat, average=average)

        ambiguities = load_ambiguities(self.language, feature)
        words = [item[0] for item in test_set["X"]]
        ambis = [len(ambiguities[word]) for word in words]
        quads = [(words[i], ambis[i], y_hat[i], y_test[i])
                 for i in range(len(words))]
        amb_scores = self.evaluate_by_ambiguity(quads, feature)
        return(f1, amb_scores)


class NonLinear(Linear):
    def __init__(self, language, random=False, hidden=768, activation=F.relu, device="cuda"):
        super().__init__(language, random)

        self.hidden = hidden
        self.activation = activation
        self.device = device

    def prepare_NN(self, num_classes):
        class ThreeLayerNN(nn.Module):
            def __init__(self, hidden, num_classes, activation):
                super(ThreeLayerNN, self).__init__()

                self.hidden = hidden
                self.num_classes = num_classes
                self.activation = activation

                self.Layer1 = nn.Linear(768, self.hidden, bias=True)
                self.Layer2 = nn.Linear(self.hidden, self.hidden, bias=True)
                self.Output = nn.Linear(self.hidden, self.num_classes, bias=True)

            def forward(self, x):
                z1 = self.Layer1(x)
                a1 = self.activation(z1)
                z2 = self.Layer2(a1)
                a2 = self.activation(z2)
                y_hat = F.softmax(a2)
                return(y_hat)

        monitor = lambda net: all(net.history[-1, ("train_loss_best", "valid_loss_best")])

        net = NeuralNetClassifier(module=ThreeLayerNN,
                                  module__hidden=self.hidden,
                                  module__num_classes=num_classes,
                                  module__activation=self.activation,
                                  max_epochs=1000,
                                  lr=0.1,
                                  iterator_train__shuffle=True,
                                  callbacks=[Checkpoint(monitor=monitor),
                                             EarlyStopping(patience=5,
                                                           threshold=0.0001),
                                             LRScheduler()],
                                  verbose=0,
                                  device=self.device,
                                  batch_size=32)
        return(net)

    def classify(self, layer, feature, average="weighted"):
        num_classes = len(self.features[feature])
        net = self.prepare_NN(num_classes)
        train_set, test_set = self.prepare_datasets(layer, feature)
        # Training
        X_train = np.stack([item[1] for item in train_set["X"]], axis=0)
        y_train = train_set["y"]
        net.fit(X_train, y_train)
        # Testing
        X_test = np.stack([item[1] for item in test_set["X"]], axis=0)
        y_test = test_set["y"]
        y_hat = net.predict(X_test)
        f1 = f1_score(y_test, y_hat, average=average)

        ambiguities = load_ambiguities(self.language, feature)
        words = [item[0] for item in test_set["X"]]
        ambis = [len(ambiguities[word]) for word in words]
        quads = [(words[i], ambis[i], y_hat[i], y_test[i])
                 for i in range(len(words))]
        amb_scores = self.evaluate_by_ambiguity(quads, feature)
        return(f1, amb_scores)

