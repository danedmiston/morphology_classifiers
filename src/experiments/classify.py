from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
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


class Classify():
    def __init__(self, language):
        self.language = language
        
        self.lexicon = Lexicon(self.language)
        self.features = Features[self.language]

        self.ambiguity_dict = {}

    def f2l(self, feature):
        return(feat2label(self.language, feature)[0])
        

    def evaluate_by_ambiguity(self, quadruples, feature):
        """
        Takes a list of triples of the form [(word, ambiguity, y_hat, y)]
        Partitions dataset by how ambiguous words are w.r.t. relevant feature

        Tests accuracy of these subsets. We want to see if performance
        is correlated with how ambiguous words are
        """
        scores = []
        possible_values = len(Features[self.language][feature])
        for i in range(1, possible_values+1):
            if i == 1:
                # Some entries not listed for a given feature; treat these as ambiguity 1
                subset = [item for item in quadruples if item[1] == 0 or item[1] == 1]
            else:
                subset = [item for item in quadruples if item[1] == i]
            if len(subset) > 0:
                y_hat = [item[2] for item in subset]
                y = [item[3] for item in subset]
                scores.append((i, f1_score(y, y_hat, average="weighted")))
        return(scores)

    def ambiguity_breakdown(self, feature):
        # Determines the ambiguity statistics for a feature
        breakdown = {}
        ambiguities = load_point_clouds(self.language, feature)[0]["Ambiguity"]
        possible_values = len(self.features[feature])
        for i in range(1, possible_values+1):
            if i == 1:
                temp = len([item for item in ambiguities if len(item)==0 or len(item)==1])
                breakdown[i] = temp/len(ambiguities)
            else:
                temp = len([item for item in ambiguities if len(item) == i])
                breakdown[i] = temp/len(ambiguities)
        return(breakdown)
            
        
        
class Clusterer(Classify):
    def __init__(self, language, method=KMeans):
        super().__init__(language)
        self.method = method
        

    def classify(self, layer, feature, average="weighted"):
        # This first part deals with calculating f1 score
        num_classes = len(Features[self.language][feature])
        clustering = self.method(n_clusters=num_classes, random_state=42)
        point_clouds = load_point_clouds(language=self.language, feature=feature)
        clustering.fit(np.stack(point_clouds[layer]["Vector"].values, axis=0))
        ground_truth = [self.f2l(feature)[list(item)[0]]
                        for item in point_clouds[layer]["Class"]]
        f1 = f1_score(ground_truth, clustering.labels_, average=average)

        # This second part concerns calculating accuracy for words which
        # are different levels of ambiguous
        # This let's us ask if there is a correlation between how ambiguous
        # a word is w.r.t. some feature, and how well a model can predict
        # the word's value in context
        words = [word for word in point_clouds[layer]["Word"]]
        ambis = [len(ambi) for ambi in point_clouds[layer]["Ambiguity"]]
        quads = [(words[i], ambis[i], clustering.labels_[i], ground_truth[i])
                 for i in range(len(words))]
        amb_scores = self.evaluate_by_ambiguity(quads, feature)
        #            
        return(f1, amb_scores)

    def run_cluster_experiment(self, feature, average="weighted"):
        f1s = []
        amb_scores_list = []
        with open(Cluster_Results[self.language] + feature + ".txt", "w") as fout:
            for i in range(13):
                f1, amb_scores = self.cluster(i, feature, average=average)
                f1s.append(f1)
                amb_scores_list.append(amb_scores)
                print(f1, amb_scores, "\n\n")

                fout.write("f1 for layer " + str(i) + ": " + str(f1) + "\n")
                fout.write("Ambiguity f1 for layer " + str(i) + ": " + str(amb_scores) + "\n\n\n")
            print("Average f1:", np.mean(f1s))
            fout.write("Average f1 for all layers:" + str(np.mean(f1s)) + "\n")
            possible_values = len(self.features[feature])
            for i in range(possible_values):
                try:
                    scores = np.stack([item[i][1] for item in amb_scores_list], axis=0)
                    mean = np.mean(scores) 
                    print("Average f1 for ambiguity=" + str(i+1) + ": " + str(mean) + "\n")
                    fout.write("Average f1 for ambiguity=" + str(i+1) + ": " + str(mean) + "\n")
                except:
                    continue
        

class Linear(Classify):
    def __init__(self, language):
        super().__init__(language)

    def prepare_datasets(self, feature):
        point_clouds = load_point_clouds(self.language, feature)
        
        train_set = {}
        test_set = {}

        for layer in range(13):
            train_set[layer] = {}
            test_set[layer] = {}

            vectors = point_clouds[layer][["Word", "Ambiguity", "Vector"]]
            ground_truth = [self.f2l(feature)[list(item)[0]] for item in point_clouds[layer]["Class"]]
            X_train, X_test, y_train, y_test = train_test_split(vectors, ground_truth,
                                                                test_size=0.15, random_state=42)
            train_set[layer]["X"] = X_train
            train_set[layer]["y"] = y_train

            test_set[layer]["X"] = X_test
            test_set[layer]["y"] = y_test

        return(train_set, test_set)
                    
    def classify(self, layer, feature, average="weighted"):
        train_set, test_set = self.prepare_datasets(feature)
        # Training
        X_train = np.stack(train_set[layer]["X"]["Vector"].values, axis=0)
        y_train = np.stack(train_set[layer]["y"], axis=0)
        clf = LogisticRegression(random_state=42, max_iter=2500)
        clf.fit(X_train, y_train)
        # Testing
        X_test = np.stack(test_set[layer]["X"]["Vector"].values, axis=0)
        y_test = np.stack(test_set[layer]["y"], axis=0)
        y_hat = clf.predict(X_test)
        f1 = f1_score(y_test, y_hat, average=average)
        
        words = [word for word in test_set[layer]["X"]["Word"]]
        ambis = [len(ambi) for ambi in test_set[layer]["X"]["Ambiguity"]]
        quads = [(words[i], ambis[i], y_hat[i], y_test[i])
                 for i in range(len(words))]
        amb_scores = self.evaluate_by_ambiguity(quads, feature)

        return(f1, amb_scores)
        
    def run_linear_experiment(self, feature, average="weighted"):
        f1s = []
        amb_scores_list = []
        with open(Linear_Results[self.language] + feature + ".txt", "w") as fout:
            for i in range(13):
                f1, amb_scores = self.linear(i, feature, average=average)
                f1s.append(f1)
                amb_scores_list.append(amb_scores)
                print(f1, amb_scores, "\n\n")

                fout.write("f1 for layer " + str(i) + ": " + str(f1) + "\n")
                fout.write("Ambiguity f1 for layer " + str(i) + ": " + str(amb_scores) + "\n\n\n")
            print("Average f1:", np.mean(f1s))
            fout.write("Average f1 for all layers:" + str(np.mean(f1s)) + "\n")
            possible_values = len(self.features[feature])
            for i in range(possible_values):
                try:
                    scores = np.stack([item[i][1] for item in amb_scores_list], axis=0)
                    mean = np.mean(scores) 
                    print("Average f1 for ambiguity=" + str(i+1) + ": " + str(mean) + "\n")
                    fout.write("Average f1 for ambiguity=" + str(i+1) + ": " + str(mean) + "\n")
                except:
                    continue


class NonLinear(Linear):
    def __init__(self, language, hidden=768, activation=F.relu):
        super().__init__(language)

        self.hidden = hidden
        self.activation = activation

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

            def forward(self, X):
                z1 = self.Layer1(X)
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
                                  max_epochs=2500,
                                  lr=0.1, iterator_train__shuffle=True,
                                  callbacks=[Checkpoint(monitor=monitor),
                                             EarlyStopping(patience=10,
                                                           threshold=0.00001),
                                             LRScheduler()],
                                  verbose=0)
        return(net)
        
    def classify(self, layer, feature, average="weighted"):
        num_classes = len(Features[self.language][feature])
        net = self.prepare_NN(num_classes)
        train_set, test_set = self.prepare_datasets(feature)
        # Training
        X_train = np.stack(train_set[layer]["X"]["Vector"].values, axis=0)
        y_train = np.stack(train_set[layer]["y"], axis=0)
        net.fit(X_train, y_train)
        # Testing
        X_test = np.stack(test_set[layer]["X"]["Vector"].values, axis=0)
        y_test = np.stack(test_set[layer]["y"], axis=0)
        y_hat = net.predict(X_test)
        f1 = f1_score(y_test, y_hat, average=average)

        words = [word for word in test_set[layer]["X"]["Word"]]
        ambis = [len(ambi) for ambi in test_set[layer]["X"]["Ambiguity"]]
        quads = [(words[i], ambis[i], y_hat[i], y_test[i])
                 for i in range(len(words))]
        amb_scores = self.evaluate_by_ambiguity(quads, feature)

        return(f1, amb_scores)

    
    def run_nonlinear_experiment(self, feature, average="weighted"):
        f1s = []
        amb_scores_list = []
        with open(NonLinear_Results[self.language] + feature + ".txt", "w") as fout:
            for i in range(13):
                f1, amb_scores = self.nonlinear(i, feature, average=average)
                f1s.append(f1)
                amb_scores_list.append(amb_scores)
                print(f1, amb_scores, "\n\n")

                fout.write("f1 for layer " + str(i) + ": " + str(f1) + "\n")
                fout.write("Ambiguity f1 for layer " + str(i) + ": " + str(amb_scores) + "\n\n\n")
            print("Average f1:", np.mean(f1s))
            fout.write("Average f1 for all layers:" + str(np.mean(f1s)) + "\n")
            possible_values = len(self.features[feature])
            for i in range(possible_values):
                try:
                    scores = np.stack([item[i][1] for item in amb_scores_list], axis=0)
                    mean = np.mean(scores) 
                    print("Average f1 for ambiguity=" + str(i+1) + ": " + str(mean) + "\n")
                    fout.write("Average f1 for ambiguity=" + str(i+1) + ": " + str(mean) + "\n")
                except:
                    continue


    
    
