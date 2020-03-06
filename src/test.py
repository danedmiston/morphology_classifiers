from conll.conll import *
from utils.book_keeping import *
from utils.tokenizer_helper import *
from transformers import BertModel, BertTokenizer, CamembertTokenizer
from embedder.embedder import *
from experiments.classify import * 
from experiments.agree import *
from sklearn.cluster import *
import torch.nn.functional as F
from conll.lexicon import Lexicon
import pickle
import pandas as pd
from utils.viz import *

from main import ambiguity_per_layer

import plotly.graph_objects as go


if __name__ == "__main__":

    """
    df = ambiguity_per_layer("Russian", "Case")

    print(df)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[i for i in range(13)], y=df[1].values,
                             mode="lines+markers",
                             name="Ambiguity=1"))    
    fig.add_trace(go.Scatter(x=[i for i in range(13)], y=df[2].values,
                             mode="lines+markers",
                             name="Ambiguity=2"))
    fig.add_trace(go.Scatter(x=[i for i in range(13)], y=df[3].values,
                             mode="lines+markers",
                             name="Ambiguity=3"))
    fig.add_trace(go.Scatter(x=[i for i in range(13)], y=df[4].values,
                             mode="lines+markers",
                             name="Ambiguity=4"))
    fig.add_trace(go.Scatter(x=[i for i in range(13)], y=df[5].values,
                             mode="lines+markers",
                             name="Ambiguity=5"))
    fig.update_layout(xaxis_title="Layer", yaxis_title="F1 score on subset of n-way ambiguous forms")
    fig.show()
    """

    """
    scores = {}
    for language in Languages:
        avg = []
        avg_feature_length = []
        classify = Classify(language)
        print(language)
        for feature in Features[language]:
            print(feature)
            breakdown = classify.ambiguity_breakdown(feature)
            avg_feature_length.append(len(breakdown))
            pct_ambiguous = 1 - breakdown[1]
            avg.append(pct_ambiguous)
            print(breakdown, "\n\n")
        scores[language] = (np.mean(avg), np.mean(avg_feature_length))

    for language in scores:
        print(language, scores[language], "\n\n")
    """

    agree = Agree("German", "Number")
    results = agree.run_chi2_average_experiment(output=False, renormalize=False)

    #for item in results:
    #    print(results[item], "\n\n")

    #clusterer = Clusterer("English")
    #clusterer.run_cluster_experiment("Mood")
    
    #linear = Linear("English")
    #linear.run_linear_experiment("Mood")
    #print(linear.linear(3, "Tense"))

    #print(nonlinear.nonlinear(3, "Tense"))




    
    #for feature in Features["French"]:
        #clusterer = Clusterer(language="French", feature=feature)
        #clusterer.run_cluster_experiment(average="weighted")
        #linear = Linear(language="French", feature=feature)
        #linear.run_linear_experiment(average="weighted")
    #    nonlinear = NonLinear(language="French", feature=feature)
    #    nonlinear.run_nonlinear_experiment(average="weighted")
        
    
    #german = German()

    #english = English()
    #english.embed_dataset_attention(feature="Number")

    #french = French_CONLL()
    #results = french.create_dataset_agree(feature="Number", n_examples=2000)
    #print(results)

    #french = French()
    #french.embed_dataset_attention("Number")

        
    #agree = Agree("French", "Number")
    #print(agree.run_agree_experiment(output=True))


    #agree.run_agree_experiment_layer(output=True)
    #agree.run_agree_experiment_layer(output=True)

    
    #for head in example["Attentions"]:
    #    print(agree.evaluate_distributions(example["IDs"], head))

    #for row in example["Attentions"][3]:
    #    print(row)
    #    print(agree.score_distribution(example["IDs"], row))


    #tokenized = german.tokenizer.tokenize(example["Sentence"])
    #word_tokenized = german.retokenize_words(tokenized)
        
    #heatmap(example["Attentions"][3], word_tokenized, 1, 4)






    
    


    #german = German()
    #german.embed_dataset_attention()
    
    #english = English()
    #english.embed_dataset_attention()

    
    #french = French_CONLL()

#    german = German()
    
    #for feature in Features["French"]:
    #    print(feature)
    #    if feature == "Mood":
    #        results = french.create_dataset_classify(feature=feature, n_examples=249, dump=True)
    #    else:
    #        results = french.create_dataset_classify(feature=feature, n_examples=750, dump=True)
    #    for value in Features["French"][feature]:
    #        print(value, len(results[results["Value"] == value]))
    #    print("\n\n")
        
    #data = german.conlls[0]
    #print("No problems")
    
    #for i in range(100):
    #    matches, ids, sentence, value = german.retrieve_agree(data[i], "Number")
    #    if len(matches) > 0:
    #        print(matches)
    #        print(ids)
    #        print(sentence.text)
    #        print(value)
    #        print("\n\n")
    #        input()
        
    
    
    #german = German()

    #sample = "Bei einem Angebot ohne Barabfindung böte das britische Unternehmen für eine Mannesmann-Aktie erneut lediglich Vodafone-Aktien , wenn auch mehr als bisher ."
    #tokenized = german.tokenizer.tokenize(sample)
    #print(sample)
    #word_tokenized = german.retokenize_words(tokenized)
    #print(word_tokenized)

    #reps, attns = german.embed(sample)

    #for i in range(2):
        #for j in range(2):
            #viz_attention(attns[i][j], word_tokenized, i, j)
            #input()
    
    #heatmap(attns[0][0], word_tokenized, 1, 1)
    
    
    #english = English()

    #english.embed_dataset_vectors("Person")
    
    #for feature in Features["Spanish"]:
    #    print(feature)
        #cluster = Clusterer(language="English", feature=feature)
        #print(cluster.cluster_by_layer(3, average="weighted"))
        #cluster.run_cluster_experiment(average="weighted")
    #    linear = Linear(language="Spanish", feature=feature)
    #    linear.run_linear_experiment(average="weighted")
#        nonlinear = NonLinear(language="English", feature=feature)
#        nonlinear.run_nonlinear_experiment(average="weighted")

    





    
    #clusterer = Clusterer(language="German", feature="Gender", method=KMeans)

    #nonlinear = NonLinear(language="German", feature="Gender")
    #print(nonlinear.nonlinear_by_layer(1))

    
    #for feature in Features["German"]:
    #    print(feature)
    #    clusterer = Clusterer(language="German", feature=feature, method=KMeans)
    #    clusterer.run_cluster_experiment(average="weighted")

#    for feature in Features["German"]:
#        print(feature)
#        linear = Linear(language="German", feature=feature)
#        linear.run_linear_experiment(average="weighted")

    #nonlinear = NonLinear(language="German", feature="Mood", activation=F.relu)
    #nonlinear.run_nonlinear_experiment(average="weighted")
    
#    for feature in Features["German"]:
#        print(feature)
#        nonlinear = NonLinear(language="German", feature=feature)
#        nonlinear.run_nonlinear_experiment(average="weighted")




    
    #sample = "Eine Reihe von Territorien stehen in enger Verbindung zum Vereinigten Königreich, sind aber völkerrechtlich von ihm abzugrenzen."

    #german = German()

    #print(type(german.extract_vectors("Reihe", 1, sample)[11]))

    #german.embed_dataset_vectors(feature="Number")
    
#    tokenized = german.tokenizer.tokenize(clean_sentence(sample))

#    print(tokenized, "\n\n")
#    print(german.retokenize_words(tokenized))

#    hiddens, attns = german.embed(sample)

#    print(attns.shape)

#    print(german.embed_dataset_vector("Gender"))
    
    #Testing CONLL 
    #conll = CONLL("German")
    #print(conll.features)



    #imps = pickle.load(open("french_imperatives.p", "rb"))
    #results = conll.create_dataset("Mood", n_examples=750, dump=False)
    #results = pd.concat([results, imps])
    #values = []
    #for value in Features["French"]["Mood"]:
    #    print(value, len(results[results["Value"] == value]))
    #    values.append(results[results["Value"] == value].sample(n=750))
    #final = pd.concat(values)
    #for value in Features["French"]["Mood"]:
    #    print(value, len(final[final["Value"] == value]))
    #pickle.dump(final, open(Examples["French"] + "Mood" + ".p", "wb"))

    
    #conll = CONLL("English")
    
    #for feature in Features["English"]:
    #    print(feature)
    #    results = conll.create_dataset(feature, n_examples=750, dump=True)
    #    for value in Features["English"][feature]:
    #        print(value, len(results[results["Value"] == value]))


        #if feature == "Mood":
        #    results = conll.create_dataset(feature, n_examples=381, dump=True)
        #else:
        #    results = conll.create_dataset(feature, n_examples=750, dump=True)
        #for value in Features["English"][feature]:
        #    print(value, len(results[results["Value"] == value]))
    
    #    print("\n\n")
    
    #print(conll.conlls)

    #print(conll.create_dataset("Mood", n_examples=750, dump=True))
    
    #for feature in conll.features:
    #    df = conll.create_dataset(feature, n_examples=750, dump=True)
    #    print(df)
