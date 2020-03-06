from glob import glob



# Datasets

Languages = ["English", "French", "German", "Russian", "Spanish"]

CoNLL = {"German" : glob("../Datasets/CoNLL/German/*.conllu"),
         "Russian" : glob("../Datasets/CoNLL/Russian/*.conllu"), 
         "Spanish" : glob("../Datasets/CoNLL/Spanish/*.conllu"),
         "English" : glob("../Datasets/CoNLL/English/*.conllu"),
         "French" : glob("../Datasets/CoNLL/French/*.conllu")
}

Lexicons = {"German" : "../Datasets/Lexicons/German/UDLex_German-Apertium.conllul",
            "Russian" : "../Datasets/Lexicons/Russian/UDLex_Russian-Apertium.conllul",
            "Spanish" : "../Datasets/Lexicons/Spanish/UDLex_Spanish-Leffe.conllul",
            "English" : "../Datasets/Lexicons/English/UDLex_English-EnLex.conllul",
            "French" : "../Datasets/Lexicons/French/UDLex_French-Lefff.conllul"}

Features = {"German" : {"Case" : [{'Nom'}, {'Gen'}, {'Dat'}, {'Acc'}],
                        "Gender" : [{'Fem'}, {'Neut'}, {'Masc'}],
                        "Mood" : [{'Ind'}, {'Sub'}, {'Imp'}],
                        "Number" : [{'Sing'}, {'Plur'}],
                        "Person" : [{'1'}, {'3'}, {'2'}],
                        "Tense" : [{'Pres'}, {'Past'}],
                        "VerbForm" : [{'Fin'}, {'Inf'}, {'Part'}]},
            "Russian" : {"Case" : [{'Nom'}, {'Gen'}, {'Acc'}, {'Loc'}, {'Dat'}, {'Ins'}],
                         "Gender" : [{'Masc'}, {'Neut'}, {'Fem'}],
                         "Number" : [{'Sing'}, {'Plur'}],
                         "Tense" : [{'Pres'}, {'Past'}, {'Fut'}],
                         "VerbForm" : [{'Part'}, {'Fin'}, {'Inf'}, {'Conv'}],
                         "Mood" : [{'Ind'}, {'Cnd'}, {'Imp'}],
                         "Person" : [{'3'}, {'1'}, {'2'}]},
            "Spanish" : {"Gender" : [{'Masc'}, {'Fem'}],
                         "Number" : [{'Sing'}, {'Plur'}],
                         "Person" : [{'3'}, {'1'}, {'2'}],
                         "Mood" : [{'Ind'}, {'Sub'}, {'Cnd'}, {'Imp'}],
                         "Tense" : [{'Past'}, {'Pres'}, {'Fut'}, {'Imp'}],
                         "VerbForm" : [{'Fin'}, {'Inf'}, {'Part'}, {'Ger'}]},
            "English" : {"Number" : [{'Sing'}, {'Plur'}],
                         "Mood" : [{'Ind'}, {'Imp'}],
                         "Tense" : [{'Past'}, {'Pres'}],
                         "VerbForm" : [{'Fin'}, {'Inf'}, {'Ger'}, {'Part'}],
                         "Person" : [{'1'}, {'3'}, {'2'}]},
            "French" : {"Gender" : [{'Masc'}, {'Fem'}],
                        "Number" : [{'Sing'}, {'Plur'}],
                        "Mood" : [{'Ind'}, {'Sub'}, {'Cnd'}, {'Imp'}],
                        "Person" : [{'3'}, {'1'}, {'2'}],
                        "Tense" : [{'Pres'}, {'Past'}, {'Imp'}, {'Fut'}],
                        "VerbForm" : [{'Fin'}, {'Inf'}, {'Part'}]}
}


def feat2label(language, feature):
    values = Features[language][feature]
    f2l = [(list(values[i])[0], i) for i in range(len(values))]
    l2f = [(v, k) for k, v in f2l]
    return(dict(f2l), dict(l2f))
    

Examples_Classify = {"German" : "../Datasets/Examples_Classify/German/",
                     "Russian" : "../Datasets/Examples_Classify/Russian/",
                     "Spanish" : "../Datasets/Examples_Classify/Spanish/",
                     "English" : "../Datasets/Examples_Classify/English/",
                     "French" : "../Datasets/Examples_Classify/French/"}

Examples_Agree = {"German" : "../Datasets/Examples_Agree/German/",
                  "Russian" : "../Datasets/Examples_Agree/Russian/",
                  "Spanish" : "../Datasets/Examples_Agree/Spanish/",
                  "English" : "../Datasets/Examples_Agree/English/",
                  "French" : "../Datasets/Examples_Agree/French/"}


Transformers = {"German" : "bert-base-german-dbmdz-cased",
                "Russian" : "/home/dan/Transformers/rubert/",
                "Spanish" : "/home/dan/Transformers/beto/",
                "English" : "bert-base-cased",
                "French" : "camembert-base"}

Point_Clouds = {"German" : "../Datasets/Point_Clouds/German/",
                "Russian" : "../Datasets/Point_Clouds/Russian/",
                "Spanish" : "../Datasets/Point_Clouds/Spanish/",
                "English" : "../Datasets/Point_Clouds/English/",
                "French" : "../Datasets/Point_Clouds/French/"}

Attentions = {"German" : "../Datasets/Attentions/German/",
              "Russian" : "../Datasets/Attentions/Russian/",
              "Spanish" : "../Datasets/Attentions/Spanish/",
              "English" : "../Datasets/Attentions/English/",
              "French" : "../Datasets/Attentions/French/"}

# Classifier Results
Cluster_Results = {"German" : "../Results/Cluster/German/",
                   "Russian" : "../Results/Cluster/Russian/",
                   "Spanish" : "../Results/Cluster/Spanish/",
                   "English" : "../Results/Cluster/English/",
                   "French" : "../Results/Cluster/French/"}

Linear_Results = {"German" : "../Results/Linear/German/",
                  "Russian" : "../Results/Linear/Russian/",
                  "Spanish" : "../Results/Linear/Spanish/",
                  "English" : "../Results/Linear/English/",
                  "French" : "../Results/Linear/French/"}

NonLinear_Results = {"German" : "../Results/NonLinear/German/",
                     "Russian" : "../Results/NonLinear/Russian/",
                     "Spanish" : "../Results/NonLinear/Spanish/",
                     "English" : "../Results/NonLinear/English/",
                     "French" : "../Results/NonLinear/French/"}

# Agree Results
Agree_Results = {"German" : "../Results/Agree/German/",
                 "English" : "../Results/Agree/English/",
                 "French" : "../Results/Agree/French/"}

Tables = "../Results/Tables/"
