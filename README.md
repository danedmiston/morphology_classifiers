# Morphology Classifiers

Code used for paper "A Systematic Analysis of Morphological Content in BERT Models for Multiple Languages" by Daniel Edmiston.

## Instructions

To run the experiments from the paper, follow the steps listed below. 

(1) Download lexicon files from http://atoll.inria.fr/~sagot/UDLexicons.0.2.zip .

Specifically, extract UDLex_English-EnLex.conllul , UDLex_French-Lefff.conllul , UDLex_German-Apertium.conllul , UDLex_Russian-Apertium.conllul , and UDLex_Spanish-Leffe.conllul ,
and place each lexicon in the proper /Datasets/Lexicons/(language) directory. 

(2) Run the following command to download all conllu treebank files and place them in the proper /Datasets/CoNLL/(language) directory. Specifics of which treebanks are used can be found in Appendix A of the paper. 

```
python main.py --download_conlls True
```

(3) Download all necessary transformer models. See specifics in Table 1 of paper.

(4) Build the datasets necessary for classification. Details of how this is done are in the paper.

```
python main.py --build_datasets_classify True
```

(5) Build the datasets necessary for the agree task.

```
python main.py --build_datasets_agree True
```

(6) Calculate the ambiguities. This may take some time...

```
python main.py --calculate_ambiguities True
```

(7) The first command of the following embeds the examples sampled from the CoNLLs for classification. 
The second embeds for the agree task. (Note: A few examples may be discarded due to mismatches between
BERT tokenizer and CONLL tokenization.)

```
python main.py --embed_datasets_classify True
python main.py --embed_datasets_agree True
```

(8) To produce table 2:

```
python main.py --calculate_statistics True
```

(9) The following recreates Table 3; toggle '--random True' to recreate Table 8.

```
python main.py --test_features_classify True
```

(10) To recreate Table 9 (from which Figure 1 was created, see /src/utils/viz.py), run the following (toggling '--random True' to recreate Table 10):

```
python main.py --test_layers_classify True
```

(11) To reproduce Table 4:

```
python main.py --test_ambiguity_correlation True
```

(12) To reproduce scores for Figure 2:

```
python main.py --test_ambiguity_per_layer True --language German --feature Case
```

(13) To reproduce results in Tables 5-7 and 11-13:

```
python main.py --test_agree True
```










