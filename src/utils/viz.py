import plotly.graph_objects as go
from .loader import *
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def viz_lang_layers(method="Linear"):
    if method == "Cluster":
        indices = [0,3,6,9,12]
    elif method == "Linear":
        indices = [1,4,7,10,13]
    elif method == "NonLinear":
        indices = [2,5,8,11,14]
    data = pickle.load(open("../Results/layer_scores.p", "rb"))
    table = data.iloc[:-1, indices]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=table.index, y=table["English"], name="English"))
    fig.add_trace(go.Scatter(x=table.index, y=table["French"], name="French"))
    fig.add_trace(go.Scatter(x=table.index, y=table["German"], name="German",
                             line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=table.index, y=table["Russian"], name="Russian",
                             line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=table.index, y=table["Spanish"], name="Spanish"))

    fig.update_layout(xaxis_title="Layer", yaxis_title="Weighted F1 Score")

    fig.show()


def viz_layer_ambiguity(language, feature):
    data = pickle.load(open("../Results/amb_per_layer_" + language + "_" + feature + ".p", "rb"))
    num_amb = data.shape[1]
    
    fig = go.Figure()

    for i in range(1, num_amb+1):
        fig.add_trace(go.Scatter(x=data.index, y=data[i], name="Ambiguity="+str(i),
                                 line=dict(dash="dash")))


    fig.update_layout(xaxis_title="Layer", yaxis_title="F1 score on subset of n-way ambiguous forms")

    fig.show()


def viz_layers_heads(distributions):
    fig, ax = plt.subplots()
    im = ax.imshow(distributions)
    ax.set_xticks(np.arange(12))
    ax.set_yticks(np.arange(12))
    ax.set_xticklabels(["Head="+str(i) for i in range(1,13)])
    ax.set_yticklabels(["Layer="+str(i) for i in range(1,13)])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_title("Heatmap for agree information")
    fig.tight_layout()
    plt.show()



    
