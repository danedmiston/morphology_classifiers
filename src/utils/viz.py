import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

def viz_heatmap(distributions, word_tokenized, layer, head):
    fig, ax = plt.subplots()
    im = ax.imshow(distributions)
    ax.set_xticks(np.arange(len(word_tokenized)))
    ax.set_yticks(np.arange(len(word_tokenized)))
    ax.set_xticklabels([item[0] for item in word_tokenized])
    ax.set_yticklabels([item[0] for item in word_tokenized])

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    ax.set_title("Attention heatmap for layer="+str(layer) + ", head="+str(head))
    fig.tight_layout()
    plt.show()

def viz_layers_heads(distributions):
    # Only works for 12x12 matrix, where rows are layers, columns attn heads
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

def viz_attention(distributions, word_tokenized, layer, head):
    fig = go.Figure(data=go.Heatmap(z=distributions[::-1],
                                    x=[item[0] for item in word_tokenized],
                                    y=[item[0] for item in word_tokenized][::-1],
                                    colorscale="gray"))
    fig.update_layout(title="Heatmap for Layer=" + str(layer) + ", Head=" + str(head))
    fig.show()