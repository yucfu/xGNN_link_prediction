import networkx as nx
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt


def visualize_explanation(
        computation_graph,
        edge_index,
        edge_mask,
        source_node,
        target_node,
        node_mask=None,
        seed=0,
        figsize=(10, 10),
        ax=None
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    nodelist = list(computation_graph.nodes())
    edgelist = list(computation_graph.edges())

    edge_mask_dict = {e: [] for e in edgelist}
    for i, e in enumerate(edge_index.T):
        s, t = e.numpy()
        if (s, t) in edgelist or (t, s) in edgelist:
            edge_mask_dict[tuple(sorted((s, t)))].append(edge_mask[i])
    edge_mask_dict = {k: v[0] if np.abs(v[0]) > np.abs(v[1]) else v[1] for k, v in edge_mask_dict.items()}

    edge_width = []
    for s, t in edgelist:
        edge_width.append(np.abs(edge_mask_dict[tuple(sorted((s, t)))]))
    edge_width = np.array(edge_width)
    draw_edge_width = 3 * edge_width / np.abs(edge_width).max()
    # draw_edge_width = edge_width

    nodecolors = ['white' if n not in [source_node, target_node] else 'green' for n in nodelist]
    edgecolors = ['black' if edge_mask_dict[tuple(sorted((s, t)))] >= 0. else 'darkorange' \
                  for s, t in edgelist]

    pos = nx.spring_layout(computation_graph, seed=seed)
    if node_mask is None:
        nx.draw_networkx_nodes(computation_graph, pos, node_color=nodecolors, edgecolors='black', ax=ax)
    else:
        draw_edge_width = np.ones(edge_width.shape)
        # nodealphas = [node_mask[n] for n in computation_graph.nodes()]
        # nodecolors = ['red' if n == 1. else 'white' for n in node_mask]

        for i in range(len(nodecolors)):
            if node_mask[i] == 1 and nodecolors[i] != 'green':
                nodecolors[i] = 'red'

        nx.draw_networkx_nodes(computation_graph, pos, node_color=nodecolors, edgecolors='black', ax=ax)
        # alpha = nodealphas

    nx.draw_networkx_labels(computation_graph, pos, ax=ax)
    nx.draw_networkx_edges(computation_graph,
                           pos,
                           edge_color=edgecolors,
                           edgelist=edgelist,
                           width=draw_edge_width, ax=ax
                           )
    ax.set_title(f"{source_node}->{target_node}")
    sb.despine(left=True, bottom=True, ax=ax)
    plt.tight_layout()

    return ax
