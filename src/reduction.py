import torch
import numpy as np
import pandas as pd
from torch_geometric.utils import k_hop_subgraph

import random


def embedding_sim(model, x, edge_index, edge_label_index, num_layers):

    source_node, target_node = edge_label_index.numpy()[:, 0]

    # 1. Get the neighbors of source node and target node
    embeddings = model.encode(x=x, edge_index=edge_index)  # [2708, 64]

    neighbors_s, _, _, _ = k_hop_subgraph(int(source_node), num_hops=num_layers, edge_index=edge_index,
                                          relabel_nodes=True)  # num_nodes=num_nodes
    neighbors_t, _, _, _ = k_hop_subgraph(int(target_node), num_hops=num_layers, edge_index=edge_index,
                                          relabel_nodes=True)  # num_nodes=num_nodes
    neighbors_s = neighbors_s.numpy()
    neighbors_t = neighbors_t.numpy()

    # 2. Compute the similarity of the neighbor of the source node with the target node
    sim_dict_s = {}
    for n in neighbors_s:
        sim = torch.sigmoid(torch.dot(embeddings[target_node], embeddings[n])).item()
        sim_dict_s[n] = sim

    sim_dict_t = {}
    for n in neighbors_t:
        sim = torch.sigmoid(torch.dot(embeddings[source_node], embeddings[n])).item()
        sim_dict_t[n] = sim

    sim_dict_s = sorted(sim_dict_s.items(), key=lambda x: x[1], reverse=True)
    sim_dict_t = sorted(sim_dict_t.items(), key=lambda x: x[1], reverse=True)

    return sim_dict_s, sim_dict_t


def reduce_graph(model, x, edge_index, edge_label_index, num_layers, computation_graph_edge_index,
                 top_num_neighbors='half', random_nodes=False):

    # # 1. Get the neighbors of source node and target node
    # # 2. Compute the similarity of the neighbor of the source node with the target node

    sim_dict_s, sim_dict_t = embedding_sim(model, x, edge_index, edge_label_index, num_layers)
    # TODO: Decide the number of top neighbors to include
    # Option 1: merge two dicts and take the top nodes
    merged_dict = {}
    for element in sim_dict_s:
        merged_dict[element[0]] = element[1]

    for element in sim_dict_t:
        key = element[0]
        if key not in merged_dict:
            merged_dict[key] = element[1]
        else:
            # take the maximum of two similarities
            merged_dict[key] = max(merged_dict[key], element[1])

    merged_dict = sorted(merged_dict.items(), key=lambda x: x[1], reverse=True)

    # if top_num_neighbors == 'half':
    #     top_num_neighbors = int(len(merged_dict) / 2)
    if top_num_neighbors.endswith('%'):
        top_num_neighbors = int(len(merged_dict) * int(top_num_neighbors[:-1]) / 100)
    print('top_num_neighbors: ', top_num_neighbors)

    neighbors = sorted([x[0] for x in merged_dict][:top_num_neighbors])

    # # Option 2: take top nodes from both ends, this would result
    # # in different number of nodes for each link to be explained
    # if top_num_neighbors == 'half':
    #     top_num_neighbors = int(max(len(sim_dict_s), len(sim_dict_t)) / 2)
    # print('top_num_neighbors: ', top_num_neighbors)
    #
    # neighbors_keep_s = [x[0] for x in sim_dict_s][:top_num_neighbors]
    # neighbors_keep_t = [x[0] for x in sim_dict_t][:top_num_neighbors]
    # neighbors = sorted(list(set(neighbors_keep_s + neighbors_keep_t)))

    print('Random Nodes: ', random_nodes)
    # Option 3: Take random nodes as explanations.
    if random_nodes:
        random.seed(42)
        neighbors = [x[0] for x in merged_dict]
        if len(neighbors) > top_num_neighbors:
            neighbors = sorted(random.sample(neighbors, top_num_neighbors))
        else:
            neighbors = sorted(neighbors)

    print(f'Number of edges in the computation graph before reduction: {computation_graph_edge_index.shape}')

    # Reduce the edges into only those in the computation graph.
    reduced_computation_graph_edge_index = []
    edge_mask = []
    for edge in computation_graph_edge_index.T:
        if edge[0].item() in neighbors or edge[1].item() in neighbors:
            reduced_computation_graph_edge_index.append(edge.tolist())
            edge_mask.append(1)
        else:
            edge_mask.append(0)
    computation_graph_edge_index = torch.Tensor(reduced_computation_graph_edge_index).T

    return neighbors, computation_graph_edge_index, edge_mask

