import torch
import numpy as np
import pandas as pd

from scipy.stats import gaussian_kde, bernoulli
from torch_geometric.utils import to_dense_adj, to_undirected

# TODO: Option 3: the choice of the distribution of noise should follow somthing
#  also need to make sure the distortion is small, try fitting the distribution of (1 - important_edge_mask)


def bernoulli(total_edge_index, computation_edge_index, edge_mask, samples, random_seed, noise_type='whole'):

    # computation_graph_edge_index_undirected = to_undirected(edge_index)
    # adj_matrix = to_dense_adj(computation_graph_edge_index_undirected)[0]

    if noise_type == 'whole':
        edge_index = total_edge_index
    elif noise_type == 'computation':
        edge_index = computation_edge_index
    elif noise_type == 'half':
        edge_index = total_edge_index

    num_edge = edge_mask.shape[0]
    adj_matrix = to_dense_adj(edge_index)[0]
    num_nodes = adj_matrix.shape[0]

    # Calculating the probability of an edge between any two nodes as a bernoulli distribution
    adj_matrix = adj_matrix.reshape(-1)
    edge_probabilities = adj_matrix.sum() / (len(adj_matrix) - num_nodes)  # Exclude diagonal elements

    np.random.seed(random_seed)
    if noise_type == 'half':
        edge_probabilities = 1 / 2
    noise = np.random.binomial(n=1, p=edge_probabilities, size=(samples, num_edge))

    print('edge_probabilities: ', edge_probabilities)

    return noise


# Option 1: Use kde to approximate the distribution of important edge masks and then randomly sample
def kde(edge_mask, samples, random_seed, mask_type='mask'):

    num_edges = edge_mask.shape[0]

    if edge_mask.equal(torch.zeros(num_edges)):
        noise = torch.zeros(samples, num_edges)
    else:
        # TODO: 1 - mask or mask
        if mask_type == 'mask':
            kde = gaussian_kde([mask.item() for mask in edge_mask])
        elif mask_type == '1-mask':
            kde = gaussian_kde([1 - mask.item() for mask in edge_mask if mask > 0])
        noise = kde.resample(samples * num_edges, seed=random_seed)
        noise = np.clip(noise, 0, 1)
        noise = noise.reshape(samples, num_edges)

    return noise


def normal(edge_mask):

    # Option 2: Normal distribution with mean and std
    edge_mask_mean = torch.mean(edge_mask).item()
    edge_mask_std = torch.std(edge_mask).item()
    noise = torch.normal(mean=edge_mask_mean, std=edge_mask_std,
                         size=edge_mask.shape,
                         generator=rng, device=device)


def generate_edge_noise(total_edge_index, computation_edge_index, edge_mask, edge_noise_type, samples, random_seed):

    if edge_noise_type == 'kde':
        noise = kde(edge_mask, samples, random_seed)
    if edge_noise_type == 'kde_1-mask':
        noise = kde(edge_mask, samples, random_seed, '1-mask')
    elif edge_noise_type == 'bernoulli_whole':
        noise = bernoulli(total_edge_index, computation_edge_index, edge_mask, samples, random_seed, noise_type='whole')
    elif edge_noise_type == 'bernoulli_computation':
        noise = bernoulli(total_edge_index, computation_edge_index, edge_mask, samples, random_seed, noise_type='computation')
    elif edge_noise_type == 'bernoulli_1/2':
        noise = bernoulli(total_edge_index, computation_edge_index, edge_mask, samples, random_seed, noise_type='half')
    elif edge_noise_type == 'none':
        noise = torch.zeros(samples, edge_mask.shape[0])
    elif edge_noise_type == 'random' or edge_noise_type == 'all':
        # TODO: Not Implemented
        noise = torch.zeros(samples, edge_mask.shape[0])
    return noise
