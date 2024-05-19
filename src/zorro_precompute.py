import torch
from tqdm import tqdm
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import APPNP
from torch_geometric.utils import k_hop_subgraph
import numpy as np
import logging
import time
from zorro import Zorro, fidelity_zorro, convert_logit_to_label


def save_precompute_full_distortion(model,
                                    edge_label_indexs,
                                    full_feature_matrix,
                                    full_edge_index,
                                    save_path,
                                    samples=100,
                                    random_seed=12345,
                                    device="cpu",
                                    ):
    # get basic attributes: num_hops, flow
    num_hops = Zorro.num_hops(model)
    flow = Zorro.flow(model)

    feature_distortion = []
    node_distortion = []

    (num_nodes, num_features) = full_feature_matrix.size()

    feature_distortion = torch.zeros((num_features, len(edge_label_indexs)), device=device)

    for index in range(len(edge_label_indexs)):

        feature_distortion_per_edge = []
        node_distortion_per_edge = []


        edge_label_index = edge_label_indexs[index]
        source_node, target_node = edge_label_index.numpy()[:, 0]

        subset, computation_graph_edge_index, mapping, edge_mask = k_hop_subgraph(
            [source_node, target_node],
            num_hops,
            full_edge_index, relabel_nodes=True, num_nodes=num_nodes, flow=flow)

        new_edge_label_index = mapping.unsqueeze(dim=1)
        computation_graph_feature_matrix = full_feature_matrix[subset]
        num_nodes_computation_graph = computation_graph_feature_matrix.size(0)

        # calculate predicted labels
        log_logits = model(x=computation_graph_feature_matrix,
                           edge_index=computation_graph_edge_index,
                           edge_label_index=new_edge_label_index)
        predicted_labels = convert_logit_to_label(log_logits, sigmoid=True)

        # calculate initial distortion
        node_mask = torch.zeros((1, num_nodes_computation_graph), device=device)
        feature_mask = torch.zeros((1, num_features), device=device)
        initial_distortion = fidelity_zorro(model,
                                            edge_label_index=mapping,
                                            full_feature_matrix=full_feature_matrix,
                                            computation_graph_feature_matrix=computation_graph_feature_matrix,
                                            edge_index=computation_graph_edge_index,
                                            node_mask=node_mask,
                                            feature_mask=feature_mask,
                                            predicted_label=predicted_labels,
                                            samples=samples,
                                            random_seed=random_seed,
                                            device=device)

        # calculate the improvement of features

        node_mask = torch.ones_like(node_mask, device=device)

        for i in tqdm(range(num_features)):
            # Include this feature
            feature_mask[0, i] += 1

            fidelity = fidelity_zorro(model,
                                      edge_label_index=mapping,
                                      full_feature_matrix=full_feature_matrix,
                                      computation_graph_feature_matrix=computation_graph_feature_matrix,
                                      edge_index=computation_graph_edge_index,
                                      node_mask=node_mask,
                                      feature_mask=feature_mask,
                                      predicted_label=predicted_labels,
                                      samples=samples,
                                      random_seed=random_seed,
                                      device=device)
            feature_distortion_per_edge.append(fidelity)

            feature_mask[0, i] -= 1

        # calculate the improvement of nodes
        node_distortion = torch.zeros((num_nodes_computation_graph, len(edge_label_indexs)), device=device)
        feature_mask = torch.ones_like(feature_mask, device=device)
        node_mask = torch.zeros_like(node_mask, device=device)

        for i in tqdm(range(num_nodes_computation_graph)):
            node_mask[0, i] += 1

            node_distortion[i] = fidelity_zorro(model,
                                                edge_label_index=mapping,
                                                full_feature_matrix=full_feature_matrix,
                                                computation_graph_feature_matrix=computation_graph_feature_matrix,
                                                edge_index=computation_graph_edge_index,
                                                node_mask=node_mask,
                                                feature_mask=feature_mask,
                                                predicted_label=predicted_labels,
                                                samples=samples,
                                                random_seed=random_seed,
                                                device=device)
            node_mask[0, i] -= 1

    np.savez_compressed(save_path,
                        **{
                            "nodes": nodes,
                            "subset": subset.cpu().numpy(),
                            "mapping": mapping.cpu().numpy(),
                            "initial_distortion": initial_distortion.cpu().numpy(),
                            "feature_distortion": feature_distortion.cpu().numpy(),
                            "node_distortion": node_distortion.cpu().numpy(),
                        }
                        )

    return subset, mapping, initial_distortion, feature_distortion, node_distortion


def main():
    edge_label_index
