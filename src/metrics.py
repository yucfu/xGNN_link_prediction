import torch
import numpy as np

import networkx as nx

from typing import Optional, Union, Any

from tqdm import tqdm

from utils import normalize_bounds

from scipy.stats import entropy
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.nn import APPNP
from torch_geometric.nn import MessagePassing

from edge_noise import generate_edge_noise

import time


def get_ws_tp_edges(graph, source_node, target_node):
    triangle_nodes = set(graph.neighbors(source_node)).intersection(graph.neighbors(target_node))

    triangle_edges = []
    for n in triangle_nodes:
        for m in (source_node, target_node):
            triangle_edges.append(tuple(sorted((n, m))))
            # triangle_edges.append((m, n))

    return triangle_edges


def get_ws_tn_edges(graph, source_node, target_node):
    tp_edges = get_ws_tp_edges(graph, source_node, target_node)
    n_graph = graph.copy()
    n_graph.remove_edges_from(tp_edges)

    return [tuple(sorted(e)) for e in n_graph.edges()]


def ws_confusion_matrix(
        edge_index,
        edge_mask,
        node_mask,
        graph,
        source_node,
        target_node,
        binary_threshold
):
    model_p_edges_idx = np.where(edge_mask > binary_threshold)[0]
    model_p_edges = edge_index[:, model_p_edges_idx].T.numpy()
    model_p_edges = [tuple(sorted(e)) for e in model_p_edges]

    model_n_edges = [tuple(sorted(e)) for e in graph.edges() if tuple(sorted(e)) not in model_p_edges]

    tp_edges = get_ws_tp_edges(graph, source_node, target_node)
    tn_edges = get_ws_tn_edges(graph, source_node, target_node)

    tp = []
    tn = []
    fp = []
    fn = []
    for e in model_p_edges:
        if e in tp_edges:
            tp.append(e)
        else:
            fp.append(e)

    for e in model_n_edges:
        if e in tn_edges:
            tn.append(e)
        else:
            fn.append(e)

    tp = list(set(tp))
    tn = list(set(tn))
    fp = list(set(fp))
    fn = list(set(fn))

    conf = np.array([
        [len(tp), len(fp)],
        [len(fn), len(tn)]]
    )

    return conf, tp, tn, fp, fn


def get_sbm_tp_edges(graph, source_node, target_node, node_labels):
    # node_labels is ground truth: which block each node belongs to
    source_block = node_labels[source_node]
    target_block = node_labels[target_node]
    if source_block != target_block:
        return []
    else:
        inter_block_edges = []
        # Find all edges that are in the same block as the source/target node
        for s, t in graph.edges:
            if node_labels[s] == node_labels[source_node] \
                    and node_labels[t] == node_labels[source_node]:
                inter_block_edges.append(tuple(sorted((s, t))))
        return inter_block_edges


def get_sbm_tn_edges(graph, source_node, target_node, node_labels):
    tp_edges = get_sbm_tp_edges(graph, source_node, target_node, node_labels)
    n_graph = graph.copy()
    n_graph.remove_edges_from(tp_edges)

    return [tuple(sorted(e)) for e in n_graph.edges()]


def sbm_confusion_matrix(
        node_labels,
        edge_index,
        edge_mask,
        node_mask,
        graph,
        source_node,
        target_node,
        binary_threshold
):
    # Important edges by the explanation
    model_p_edges_idx = np.where(edge_mask > binary_threshold)[0]
    model_p_edges = edge_index[:, model_p_edges_idx].T.numpy()
    model_p_edges = [tuple(sorted(e)) for e in model_p_edges]

    model_n_edges = [tuple(sorted(e)) for e in graph.edges() if tuple(sorted(e)) not in model_p_edges]

    # Get TP and TN edges
    tp_edges = get_sbm_tp_edges(graph, source_node, target_node, node_labels)
    tn_edges = get_sbm_tn_edges(graph, source_node, target_node, node_labels)

    tp = []
    tn = []
    fp = []
    fn = []
    for e in model_p_edges:
        if e in tp_edges:
            tp.append(e)
        else:
            fp.append(e)

    for e in model_n_edges:
        if e in tn_edges:
            tn.append(e)
        else:
            fn.append(e)

    tp = list(set(tp))
    tn = list(set(tn))
    fp = list(set(fp))
    fn = list(set(fn))

    conf = np.array([
        [len(tp), len(fp)],
        [len(fn), len(tn)]]
    )

    return conf, tp, tn, fp, fn


def sensitivity_specificity(confusion_matrix):  # TPR = TP/(TP+FN) TNR = TN/(TN+FP)

    tp = confusion_matrix[0, 0]
    tn = confusion_matrix[1, 1]
    fp = confusion_matrix[0, 1]
    fn = confusion_matrix[1, 0]

    if (tp + fn) == 0.:
        sensitivity = 1.
    else:
        sensitivity = tp / (tp + fn)

    if (tn + fp) == 0.:
        specificity = 1.
    else:
        specificity = tn / (tn + fp)

    return sensitivity, specificity


def get_tpr_tnr_curves(
        edge_index,
        edge_mask,
        node_mask,
        computation_graph,
        source_node,
        target_node,
        bt_range,
        graph='ws',
        node_block_labels=None
):
    tpr_tnr_curve = []
    for binary_threshold in bt_range:
        assert graph in ['ws', 'sbm']
        if graph == 'ws':
            conf, tp, tn, fp, fn = ws_confusion_matrix(
                edge_index,
                edge_mask,
                node_mask,
                computation_graph,
                source_node,
                target_node,
                binary_threshold
            )
        if graph == 'sbm':
            conf, tp, tn, fp, fn = sbm_confusion_matrix(
                node_block_labels,
                edge_index,
                edge_mask,
                node_mask,
                computation_graph,
                source_node,
                target_node,
                binary_threshold
            )
        tpr, tnr = sensitivity_specificity(conf)
        tpr_tnr_curve.append([tpr, tnr])
    return np.array(tpr_tnr_curve)


def deletion_curve_edges(
        model,
        x,
        edge_index,
        edge_label_index,
        edge_mask,
        sorting='descending',
        trials=20,
        sigmoid=True,
        seed=None
):
    if sorting == 'descending':
        # Descending sort of edge mask (importance).
        edge_mask_sort_idx = np.argsort(edge_mask)[::-1]
        # print(edge_mask_sort_idx)
        hard_mask = torch.ones(edge_mask.shape[0], dtype=torch.long)

        # prediction using all edges of the subgraph
        if sigmoid:
            deletion_curve = [model(x, edge_index, edge_label_index).sigmoid().item()]
        else:
            deletion_curve = [model(x, edge_index, edge_label_index).item()]

        # Gradually remove edges from the subgraph
        for i in tqdm(edge_mask_sort_idx):
            hard_mask[i] = 0
            sub_edge_index = (edge_index * hard_mask).clone()
            if sigmoid:
                deletion_curve.append(model(x, sub_edge_index, edge_label_index).sigmoid().item())
            else:
                deletion_curve.append(model(x, sub_edge_index, edge_label_index).item())

    if sorting == 'random':
        if seed is not None:
            np.random.seed(seed)

        deletion_curve = []
        for _ in tqdm(range(trials)):
            edge_mask_sort_idx = list(range(edge_mask.shape[0]))
            np.random.shuffle(edge_mask_sort_idx)
            hard_mask = torch.ones(edge_mask.shape[0], dtype=torch.long)
            if sigmoid:
                single_deletion_curve = [model(x, edge_index, edge_label_index).sigmoid().item()]
            else:
                single_deletion_curve = [model(x, edge_index, edge_label_index).item()]
            for i in edge_mask_sort_idx:
                hard_mask[i] = 0
                sub_edge_index = (edge_index * hard_mask).clone()
                if sigmoid:
                    single_deletion_curve.append(model(x, sub_edge_index, edge_label_index).sigmoid().item())
                else:
                    single_deletion_curve.append(model(x, sub_edge_index, edge_label_index).item())
            deletion_curve.append(single_deletion_curve)

    return np.array(deletion_curve)


def deletion_curve_features(
        model,
        x,
        edge_index,
        edge_label_index,
        feature_mask,
        feature_base_values,
        sorting='descending',
        trials=20,
        sigmoid=True,
        seed=None
):
    if sorting == 'descending':
        feature_mask_sort_idx = np.argsort(feature_mask)[::-1]
        hard_mask = x.clone()
        if sigmoid:
            deletion_curve = [model(x, edge_index, edge_label_index).sigmoid().item()]
        else:
            deletion_curve = [model(x, edge_index, edge_label_index).item()]
        for i in tqdm(feature_mask_sort_idx):
            hard_mask[:, i] = feature_base_values[i]
            if sigmoid:
                deletion_curve.append(model(hard_mask, edge_index, edge_label_index).sigmoid().item())
            else:
                deletion_curve.append(model(hard_mask, edge_index, edge_label_index).item())

    if sorting == 'random':
        if seed is not None:
            np.random.seed(seed)

        deletion_curve = []
        for _ in tqdm(range(trials)):
            feature_mask_sort_idx = list(range(feature_mask.shape[0]))
            np.random.shuffle(feature_mask_sort_idx)
            hard_mask = x.clone()
            if sigmoid:
                single_deletion_curve = [model(x, edge_index, edge_label_index).sigmoid().item()]
            else:
                single_deletion_curve = [model(x, edge_index, edge_label_index).item()]
            for i in feature_mask_sort_idx:
                hard_mask[:, i] = feature_base_values[i]
                if sigmoid:
                    single_deletion_curve.append(model(hard_mask, edge_index, edge_label_index).sigmoid().item())
                else:
                    single_deletion_curve.append(model(hard_mask, edge_index, edge_label_index).item())
            deletion_curve.append(single_deletion_curve)

    return np.array(deletion_curve)


def random_line(n, c0, cf, x):
    return c0 + (cf - c0) / x * n


def compute_upper_area(c0, cf):
    return 0.5 * (2 - c0 - cf)


def linear_area_score(deletion_curve, normalize=False):
    if normalize:
        deletion_curve = normalize_bounds(deletion_curve)
    c0 = deletion_curve[0]
    cf = deletion_curve[-1]
    norm = deletion_curve.shape[0]
    random_baseline = np.array([random_line(n, c0, cf, norm) for n in range(norm)])
    upper_area = compute_upper_area(c0, cf)
    lower_area = 1. - upper_area
    Ap = np.maximum(deletion_curve - random_baseline, 0).sum() / norm
    Am = np.maximum(random_baseline - deletion_curve, 0).sum() / norm

    return Am / lower_area - Ap / upper_area


def area_score(deletion_curve, random_baseline, norm=True):
    assert deletion_curve.shape == random_baseline.shape
    if norm:
        deletion_curve = normalize_bounds(deletion_curve)
        random_baseline = normalize_bounds(random_baseline)
    # score = 0
    Ap = 0
    Am = 0
    l = 0
    u = 0
    for i in range(deletion_curve.shape[0]):
        Ap += np.max([deletion_curve[i] - random_baseline[i], 0])
        Am += np.max([random_baseline[i] - deletion_curve[i], 0])
        l += random_baseline[i] - np.min([np.min(deletion_curve), np.min(random_baseline)])
        u += 1. - random_baseline[i]
        # score += random_baseline[i]-deletion_curve[i]
    # return score/(deletion_curve.shape[0]-2)
    print(f"Am {Am:.2f}, Ap {Ap:.2f}, l {l:.2f}, u {u:.2f}, Am/l {Am / l:.2f}, Ap/u {Ap / u:.2f}")
    return Am / l - Ap / u


def fidelity_sparsity(deletion_curve, normalize=False):
    if normalize:
        deletion_curve = normalize_bounds(deletion_curve)
    c0 = deletion_curve[0]
    fidelity_at_sparsity = []
    for s in range(deletion_curve.shape[0]):
        fidelity_at_sparsity.append((c0 - deletion_curve[s]).sum())
    return np.array(fidelity_at_sparsity)


def subgraph_lp(model, source_node, target_node, x, edge_index):
    num_nodes, num_edges = x.size(0), edge_index.size(1)

    flow = 'source_to_target'
    for module in model.modules():
        if isinstance(module, MessagePassing):
            flow = module.flow
            break

    num_hops = 0
    for module in model.modules():
        if isinstance(module, MessagePassing):
            if isinstance(module, APPNP):
                num_hops += module.K
            else:
                num_hops += 1

    print('num_hops: ', num_hops)
    # subset, edge_index, mapping, edge_mask = k_hop_subgraph(
    #     node_idx, num_hops, edge_index, relabel_nodes=True,
    #     num_nodes=num_nodes, flow=flow)

    subset, sub_edge_index, sub_mapping, sub_edge_mask = k_hop_subgraph(
        [source_node, target_node],
        num_hops,
        edge_index, relabel_nodes=True, num_nodes=num_nodes, flow=flow)

    x = x[subset]
    # for key, item in kwargs:
    #     if torch.is_tensor(item) and item.size(0) == num_nodes:
    #         item = item[subset]
    #     elif torch.is_tensor(item) and item.size(0) == num_edges:
    #         item = item[edge_mask]
    #     kwargs[key] = item

    return x, sub_edge_index, sub_mapping, sub_edge_mask


def edge_mask_to_node_mask(data, edge_mask, aggregation="mean"):
    node_weights = torch.zeros(data.x.shape[0])

    if aggregation == "sum":

        for weight, nodes in zip(edge_mask, data.edge_index.T):  # [8446], [8446, 2]
            node_weights[nodes[0].item()] += weight.item() / 2
            node_weights[nodes[1].item()] += weight.item() / 2

    elif aggregation == "mean":

        node_degrees = torch.zeros(data.x.shape[0])
        for weight, nodes in zip(edge_mask, data.edge_index.T):
            node_weights[nodes[0].item()] += weight.item()
            node_weights[nodes[1].item()] += weight.item()
            node_degrees[nodes[0].item()] += 1
            node_degrees[nodes[1].item()] += 1
        node_weights = node_weights / node_degrees.clamp(min=1.)

    elif aggregation == "max":

        for weight, nodes in zip(edge_mask, data.edge_index.T):
            node_weights[nodes[0].item()] = max(weight.item(), node_weights[nodes[0].item()])
            node_weights[nodes[1].item()] = max(weight.item(), node_weights[nodes[1].item()])

    else:
        raise NotImplementedError(f"No such aggregation method: {aggregation}")

    return node_weights


def sub_edge_mask_to_sub_node_mask(sub_num_nodes, sub_edge_index, sub_edge_mask, aggregation="mean"):
    # shape of edge_index: [2, 96]
    node_weights = torch.zeros(sub_num_nodes)  # Number of nodes
    # node_weights = torch.zeros(data.x.shape[0])

    if aggregation == "sum":
        for weight, nodes in zip(sub_edge_mask, sub_edge_index.T):  # [8446], [8446, 2]
            node_weights[nodes[0].item()] += weight.item() / 2
            node_weights[nodes[1].item()] += weight.item() / 2

    elif aggregation == "mean":
        node_degrees = torch.zeros(sub_num_nodes)
        for weight, nodes in zip(sub_edge_mask, sub_edge_index.T):
            node_weights[nodes[0].item()] += weight.item()
            node_weights[nodes[1].item()] += weight.item()
            node_degrees[nodes[0].item()] += 1
            node_degrees[nodes[1].item()] += 1
        node_weights = node_weights / node_degrees.clamp(min=1.)

    elif aggregation == "max":
        for weight, nodes in zip(sub_edge_mask, sub_edge_index.T):
            node_weights[nodes[0].item()] = max(weight.item(), node_weights[nodes[0].item()])
            node_weights[nodes[1].item()] = max(weight.item(), node_weights[nodes[1].item()])

    else:
        raise NotImplementedError(f"No such aggregation method: {aggregation}")

    return node_weights


def convert_logit_to_label(log_logits, sigmoid):
    if sigmoid:
        if int(log_logits.sigmoid().item() > 0.5):
            predicted_label = 1
        else:
            predicted_label = 0
    else:
        if int(log_logits.item() > 0.5):
            predicted_label = 1
        else:
            predicted_label = 0

    return predicted_label


# The function to return the k-hop subgraph of the selected nodes
def fidelity(model,  # is a must
             source_node,  # is a must
             target_node,  # is a must
             full_feature_matrix,  # must
             edge_index=None,  # the whole, so data.edge_index
             node_mask=None,  # at least one of these three node, feature, edge
             feature_mask=None,
             edge_mask=None,
             samples=100,
             random_seed=12345,
             device="cpu",
             validity=False,
             edge_noise_type='kde'):
    """
    Distortion/Fidelity (for Node Classification), modified for link prediction
    :param model: GNN model which is explained
    :param node_idx: The node which is explained
    :param full_feature_matrix: The feature matrix from the Graph (X)
    :param edge_index: All edges
    :param node_mask: Is a (binary) tensor with 1/0 for each node in the computational graph
    => 1 means the features of this node will be fixed
    => 0 means the features of this node will be pertubed/randomized
    if not available torch.ones((1, num_computation_graph_nodes))
    :param feature_mask: Is a (binary) tensor with 1/0 for each feature
    => 1 means this features is fixed for all nodes with 1
    => 0 means this feature is randomized for all nodes
    if not available torch.ones((1, number_of_features))
    :param edge_mask:
    :param samples:
    :param random_seed:
    :param device:
    :param validity:
    :return:
    """
    if edge_mask is None and feature_mask is None and node_mask is None:
        raise ValueError("At least supply one mask")

    # TODO: check whether to include kwargs as output
    computation_graph_feature_matrix, computation_graph_edge_index, mapping, hard_edge_mask = \
        subgraph_lp(model, source_node, target_node, full_feature_matrix, edge_index)

    # edge_label_index is the position of both nodes of the edge in the computation graph, which is the same as the
    # mapping (the mapping from node indices in node_idx to their new location)
    # TODO: check if this is correct
    edge_label_index = mapping.unsqueeze(dim=1)

    # get predicted label
    log_logits = model(x=computation_graph_feature_matrix,
                       edge_index=computation_graph_edge_index,
                       edge_label_index=edge_label_index)

    # TODO: check the sigmoid parameter
    # original prediction using the whole computation graph
    predicted_label = convert_logit_to_label(log_logits, sigmoid=True)

    # fill missing masks
    if feature_mask is None:
        (num_nodes, num_features) = full_feature_matrix.size()
        feature_mask = torch.ones((1, num_features), device=device)

    num_computation_graph_nodes = computation_graph_feature_matrix.size(0)
    if node_mask is None:  # all nodes selected
        node_mask = torch.ones((1, num_computation_graph_nodes), device=device)

    (num_nodes, num_features) = full_feature_matrix.size()
    num_nodes_computation_graph = computation_graph_feature_matrix.size(0)

    # retrieve complete mask as matrix
    node_mask = torch.Tensor(node_mask)
    feature_mask = torch.Tensor(feature_mask)
    mask = node_mask.T.matmul(feature_mask)  # [1433]

    if validity:
        samples = 1
        full_feature_matrix = torch.zeros_like(full_feature_matrix)

    correct = 0.0

    # Generate the random indices to select features used for generating the perturbed input.
    rng = torch.Generator(device=device)
    rng.manual_seed(random_seed)
    # num_nodes is the high range
    random_indices = torch.randint(num_nodes, (samples, num_nodes_computation_graph, num_features),
                                   generator=rng,
                                   device=device,
                                   )
    random_indices = random_indices.type(torch.int64)
    # samples, num_nodes_computation_graph, num_features
    # [100, 9, 1433]

    # TODO: should sample 100 noises here. Make sure the perturbed edge mask is different every time.
    #  Perturb the edge mask here.
    #  Key problem: How to add noise?
    #  S_A + (1 - S_A) * Bernoulli
    #  perturbed_edge_importance = edge_mask + (1 - edge_mask) * noise
    #  S_A: edge mask output by the explainer
    #  V_sa: Bernoulli distribution
    #  (1 - S_A): selecting the non-important edges
    #  Then apply Bernoulli
    #  1. 1-important_edge_mask: (but noise should be from the data, and independent of the explainer)
    #  2. bernoulli / Gumbel-Softmax distribution
    #  (Bernoulli distribution learned over the adjacency matrix of the complete/computation graph)
    #  - learn the bernoulli distribution from the adjacency matrix, then ?
    #  - learn the Gumbel-Softmax distribution from the adjacency matrix, then ?

    # # TODO: Steps to generate noise for edge_mask (Part 1)
    # if edge_mask is not None:
    #     # Distort the edge masks in this case:
    #     # keep the important edges and add random noise to the non-important edges.
    #     edge_mask = torch.Tensor(edge_mask)
    #
    #     # Generate 100 of V_sa with each shape being [96], final shape should be [100, 96]
    #     noise = generate_edge_noise(total_edge_index=edge_index, computation_edge_index=computation_graph_edge_index,
    #                                 edge_mask=edge_mask,
    #                                 edge_noise_type=edge_noise_type, samples=samples, random_seed=random_seed)
    #     print('Noise of the first sample: ', noise[0])

    change_in_pred_prob = []

    for i in range(samples):
        # TODO: 1. Perturb the features: keep important features unchanged, perturb the non-important features
        # 1. generate the perturbed input
        # get the features according to the random_indices from the full feature matrix
        random_features = torch.gather(full_feature_matrix,
                                       dim=0,
                                       index=random_indices[i])
        # TODO: 测试选中全部feature
        # mask = torch.ones(mask.shape)
        perturbed_input = computation_graph_feature_matrix * mask + random_features * (torch.ones(mask.shape) - mask)

        # TODO: Steps to generate noise for edge_mask (Part 2)
        # if edge_mask is not None:
        #     # If edge_mask is provided, we perturb the edges to get edge-perturbed RDT-Fidelity.
        #     # print(f'sample {i}')
        #     # print('edge_mask: ', edge_mask)
        #     # print('perturbed_edge_importance: ', perturbed_edge_importance)
        #     random_edge_noise = noise[i]
        #     perturbed_edge_importance = edge_mask + (1 - edge_mask) * random_edge_noise
        #     perturbed_edge_importance = perturbed_edge_importance.float()
        #     # TODO: 测试选中全部edge
        #     # perturbed_edge_importance = torch.ones(edge_mask.shape)
        #
        #     # RDT-Fidelity with edge mask as edge weight, and perturbed
        #     log_logits = model.forward(x=perturbed_input,  # [31, 1433]
        #                                edge_index=computation_graph_edge_index,  # [2, 96]
        #                                edge_label_index=edge_label_index,  # [2, 1]
        #                                edge_weight=perturbed_edge_importance)  # [96]
        # else:
        #     # Otherwise, just use original RDT-Fidelity.
        #     # 2. get the prediction from the trained model using the perturbed features as input
        #     log_logits = model.forward(x=perturbed_input,
        #                                edge_index=computation_graph_edge_index,
        #                                edge_label_index=edge_label_index)

        log_logits_perturbed = model.forward(x=perturbed_input,
                                             edge_index=computation_graph_edge_index,
                                             edge_label_index=edge_label_index)
        perturbed_predicted_label = convert_logit_to_label(log_logits_perturbed, sigmoid=True)

        change_in_pred_prob.append(log_logits.sigmoid().item() - log_logits_perturbed.sigmoid().item())

        # 3. calculate the number of correct predicted labels:
        # the idea is to compare the predicted label of the perturbed input
        # with the predicted label of the original full input (full computation graph)

        # mapping from pyg subgraph function:
        # the mapping from node indices in :obj:`node_idx` to their new location in the subgraph.
        # check whether the prediction for the node to explain is the same
        # TODO: check if mapping is necessary
        # correct += (perturbed_predicted_labels[mapping] == predicted_label).sum().item()
        # correct += (perturbed_predicted_label == predicted_label).sum().item()
        if perturbed_predicted_label == predicted_label:
            correct += 1

        # TODO: 2. Perturb the edges: keep the non-zero edge masks unchanged, perturb the non-important edge masks
        # (assigning importance to other edges)

    return correct / samples, np.mean(change_in_pred_prob), predicted_label


# def fidelity_reduced_graph_complete_explanation(model,  # is a must
#                                                 # node_idx,  # is a must
#                                                 source_node,  # is a must
#                                                 target_node,  # is a must
#                                                 full_feature_matrix,  # must
#                                                 edge_index=None,  # the whole, so data.edge_index
#                                                 node_mask=None,  # at least one of these three node, feature, edge
#                                                 feature_mask=None,
#                                                 edge_mask=None,
#                                                 samples=100,
#                                                 random_seed=12345,
#                                                 device="cpu",
#                                                 validity=False,
#                                                 edge_noise_type='kde'):
#     if edge_mask is None and feature_mask is None and node_mask is None:
#         raise ValueError("At least supply one mask")
#
#     # TODO: check whether to include kwargs as output
#     computation_graph_feature_matrix, computation_graph_edge_index, mapping, hard_edge_mask = \
#         subgraph_lp(model, source_node, target_node, full_feature_matrix, edge_index)
#
#     # edge_label_index is the position of both nodes of the edge in the computation graph, which is the same as the
#     # mapping (the mapping from node indices in node_idx to their new location)
#     # TODO: check if this is correct
#     edge_label_index = mapping.unsqueeze(dim=1)
#
#     # get predicted label
#     log_logits = model(x=computation_graph_feature_matrix,
#                        edge_index=computation_graph_edge_index,
#                        edge_label_index=edge_label_index)
#
#     # TODO: check the sigmoid parameter
#     predicted_label = convert_logit_to_label(log_logits, sigmoid=True)
#
#     # fill missing masks
#     if feature_mask is None:
#         (num_nodes, num_features) = full_feature_matrix.size()
#         feature_mask = torch.ones((1, num_features), device=device)
#
#     num_computation_graph_nodes = computation_graph_feature_matrix.size(0)
#     if node_mask is None:  # all nodes selected
#         node_mask = torch.ones((1, num_computation_graph_nodes), device=device)
#
#     (num_nodes, num_features) = full_feature_matrix.size()
#     num_nodes_computation_graph = computation_graph_feature_matrix.size(0)
#
#     # retrieve complete mask as matrix
#     node_mask = torch.Tensor(node_mask)
#     feature_mask = torch.Tensor(feature_mask)
#     mask = node_mask.T.matmul(feature_mask)  # [1433]
#
#     if validity:
#         samples = 1
#         full_feature_matrix = torch.zeros_like(full_feature_matrix)
#
#     correct = 0.0
#
#     # Generate the random indices to select features used for generating the perturbed input.
#     rng = torch.Generator(device=device)
#     rng.manual_seed(random_seed)
#     # num_nodes is the high range
#     random_indices = torch.randint(num_nodes, (samples, num_nodes_computation_graph, num_features),
#                                    generator=rng,
#                                    device=device,
#                                    )
#     random_indices = random_indices.type(torch.int64)
#     # samples, num_nodes_computation_graph, num_features
#     # [100, 9, 1433]
#
#     for i in range(samples):
#         # TODO: 1. Perturb the features: keep important features unchanged, perturb the non-important features
#         # 1. generate the perturbed input
#         # get the features according to the random_indices from the full feature matrix
#         random_features = torch.gather(full_feature_matrix,
#                                        dim=0,
#                                        index=random_indices[i])
#         perturbed_input = computation_graph_feature_matrix * mask + random_features * (
#                 torch.ones(mask.shape) - mask)
#         log_logits = model.forward(x=perturbed_input,
#                                    edge_index=computation_graph_edge_index,
#                                    edge_label_index=edge_label_index)
#
#         perturbed_predicted_label = convert_logit_to_label(log_logits, sigmoid=True)
#         if perturbed_predicted_label == predicted_label:
#             correct += 1
#
#     return correct / samples, predicted_label


def sparsity(mask):
    mask = mask.flatten()
    normalized_mask = mask / torch.sum(mask)
    return entropy(normalized_mask)


# Reference code: https://github.com/AslanDing/Fidelity/blob/main/tools/fidelity.py
def fidelity_original(model,  # is a must
                      source_node,  # is a must
                      target_node,  # is a must
                      target_label,  # the GT value of the prediction
                      full_feature_matrix,  # must
                      edge_index=None,  # the whole, so data.edge_index
                      node_mask=None,  # at least one of these three node, feature, edge
                      feature_mask=None,
                      edge_mask=None,
                      device="cpu"):
    # Calculate the Fidelity+ score: Prediction change by removing important nodes/edges/features.

    if edge_mask is None and feature_mask is None and node_mask is None:
        raise ValueError("At least supply one mask")

    computation_graph_feature_matrix, computation_graph_edge_index, mapping, hard_edge_mask = \
        subgraph_lp(model, source_node, target_node, full_feature_matrix, edge_index)

    edge_label_index = mapping.unsqueeze(dim=1)

    # TODO: 1. Compute the original prediction.
    log_logits = model(x=computation_graph_feature_matrix,
                       edge_index=computation_graph_edge_index,
                       edge_label_index=edge_label_index)
    log_logits_prob = log_logits.sigmoid().item()

    # original prediction using the whole computation graph
    predicted_label = convert_logit_to_label(log_logits, sigmoid=True)

    # fill missing masks
    if feature_mask is None:
        (num_nodes, num_features) = full_feature_matrix.size()
        feature_mask = torch.ones((1, num_features), device=device)

    num_computation_graph_nodes = computation_graph_feature_matrix.size(0)
    if node_mask is None:  # all nodes selected
        node_mask = torch.ones((1, num_computation_graph_nodes), device=device)

    node_mask = torch.Tensor(node_mask)
    feature_mask = torch.Tensor(feature_mask)

    # TODO: 2. Get the prediction from the new graph.
    # 2.1 Calculate Fidelity+ score
    # Prediction change by removing important nodes/edges/features.
    mask_plus = (1 - node_mask).T.matmul(feature_mask)
    masked_input_plus = computation_graph_feature_matrix * mask_plus
    log_logits_new_plus = model.forward(x=masked_input_plus,
                                        edge_index=computation_graph_edge_index,
                                        edge_label_index=edge_label_index)
    predicted_label_new_plus = convert_logit_to_label(log_logits_new_plus, sigmoid=True)
    log_logits_prob_new_plus = log_logits_new_plus.sigmoid().item()

    fid_label_plus = int(predicted_label == target_label) - int(predicted_label_new_plus == target_label)
    fid_prob_plus = log_logits_prob - log_logits_prob_new_plus


    # 2.2 Calculate Fidelity- score
    # prediction change by keeping important input features and removing unimportant features.
    mask_minus = node_mask.T.matmul(feature_mask)
    masked_input_minus = computation_graph_feature_matrix * mask_minus
    log_logits_new_minus = model.forward(x=masked_input_minus,
                                         edge_index=computation_graph_edge_index,
                                         edge_label_index=edge_label_index)
    predicted_label_new_minus = convert_logit_to_label(log_logits_new_minus, sigmoid=True)
    log_logits_prob_new_minus = log_logits_new_minus.sigmoid().item()

    fid_label_minus = int(predicted_label == target_label) - int(predicted_label_new_minus == target_label)
    fid_prob_minus = log_logits_prob - log_logits_prob_new_minus

    return fid_label_plus, fid_prob_plus, fid_label_minus, fid_prob_minus
