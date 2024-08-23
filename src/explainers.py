import torch_geometric.transforms as T
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig, ExplainerConfig, Explanation
from torch_geometric.nn import to_captum_input, to_captum_model
from captum.attr import IntegratedGradients, Deconvolution, GuidedBackprop, LRP
# from torch_geometric.utils import k_hop_subgraph

import torch
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.nn import APPNP

# from metrics import subgraph_lp

import pgmexplainer as pe
from zorro import SoftZorro, ZorroBaseline, Zorro
from reduction import reduce_graph
from graphlime import GraphLIME


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


def gnnexplainer(
        model,
        x,
        edge_index,
        edge_label_index,
        return_type='raw',
        target=None):
    model_config = ModelConfig(
        mode='binary_classification',
        task_level='edge',
        return_type=return_type,
    )

    if not target:
        explainer = Explainer(
            model=model,
            explanation_type='model',
            algorithm=GNNExplainer(epochs=200),
            node_mask_type='common_attributes',
            edge_mask_type='object',
            model_config=model_config,
        )
        explanation = explainer(
            x=x,
            edge_index=edge_index,
            edge_label_index=edge_label_index
        )
    else:
        explainer = Explainer(
            model=model,
            explanation_type='phenomenon',
            algorithm=GNNExplainer(epochs=200),
            node_mask_type='common_attributes',
            edge_mask_type='object',
            model_config=model_config,
        )
        explanation = explainer(
            x=x,
            edge_index=edge_index,
            target=target,
            edge_label_index=edge_label_index,
        )

    return explanation


def ig(
        model,
        x,
        edge_index,
        edge_label_index,
        return_type='raw',
        target=None):
    args, additional_args = to_captum_input(
        x,
        edge_index,
        'node_and_edge',
        edge_label_index)
    # function to_captum_input:
    # x: node features
    # edge_index: edge indices
    # mask_type: "edge", "node", and "node_and_edge"
    # Returns inputs and additional_forward_args required for Captum’s attribute functions

    captum_model = to_captum_model(model, mask_type='node_and_edge',
                                   output_idx=0)
    # output_idx: index of the output node for which attributions are computed
    # Explain predictions for node output_idx

    ig = IntegratedGradients(captum_model)

    ig_attr_node, ig_attr_edge = ig.attribute(
        args,
        additional_forward_args=additional_args, internal_batch_size=1)
    # print(ig_attr_node.shape)
    # torch.Size([1, 2708, 1433])

    # print(ig_attr_edge.shape)
    # torch.Size([1, 8448])

    # target: Output indices for which gradients are computed (for classification cases,
    # this is usually the target class). Default it None.

    # subset, sub_edge_index, mapping, sub_edge_mask = k_hop_subgraph(
    #     [source_node, target_node], 2, edge_index, directed=False
    # )
    explanation = Explanation(
        node_mask=ig_attr_node[0, :, :].detach(),  # [2708, 1433]
        edge_mask=ig_attr_edge[0, :].detach(),  # [8448]
        x=x,
        edge_index=edge_index,
        edge_label_index=edge_label_index
    )

    return explanation


def deconvolution(
        model,
        x,
        edge_index,
        edge_label_index,
        return_type='raw',
        target=None):
    args, additional_args = to_captum_input(
        x,
        edge_index,
        'node_and_edge',
        edge_label_index)

    captum_model = to_captum_model(model, mask_type='node_and_edge',
                                   output_idx=0)

    ig = Deconvolution(captum_model)

    ig_attr_node, ig_attr_edge = ig.attribute(
        args,
        additional_forward_args=additional_args)

    # subset, sub_edge_index, mapping, sub_edge_mask = k_hop_subgraph(
    #     [source_node, target_node], 2, edge_index, directed=False
    # )
    explanation = Explanation(
        node_mask=ig_attr_node[0, :, :].detach(),
        edge_mask=ig_attr_edge[0, :].detach(),
        x=x,
        edge_index=edge_index,
        edge_label_index=edge_label_index
    )

    return explanation


def backprop(
        model,
        x,
        edge_index,
        edge_label_index,
        return_type='raw',
        target=None):
    args, additional_args = to_captum_input(
        x,
        edge_index,
        'node_and_edge',
        edge_label_index)

    captum_model = to_captum_model(model, mask_type='node_and_edge',
                                   output_idx=0)

    ig = GuidedBackprop(captum_model)

    ig_attr_node, ig_attr_edge = ig.attribute(
        args,
        additional_forward_args=additional_args)

    # subset, sub_edge_index, mapping, sub_edge_mask = k_hop_subgraph(
    #     [source_node, target_node], 2, edge_index, directed=False
    # )
    explanation = Explanation(
        node_mask=ig_attr_node[0, :, :].detach(),
        edge_mask=ig_attr_edge[0, :].detach(),
        x=x,
        edge_index=edge_index,
        edge_label_index=edge_label_index
    )

    return explanation


def lrp(
        model,
        x,
        edge_index,
        edge_label_index,
        target=None):
    args, additional_args = to_captum_input(
        x,
        edge_index,
        'node_and_edge',
        edge_label_index)

    captum_model = to_captum_model(model, mask_type='node_and_edge',
                                   output_idx=0)

    ig = LRP(captum_model)

    ig_attr_node, ig_attr_edge = ig.attribute(
        args,
        additional_forward_args=additional_args)

    # subset, sub_edge_index, mapping, sub_edge_mask = k_hop_subgraph(
    #     [source_node, target_node], 2, edge_index, directed=False
    # )
    explanation = Explanation(
        node_mask=ig_attr_node[0, :, :].detach(),
        edge_mask=ig_attr_edge[0, :].detach(),
        x=x,
        edge_index=edge_index,
        edge_label_index=edge_label_index
    )

    return explanation


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

    subset, sub_edge_index, sub_mapping, sub_edge_mask = k_hop_subgraph(
        [source_node, target_node],
        num_hops,
        edge_index, relabel_nodes=True, num_nodes=num_nodes, flow=flow)

    x = x[subset]

    return x, sub_edge_index, sub_mapping, sub_edge_mask


def execute_model_with_gradient(model, x, edge_index, edge_label_index):
    ypred = model(x, edge_index, edge_label_index)
    # print(ypred)  # tensor([0.9042], grad_fn=<ViewBackward0>)
    # print(ypred.shape)  # torch.Size([1])
    predicted_label = convert_logit_to_label(ypred, sigmoid=True)
    loss = -torch.log(ypred)
    loss.backward()

    # ypred = model(x, edge_index, edge_label_index)
    # # predicted_labels = ypred.argmax(dim=-1)
    # predicted_labels = convert_logit_to_label(ypred, sigmoid=True)
    # predicted_label = predicted_labels[node]
    # logit = torch.nn.functional.softmax((ypred[node, :]).squeeze(), dim=0)
    # logit = logit[predicted_label]
    # loss = -torch.log(logit)
    # loss.backward()


# def grad_edge_explanation(model, node, x, edge_index, edge_label_index):
#     model.zero_grad()
#
#     E = edge_index.size(1)
#     edge_mask = torch.nn.Parameter(torch.ones(E))
#
#     for module in model.modules():
#         if isinstance(module, MessagePassing):
#             module.__explain__ = True
#             module.__edge_mask__ = edge_mask
#
#     edge_mask.requires_grad = True
#     x.requires_grad = True
#
#     if edge_mask.grad is not None:
#         edge_mask.grad.zero_()
#     if x.grad is not None:
#         x.grad.zero_()
#
#     execute_model_with_gradient(model, node, x, edge_index, edge_label_index)
#
#     adj_grad = edge_mask.grad
#     adj_grad = torch.abs(adj_grad)
#     masked_adj = adj_grad + adj_grad.t()
#     masked_adj = torch.sigmoid(masked_adj)
#     masked_adj = masked_adj.cpu().detach().numpy()
#
#     feature_mask = torch.abs(x.grad).cpu().detach().numpy()
#
#     return np.max(feature_mask, axis=0), masked_adj


def grad_node_explanation(model, x, edge_index, edge_label_index):
    model.zero_grad()

    num_nodes, num_features = x.size()

    node_grad = torch.nn.Parameter(torch.ones(num_nodes))
    feature_grad = torch.nn.Parameter(torch.ones(num_features))

    node_grad.requires_grad = True
    feature_grad.requires_grad = True

    mask = node_grad.unsqueeze(0).T.matmul(feature_grad.unsqueeze(0))

    execute_model_with_gradient(model, mask * x, edge_index, edge_label_index)

    # node_mask = torch.abs(node_grad.grad).cpu().detach().numpy()
    # feature_mask = torch.abs(feature_grad.grad).cpu().detach().numpy()

    # node_mask = torch.abs(node_grad.grad)
    # feature_mask = torch.abs(feature_grad.grad)

    node_mask = node_grad.grad
    feature_mask = feature_grad.grad

    return feature_mask, node_mask


def grad(model, x, edge_index, edge_label_index):
    source_node, target_node = edge_label_index.numpy()[:, 0]

    computation_graph_feature_matrix, computation_graph_edge_index, mapping, hard_edge_mask = \
        subgraph_lp(model, source_node, target_node, x, edge_index)

    edge_label_index = mapping.unsqueeze(dim=1)
    feature_mask, node_mask = grad_node_explanation(model,
                                                    x=computation_graph_feature_matrix,
                                                    edge_index=computation_graph_edge_index,
                                                    edge_label_index=edge_label_index)
    node_feature_mask = feature_mask.unsqueeze(dim=0)
    node_mask = node_mask.unsqueeze(dim=0)

    explanation = Explanation(
        node_mask=node_mask,
        feature_mask=node_feature_mask,
        x=x,
        edge_index=edge_index,
        edge_label_index=edge_label_index
    )

    return explanation


def empty(model, x, edge_index, edge_label_index):
    source_node, target_node = edge_label_index.numpy()[:, 0]

    computation_graph_feature_matrix, computation_graph_edge_index, mapping, hard_edge_mask = \
        subgraph_lp(model, source_node, target_node, x, edge_index)

    computation_graph_feature_matrix

    feature_mask = torch.zeros(computation_graph_feature_matrix.size(0))

    feature_mask = feature_mask.unsqueeze(dim=0)

    explanation = Explanation(
        feature_mask=node_feature_mask,
        x=x,
        edge_index=edge_index,
        edge_label_index=edge_label_index
    )

    return explanation


def pgmexplainer(model, edge_label_index, test_data, num_hops, reduction, top_num_neighbors):
    explainer = pe.PGMExplainer(model, A=None, X=test_data.x, edge_index=test_data.edge_index, ori_pred=None,
                                num_layers=num_hops, mode=0, print_result=1, reduction=reduction)

    node_mask, subnodes, _, _ = explainer.explain_link(graph_data=test_data, edge_label_index=edge_label_index,
                                                       num_samples=100, top_node=None, p_threshold=0.05,
                                                       pred_threshold=0.1, top_num_neighbors=top_num_neighbors)

    # print(subnodes)  # [762, 859, 1030, 1199, 1566]

    explanation = Explanation(
        node_mask=node_mask,
        x=test_data.x,
        edge_index=test_data.edge_index,
        edge_label_index=edge_label_index
    )

    return explanation


def reduced_graph_complete_explanation(model, x, edge_index, edge_label_index, num_hops,
                                       top_num_neighbors, random_nodes):
    source_node, target_node = edge_label_index.numpy()[:, 0]

    num_nodes, num_edges = x.size(0), edge_index.size(1)
    original_neighbors, computation_graph_edge_index, _, _ = k_hop_subgraph(
        [source_node, target_node], num_hops=num_hops, edge_index=edge_index,
        relabel_nodes=False, num_nodes=num_nodes, flow='source_to_target')
    original_neighbors = original_neighbors.numpy()

    print(f'Number of neighbors of edge {edge_label_index}: {len(original_neighbors)}')
    print('Neighbors before reduction: ', len(original_neighbors))

    neighbors, computation_graph_edge_index, edge_mask = reduce_graph(model, x, edge_index,
                                                                      edge_label_index, num_hops,
                                                                      computation_graph_edge_index, top_num_neighbors,
                                                                      random_nodes=random_nodes)

    print('Neighbors after reduction: ', len(neighbors))
    print(f'Number of edges in the computation graph after reduction: {computation_graph_edge_index.shape}')

    node_mask = [1 if n in neighbors else 0 for n in original_neighbors]

    explanation = Explanation(
        node_mask=node_mask,
        edge_to_include=edge_mask,
        x=x,
        edge_index=edge_index,
        edge_label_index=edge_label_index
    )

    return explanation


# def zorro(model, edge_label_index, test_data, num_hops):
def softzorro(model, x, edge_index, edge_label_index):
    zorro = SoftZorro(model, device='cpu', log=True, record_process_time=False, samples=100, learning_rate=0.01)
    node_mask, feature_mask = zorro.explain_link(edge_label_index=edge_label_index,
                                                 full_feature_matrix=x, edge_index=edge_index)

    explanation = Explanation(
        node_mask=node_mask,
        feature_mask=feature_mask,
        x=x,
        edge_index=edge_index,
        edge_label_index=edge_label_index
    )

    return explanation


def zorro_baseline(model, x, edge_index, edge_label_index, tau):
    zorro = ZorroBaseline(model, device='cpu', log=True, record_process_time=False, samples=100)

    node_mask, feature_mask, fidelity_list = zorro.explain_link(edge_label_index=edge_label_index,
                                                                full_feature_matrix=x, edge_index=edge_index, tau=tau)

    explanation = Explanation(
        node_mask=node_mask,
        feature_mask=feature_mask,
        x=x,
        edge_index=edge_index,
        edge_label_index=edge_label_index,
        fidelity_list=fidelity_list
    )

    return explanation


def zorro(model, x, edge_index, edge_label_index, tau, model_type, dataset):
    zorro = Zorro(model, model_type=model_type, dataset=dataset, device='cpu', log=True, greedy=True,
                  record_process_time=False, samples=100, path_to_precomputed_distortions=None)

    minimal_nodes_and_features_sets, count = zorro.explain_link(edge_label_index=edge_label_index,
                                                                full_feature_matrix=x, edge_index=edge_index, tau=tau,
                                                                save_initial_improve=False,
                                                                use_precomputed_node_mask_as_fixed=False,
                                                                use_precomputed_node_mask_as_initial=True)
    print(minimal_nodes_and_features_sets)

    fidelity_list = []

    explanation = Explanation(
        node_mask=minimal_nodes_and_features_sets[0][0],
        feature_mask=minimal_nodes_and_features_sets[0][1],
        count=count,
        x=x,
        edge_index=edge_index,
        edge_label_index=edge_label_index,
        fidelity_list=fidelity_list
    )

    return explanation


def graphlime(model, x, edge_index, edge_label_index):
    graphlime = GraphLIME(model, hop=2, rho=0.1, cached=True)

    res = graphlime.explain_link(x, edge_index, edge_label_index)

    print(res)

    explanation = Explanation(
        feature_mask=xx,
        x=x,
        edge_index=edge_index,
        edge_label_index=edge_label_index
    )

    return explanation

# def gradinput_node_explanation(model, node, x, edge_index, edge_label_index):
#     model.zero_grad()
#
#     x.requires_grad = True
#     if x.grad is not None:
#         x.grad.zero_()
#
#     execute_model_with_gradient(model, node, x, edge_index, edge_label_index)
#
#     feature_mask = torch.abs(x.grad * x).cpu().detach().numpy()
#
#     return np.mean(feature_mask, axis=0), np.mean(feature_mask, axis=1)
