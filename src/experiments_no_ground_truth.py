import os
import os.path as osp

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from torch_geometric import seed_everything

import numpy as np
import pandas as pd

import networkx as nx
from torch_geometric.utils.convert import from_networkx

from tqdm import tqdm
import shutil

from models import LinkGIN, LinkGCN, LinkSAGE, DeepVGAE
from decoders import InnerProductDecoder, CosineDecoder
from metrics import deletion_curve_edges, deletion_curve_features
from metrics import subgraph_lp, sub_edge_mask_to_sub_node_mask, fidelity, sparsity, edge_mask_to_node_mask, \
    fidelity_original
# fidelity_reduced_graph_complete_explanation
from utils import get_explanation
from torch_geometric.utils import k_hop_subgraph
from matplotlib import pyplot as plt
import seaborn as sb

from explainers import grad_node_explanation
from torch_geometric.nn import APPNP, MessagePassing

import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

graph_model = 'cora'  # PubMed, cora
model_name = 'gcn'  # ['gin', 'gcn', 'sage', 'vgae']
explainer = 'ig'
# ['random', 'ig', 'gnnexplainer', 'deconvolution', 'grad', 'empty', 'pgmexplainer', 'softzorro']
# reduced_graph_complete_explanation
# zorro_baseline, zorro

# Zorro parameters
tau = 0.02
seed = 0

decoder = 'inner'  # ['inner', 'cosine']
edge_noise_type = 'all'
# ['bernoulli_whole', 'bernoulli_computation', 'bernoulli_1/2', 'kde', 'kde_1-mask', 'none']
load_model = True

# Reduction parameters
reduction = True
top_num_neighbors = '50%'
# 'half', 5, 10
random_nodes = True

if model_name == 'vgae':
    sigmoid = False
else:
    sigmoid = True

if model_name == 'vgae':
    return_type = 'probs'
    from train_test import train_vgae as train
    from train_test import test_vgae as test
else:
    return_type = 'raw'
    from train_test import train
    from train_test import test

if explainer == 'random':
    sorting = 'random'
else:
    sorting = 'descending'

print(seed, graph_model, model_name, explainer, decoder, return_type)

output_folder = f"../outputs/{graph_model}/{model_name}/{explainer}/"
model_folder = f"../outputs/{graph_model}/{model_name}/"

for seed in range(1):
    print(seed, graph_model, model_name, explainer, decoder, return_type)

    seed_everything(seed)

    # The split is performed such that the training split does not include edges in validation and test splits;
    # and the validation split does not include edges in the test split.
    # In our case, we ingore the validation data for now.
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.0, num_test=0.2, is_undirected=True),
    ])

    dataset = graph_model.capitalize()
    path = osp.join('../', 'data', 'Planetoid')
    dataset = Planetoid(path, dataset, transform=transform)
    # train_data, val_data, test_data = dataset[0]
    train_data, _, test_data = dataset[0]

    seed_everything(seed)
    if model_name == 'gin':
        model = LinkGIN(train_data.num_features, 128, 64, sim=decoder).to(device)
        tot_epochs = 61
    if model_name == 'gcn':
        model = LinkGCN(train_data.num_features, 128, 64, sim=decoder).to(device)
        tot_epochs = 201
    if model_name == 'sage':
        model = LinkSAGE(train_data.num_features, 128, 64, sim=decoder).to(device)
        tot_epochs = 101
    if model_name == 'vgae':
        if decoder == 'inner':
            model = DeepVGAE(train_data.num_features, 128, 64, InnerProductDecoder()).to(device)
        if decoder == 'cosine':
            model = DeepVGAE(train_data.num_features, 128, 64, CosineDecoder()).to(device)
        tot_epochs = 1501

    if load_model:
        model.load_state_dict(torch.load(f"{model_folder}/model.pt"))
        model.eval()
        print('Model loaded')
        # val_auc = test(model, val_data)
        test_auc, test_accuracy = test(model, test_data)
        # print(f'Val auc: {val_auc:.4f}, Test auc: {test_auc:.4f}')
        print(f'Test auc: {test_auc:.4f}, Test accuracy: {test_accuracy:.4f}')

    else:
        seed_everything(0)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
        for epoch in range(1, tot_epochs):
            loss = train(model, optimizer, train_data)
            if epoch % 20 == 0:
                if model_name == 'vgae':
                    # val_auc = test(model, train_data, val_data)
                    test_auc, test_accuracy = test(model, train_data, test_data)
                else:
                    # val_auc = test(model, val_data)
                    test_auc, test_accuracy = test(model, test_data)
                # print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
                #       f'Test: {test_auc:.4f}')
                print(
                    f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Test auc: {test_auc:.4f}, Test accuracy: {test_accuracy:.4f}')
        if not osp.exists(osp.dirname(model_folder)):
            os.makedirs(osp.dirname(model_folder))
        torch.save(model.state_dict(), f'{model_folder}/model.pt')

    # Take the first 100 examples.
    # edge_label_index is the edge that is to be predicted
    metric_list = []
    # fidelity_original_list = []

    # select random edges
    selected_nodes_random_seed = 42
    num_selected_query_edges = 100

    # select edges with positive labels, and sample the num we need
    # TODO: val_data -> test_data
    positive_idx = (test_data.edge_label == torch.ones(test_data.edge_label.shape)).nonzero().flatten().numpy()
    print('The number of positive edges: ', len(positive_idx))
    # num_selected_query_edges = min(num_selected_query_edges, len(positive_idx))

    if num_selected_query_edges > len(positive_idx):
        num_selected_query_edges = len(positive_idx)

    # TODO: could still result in the problem of negative prediction despite the positive label
    np.random.seed(selected_nodes_random_seed)
    selected_edge_pair_idx = np.random.choice(positive_idx, num_selected_query_edges, replace=False)
    print('Selected edges index: ', selected_edge_pair_idx)

    print('Start to explain links.')
    start_time = time.time()

    total_time_to_explain = 0

    zorro_fidelity_list = []
    node_mask_num_list = []
    node_mask_proportion_list = []

    feature_mask_num_list = []
    feature_mask_proportion_list = []

    count_list = []

    for i in tqdm(selected_edge_pair_idx):
        # for i in tqdm(range(test_data.edge_label_index.size(1))[:num_selected_query_edges]):
        edge_label_index = test_data.edge_label_index[:, [i]]
        source_node, target_node = edge_label_index.numpy()[:, 0]

        target = test_data.edge_label[i].item()
        if sigmoid:
            pred = model(train_data.x, train_data.edge_index, edge_label_index).sigmoid().item()
        else:
            pred = model(train_data.x, train_data.edge_index, edge_label_index).item()

        # TODO: Only when the prediction is true positive, then compute the explanation.
        if target == 1 and int(pred > 0.5) == target:

            # TODO: 2 hop
            num_hops = 0
            for module in model.modules():
                if isinstance(module, MessagePassing):
                    if isinstance(module, APPNP):
                        num_hops += module.K
                    else:
                        num_hops += 1

            explain_start_time = time.time()

            if explainer == 'reduced_graph_complete_explanation':
                # reduced_graph_f_upper_limit

                # baseline explainer that represents the upper limit of fidelity
                # that can be achieved by the reduced computation graph

                # select all the nodes, features, and edges in the computation subgraph
                # need to return the exact idxs of the nodes?

                edge_mask = np.ones(train_data.edge_index.size(1))
                feature_mask = np.ones(train_data.x.size(1))
                explanation = get_explanation(explainer, model, train_data, edge_label_index, test_data, num_hops,
                                              top_num_neighbors=top_num_neighbors, random_nodes=random_nodes)
                node_mask = np.expand_dims(explanation['node_mask'], axis=0)  # (1, 31)
                # edge_to_include is binary masks over the edges of the computation graph to
                # represent the edges in the reduced computation graph
                edge_to_include = explanation['edge_to_include']  # [96]

            elif explainer == 'pgmexplainer':
                # edge_mask = np.zeros(train_data.edge_index.size(1))
                # feature_mask = np.zeros(train_data.x.size(1))
                edge_mask = np.ones(train_data.edge_index.size(1))
                feature_mask = np.ones(train_data.x.size(1))

                print('-' * 100)
                t = time.time()
                explanation = get_explanation(explainer, model, train_data, edge_label_index, test_data, num_hops,
                                              reduction=reduction, top_num_neighbors=top_num_neighbors,
                                              return_type=return_type)
                node_mask = np.expand_dims(explanation['node_mask'], axis=0)  # (1, 31)
                print(f'Time taken to reduce is: {time.time() - t}')
                print('-' * 100)

                node_mask_num = np.count_nonzero(node_mask[0])
                node_mask_proportion = node_mask_num / len(node_mask[0])

                node_mask_num_list.append(node_mask_num)
                node_mask_proportion_list.append(node_mask_proportion)

                print(f'Node mask num: {node_mask_num}, Node mask proportion: {node_mask_proportion}')

            elif explainer in ['zorro_baseline', 'zorro']:
                # edge_mask = np.zeros(train_data.edge_index.size(1))
                edge_mask = np.ones(train_data.edge_index.size(1))

                # Feature Mask, Node Mask
                explanation = get_explanation(explainer, model, train_data, edge_label_index, test_data, num_hops,
                                              tau=tau, return_type=return_type, model_type=model_name,
                                              dataset=graph_model)
                feature_mask = explanation['feature_mask']
                node_mask = explanation['node_mask']

                fidelity_list = explanation['fidelity_list']
                zorro_fidelity_list.append(fidelity_list)
                print(fidelity_list)

                # calculate node mask and proportion
                node_mask_num = np.count_nonzero(node_mask[0])
                node_mask_proportion = node_mask_num / len(node_mask[0])
                node_mask_num_list.append(node_mask_num)
                node_mask_proportion_list.append(node_mask_proportion)

                # calculate feature mask and proportion
                feature_mask_num = np.count_nonzero(feature_mask[0])
                feature_mask_proportion = feature_mask_num / len(feature_mask[0])
                feature_mask_num_list.append(feature_mask_num)
                feature_mask_proportion_list.append(feature_mask_proportion)

                print(f'Node mask num: {node_mask_num}, Node mask proportion: {node_mask_proportion}')
                print(f'Feature mask num: {feature_mask_num}, Feature mask proportion: {feature_mask_proportion}')

                if explainer == 'zorro':
                    count = explanation['count']
                    count_list.append(count)
                    print(f'count of iterations to get this explanation: {count}')

            elif explainer == 'softzorro':
                # edge_mask = np.zeros(train_data.edge_index.size(1))
                edge_mask = np.ones(train_data.edge_index.size(1))

                # Feature Mask, Node Mask
                explanation = get_explanation(explainer, model, train_data, edge_label_index, test_data, num_hops,
                                              return_type=return_type)
                node_mask = explanation['node_mask']
                feature_mask = explanation['feature_mask']

            elif explainer in ['gnnexplainer', 'ig', 'deconvolution', 'grad']:
                # node_mask = np.zeros(train_data.x.size(0))
                node_mask = np.ones(train_data.x.size(0))

                # Feature Mask, Edge Mask
                explanation = get_explanation(explainer, model, train_data, edge_label_index, test_data, num_hops,
                                              return_type=return_type)

                # Only keep the positive masks and make the rest to be 0
                # edge_mask = explanation['edge_mask'].numpy()  # level mask with shape[num_edges]
                # edge_mask = np.array([max(element, 0) for element in edge_mask])

                edge_mask = explanation['edge_mask']  # level mask with shape[num_edges]
                edge_mask = torch.Tensor([max(element.item(), 0) for element in explanation['edge_mask']])

                feature_mask = explanation['node_mask'].numpy()
                feature_mask = np.array([[max(element, 0) for element in sub_list] for sub_list in feature_mask])

                # TODO: 检查这里是否需要
                # Take the average of the node feature mask: (2708, 1433) -> (1, 1433)
                feature_mask = np.expand_dims(feature_mask.mean(axis=0), axis=0)  # (1, 1433)

            # elif explainer == 'graphlime':
            #     edge_mask = np.ones(train_data.edge_index.size(1))
            #     explanation = get_explanation(explainer, model, train_data, edge_label_index, test_data, num_hops,
            #                                   top_num_neighbors=top_num_neighbors)
            #     # node_mask = np.expand_dims(explanation['node_mask'], axis=0)  # (1, 31)
            #     node_feature_mask = np.expand_dims(explanation['feature_mask'], axis=0)
            #     print(node_feature_mask)

            # elif explainer == 'empty':
            #     edge_mask = np.zeros(train_data.edge_index.size(1))
            #     feature_mask = np.zeros(train_data.x.size(1))
            #     feature_mask = np.expand_dims(feature_mask, axis=0)  # (1, 1433)

            # elif explainer == 'grad':
            #     explanation = get_explanation(explainer, model, train_data, edge_label_index, test_data, num_hops,
            #                                   return_type=return_type)
            #     node_mask = explanation['node_mask'].numpy()
            #     feature_mask = explanation['feature_mask'].numpy()
            #     edge_mask = np.zeros(train_data.edge_index.size(1))

            total_time_to_explain += (time.time() - explain_start_time)

            # Get the subgraph for the source and target nodes
            # 两种做法的sub_edge_mask相同，因为不涉及具体的edge_index，但是sub_edge_index不同，
            #  因为进行了relabel，因此需要使用原始的，因为后面deletion计算用到了这里的参数
            # For deletion curves of the original paper.
            subset, sub_edge_index, sub_mapping, sub_edge_mask = k_hop_subgraph(
                [source_node, target_node], num_hops, train_data.edge_index)

            # For converting edge masks to node masks within the computation graph
            subset_relabel, sub_edge_index_relabel, \
                sub_mapping_relabel, sub_edge_mask_relabel = k_hop_subgraph([source_node, target_node], num_hops,
                                                                            train_data.edge_index,
                                                                            num_nodes=train_data.x.shape[0],
                                                                            relabel_nodes=True)

            # # 1. Edge deletion Curve
            # deletion_curve = deletion_curve_edges(
            #     model,
            #     train_data.x,
            #     sub_edge_index,
            #     edge_label_index,
            #     edge_mask[sub_edge_mask],
            #     sigmoid=sigmoid,
            #     sorting=sorting
            # )
            # output_path = osp.join(output_folder, 'curves',
            #                        f"{seed}_{i}_{source_node}_{target_node}_{pred:.3f}_{target}_{test_auc:.3f}_{decoder}_edge_deletion_curve.npy")
            # if not osp.exists(osp.dirname(output_path)):
            #     os.makedirs(osp.dirname(output_path))
            # with open(output_path, 'wb') as f:
            #     np.save(f, deletion_curve)
            #
            # # 检查是否需要
            # # #  Features / Nodes
            # # if explainer in ['ig', 'deconvolution']:
            # #     feature_mean_mask = feature_mask.mean(axis=0)
            # # elif explainer == 'gnnexplainer':
            # #     feature_mean_mask = feature_mask.mean(axis=0)
            #
            # feature_mean_mask = feature_mask.mean(axis=0)
            # # node_mean_mask = feature_mask.mean(axis=1)
            # feature_base_values = train_data.x.mean(dim=0)
            #
            # # 2. Feature deletion Curve
            # feature_deletion_curve = deletion_curve_features(
            #     model,
            #     train_data.x,
            #     edge_index=sub_edge_index,
            #     edge_label_index=edge_label_index,
            #     feature_mask=feature_mean_mask,
            #     feature_base_values=feature_base_values,
            #     sigmoid=sigmoid,
            #     sorting=sorting
            # )
            #
            # output_path = osp.join(output_folder, 'curves',
            #                        f"{seed}_{i}_{source_node}_{target_node}_{pred:.3f}_{target}_{test_auc:.3f}_{decoder}_feature_deletion_curve.npy")
            # with open(output_path, 'wb') as f:
            #     np.save(f, feature_deletion_curve)

            if explainer == 'pgmexplainer':
                # Only node mask
                # node_mask should be the masks over nodes only in the computation graph

                f_original, change_in_pred_prob, \
                    predicted_label = fidelity(model,
                                               source_node=source_node,
                                               target_node=target_node,
                                               full_feature_matrix=train_data.x,
                                               edge_index=train_data.edge_index,
                                               node_mask=node_mask,
                                               feature_mask=None,
                                               # edge_mask should only contain the edge masks of the computation graph.
                                               # And we need to distort edge masks based on this.
                                               edge_mask=None,  # [96]
                                               samples=100,
                                               random_seed=12345,
                                               device="cpu",
                                               validity=False,
                                               edge_noise_type=edge_noise_type)
                feature_sparsity = None
                edge_sparsity = None
                node_sparsity_original = sparsity(torch.Tensor(node_mask))
                node_sparsity_converted = None

                fid_plus_label, fid_plus_prob, fid_minus_label, fid_minus_prob = \
                    fidelity_original(model, source_node=source_node, target_node=target_node,
                                      target_label=target, full_feature_matrix=train_data.x,
                                      edge_index=train_data.edge_index, node_mask=node_mask,
                                      feature_mask=None, edge_mask=None, device="cpu")

            if explainer == 'reduced_graph_complete_explanation':
                f_original, change_in_pred_prob, \
                    predicted_label = fidelity(model,
                                               source_node=source_node,
                                               target_node=target_node,
                                               full_feature_matrix=train_data.x,
                                               edge_index=train_data.edge_index,
                                               node_mask=node_mask,
                                               feature_mask=None,
                                               # edge_mask should only contain the edge masks of the computation graph.
                                               # And we need to distort edge masks based on this.
                                               edge_mask=None,
                                               # [96]
                                               samples=100,
                                               random_seed=12345,
                                               device="cpu",
                                               validity=False,
                                               edge_noise_type=edge_noise_type)

                feature_sparsity = None
                edge_sparsity = None
                node_sparsity_original = sparsity(torch.Tensor(node_mask))
                node_sparsity_converted = None

                fid_plus_label, fid_plus_prob, fid_minus_label, fid_minus_prob = \
                    fidelity_original(model, source_node=source_node, target_node=target_node,
                                      target_label=target, full_feature_matrix=train_data.x,
                                      edge_index=train_data.edge_index, node_mask=node_mask,
                                      feature_mask=None, edge_mask=None, device="cpu")

            if explainer in ['zorro_baseline', 'zorro', 'softzorro']:
                # node mask and feature mask
                f_original, change_in_pred_prob, \
                    predicted_label = fidelity(model,
                                               source_node=source_node,
                                               target_node=target_node,
                                               full_feature_matrix=train_data.x,
                                               edge_index=train_data.edge_index,
                                               node_mask=node_mask,
                                               feature_mask=feature_mask,
                                               edge_mask=None,  # [96]
                                               samples=100,
                                               random_seed=12345,
                                               device="cpu",
                                               validity=False,
                                               edge_noise_type=edge_noise_type)
                edge_sparsity = None
                node_sparsity_original = sparsity(torch.Tensor(node_mask))
                node_sparsity_converted = None

                fid_plus_label, fid_plus_prob, fid_minus_label, fid_minus_prob = \
                    fidelity_original(model, source_node=source_node, target_node=target_node,
                                      target_label=target, full_feature_matrix=train_data.x,
                                      edge_index=train_data.edge_index, node_mask=node_mask,
                                      feature_mask=feature_mask, edge_mask=None, device="cpu")

            # 1. Compute the feature sparsity
            feature_sparsity = sparsity(torch.Tensor(feature_mask))

            # 2. Compute the RDT-Fidelity, Validity, and Node Sparsity
            # if explainer == 'grad':
            #     # RDT-Fidelity
            #     f, predicted_label = fidelity(model,
            #                                   source_node=source_node,
            #                                   target_node=target_node,
            #                                   full_feature_matrix=train_data.x,
            #                                   edge_index=train_data.edge_index,
            #                                   node_mask=node_mask,
            #                                   feature_mask=feature_mask,
            #                                   edge_mask=None,  # None
            #                                   samples=100,
            #                                   random_seed=12345,
            #                                   device="cpu",
            #                                   validity=False)
            #     # node sparsity
            #     node_sparsity = sparsity(torch.Tensor(node_mask))

            if explainer in ['gnnexplainer', 'ig', 'deconvolution', 'grad']:
                # Feature Mask and Edge Mask, no Node Mask

                # TODO: in computing sparsity, do we need to include negative values?
                # node sparsity
                # Convert the edge mask to node mask: originally edge_mask,
                # but in our case, only the edge masks in the computation graph.

                # TODO: 1. Use edge masks of the whole graph, and filter based on the node id
                #  2. Only use edges from the computation graph, and convert to the corresponding id

                # node_mask_converted = edge_mask_to_node_mask(train_data, edge_mask[sub_edge_mask], aggregation="sum")

                # Option 2. Within the computation graph, convert the sub edge masks into sub node masks
                # TODO: 需要确保relabel前后的对应关系相同，必须relabel是因为在调用转换函数时，
                #  zip(sub_edge_mask, sub_edge_index.T)，取sub_edge_index.T中的nodes对应的source and target nodes时
                #  需要对应node_weights中的index，而node_weights的shape为[1,31] (only computation graph)
                node_mask_converted = sub_edge_mask_to_sub_node_mask(sub_num_nodes=subset_relabel.shape[0],
                                                                     sub_edge_index=sub_edge_index_relabel,
                                                                     sub_edge_mask=edge_mask[sub_edge_mask],
                                                                     aggregation="sum")
                node_mask_converted = node_mask_converted.unsqueeze(dim=0)  # [1, 31]
                # print('node_mask_converted: ', node_mask_converted)

                # TODO: check why there are nan values
                node_sparsity_converted = sparsity(torch.Tensor(node_mask_converted))
                node_sparsity_original = 0.
                edge_sparsity = sparsity(torch.Tensor(edge_mask))

                # TODO:
                #  case 1: converted node mask, no edge mask -> original RDT-Fidelity (feature perturbed)
                #  case 2: no node mask, but edge mask -> edge-perturbed RDT-Fidelity with all nodes selected
                #  case 3: node mask, edge mask -> edge-perturbed RDT-Fidelity with node masks by explainers

                # case 1
                f_original, change_in_pred_prob, \
                    predicted_label = fidelity(model,
                                               source_node=source_node,
                                               target_node=target_node,
                                               full_feature_matrix=train_data.x,
                                               edge_index=train_data.edge_index,
                                               node_mask=node_mask_converted,  # [1, 31]
                                               feature_mask=feature_mask,  # [1, 1433]
                                               edge_mask=None,
                                               samples=100,
                                               random_seed=12345,
                                               device="cpu",
                                               validity=False,
                                               edge_noise_type=edge_noise_type)

                # TODO: 这里需要进一步确认
                #  instead of using the node mask converted as node masks, we now use all the nodes,
                #  but also include edge masks as edge weights
                fid_plus_label, fid_plus_prob, fid_minus_label, fid_minus_prob = \
                    fidelity_original(model, source_node=source_node, target_node=target_node,
                                      target_label=target, full_feature_matrix=train_data.x,
                                      edge_index=train_data.edge_index, node_mask=None,  # node_mask_converted
                                      feature_mask=feature_mask, edge_mask=edge_mask[sub_edge_mask], device="cpu")

                print('edge_mask of computation graph: ', edge_mask[sub_edge_mask])

                # case 2
                # f_with_edge_mask = None
                f_with_edge_mask, _, _ = fidelity(model,
                                                  source_node=source_node,
                                                  target_node=target_node,
                                                  full_feature_matrix=train_data.x,
                                                  edge_index=train_data.edge_index,
                                                  node_mask=None,
                                                  feature_mask=feature_mask,
                                                  # edge_mask should only contain the edge masks of the computation graph.
                                                  # And we need to distort edge masks based on this.
                                                  edge_mask=edge_mask[sub_edge_mask],  # [96]    # None
                                                  samples=100,
                                                  random_seed=12345,
                                                  device="cpu",
                                                  validity=False,
                                                  edge_noise_type=edge_noise_type)

            # f_with_edge_mask = None

            fidelity_dict = {
                'fid_plus_label': fid_plus_label,
                'fid_plus_prob': fid_plus_prob,
                'fid_minus_label': fid_minus_label,
                'fid_minus_prob': fid_minus_prob
            }

            # fidelity_original_list.append(fidelity_dict)
            print(f'Fidelity dict is {fidelity_dict}')

            # TODO: Attention
            # the prediction is generated based on using only the subgraph as input,
            metric_dict = {'link': (source_node, target_node),
                           'true_label': target,
                           'prediction_with_comp_subgraph': predicted_label,
                           'fidelity': f_original,
                           'fidelity_change_in_pred_prob': change_in_pred_prob,
                           'fidelity_original': fidelity_dict,
                           'fidelity_with_edge_mask': f_with_edge_mask,
                           'feature_sparsity': feature_sparsity,
                           'edge_sparsity': edge_sparsity,
                           'node_sparsity_converted': node_sparsity_converted,
                           'node_sparsity_original': node_sparsity_original,
                           'total_time_to_explain': total_time_to_explain}

            if explainer == 'zorro':
                metric_dict['count'] = count

            metric_list.append(metric_dict)
            print(f'Metric dict is {metric_dict}')

    print('time_taken for explaining: ', total_time_to_explain)

    time_taken = time.time() - start_time
    print('time_taken for explaining + evaluation: ', time_taken)

    print('metric_list: ', metric_list)
    output_path = osp.join(output_folder, 'metric_list.npy')
    if not osp.exists(osp.dirname(output_path)):
        os.makedirs(osp.dirname(output_path))
    with open(output_path, 'wb') as f:
        np.save(f, metric_list)

    # print('fidelity_original_list: ', fidelity_original_list)
    # output_path = osp.join(output_folder, 'fidelity_original_list.npy')
    # with open(output_path, 'wb') as f:
    #     np.save(f, fidelity_original_list)

    print(zorro_fidelity_list)
    print()
    print(node_mask_num_list)
    print()
    print(node_mask_proportion_list)
    print()
    print(feature_mask_num_list)
    print()
    print(feature_mask_proportion_list)
    print()
    print(count_list)
