import logging

logger = logging.getLogger(__name__)

import torch
from typing import Union
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import remove_isolated_nodes, remove_self_loops, to_dense_adj, to_dense_batch

def find_common_nodes(graphs, frequency=50):
    x_all = []
    for graph in graphs:
        x_all.append(graph.x.sum(dim=0))
    x_all = torch.stack(x_all).sum(dim=0)
    common_nodes = torch.where(x_all > frequency)[0]
    logger.info(f'Using node frequency limit of: {frequency}')
    logger.info(f'Number of node labels: , {len(x_all)}')
    logger.info(f'There are {len(common_nodes)} common node labels!')
    return common_nodes

def contains_only_common_nodes(graph, common_nodes):
    return graph.x.sum() == graph.x[:, common_nodes].sum()

def remove_node_attribute(graph, attribute_idx):
    # Remove edges to nodes with this attibute
    # Remove nodes which had this attribute
    # Remove attribute
    return graph

# Filters away graphs larger than a certain size
def remove_large_graphs(limit=26):
    def remove_large_graphs_(data : Data) -> bool:
        return (data.num_nodes <= limit) and not data.has_isolated_nodes()
    return remove_large_graphs_

class HydrogenRemover(BaseTransform):
    def __call__(self, data: Union[Data, HeteroData]):
        # Change edge features:
        number_of_edges = data.num_edges
        updated_edges_index = []
        updated_edge_attr = []
        start_nodes = data.edge_index[0]
        end_nodes = data.edge_index[1]
        for i in range(number_of_edges):
            if data.x[start_nodes[i], 0] != 1 and data.x[end_nodes[i], 0] != 1:
                updated_edges_index.append(data.edge_index[:,i])
                updated_edge_attr.append(data.edge_attr[i])
            else:
                updated_edges_index.append(torch.zeros_like(data.edge_index[:,i]))
                updated_edge_attr.append(torch.zeros_like(data.edge_attr[i]))
        if len(updated_edges_index ) > 0:
            data.edge_index = torch.stack(updated_edges_index).T
            data.edge_attr = torch.stack(updated_edge_attr)
        else:
            data.edge_index = None
            data.edge_attr = None

        # Change features:
        edge_idx, edge_attr, node_mask = remove_isolated_nodes(data.edge_index, edge_attr=data.edge_attr, num_nodes=data.num_nodes)
        edge_idx, edge_attr = remove_self_loops(edge_idx, edge_attr)

        # Forces model to keep the first atom:
        node_mask[0] = True

        # New molecule object
        data.edge_index = edge_idx
        data.edge_attr = edge_attr
        data.x = data.x[node_mask]
        data.x = data.x[:,1:5] # Removes the hydrogen feature and none nodetype features.

        return data

class NodeLimitFilterTransform(BaseTransform):
    def __init__(self, graph_size_limit):
        self.graph_size_limit = graph_size_limit
        self.condition_fn = (lambda data: data.x.shape[0] <= self.graph_size_limit)

    def __call__(self, data):
        return self.condition_fn(data)

class RemoveUncommonNodes(BaseTransform):
    def __init__(self, list_of_common_node_indices):
        super().__init__()
        self.list_of_common_node_indices = list_of_common_node_indices

    def __call__(self, data: Data):
        data.x = data.x[:, self.list_of_common_node_indices]
        return data


class ToOneHot(BaseTransform):
    def __init__(self, number_of_node_classes, number_of_edge_classes):
        super().__init__()
        self.number_of_node_classes = number_of_node_classes # 118
        self.number_of_edge_classes = number_of_edge_classes # 4

    def __call__(self, data: Data):
        x = data.x[:,0]
        edge_attr = data.edge_attr[:,0]
        data.x = torch.nn.functional.one_hot(x, num_classes=self.number_of_node_classes).float()
        data.edge_attr = torch.nn.functional.one_hot(edge_attr, num_classes=self.number_of_edge_classes).float()
        return data

class ToDense(BaseTransform):
    def __init__(self, graph_size_limit):
        super().__init__()
        self.graph_size_limit = graph_size_limit

    def __call__(self, data: Data):
        # Extract data
        x = data.x.clone()
        edge_attr = data.edge_attr.clone()
        edge_index = data.edge_index.clone()

        # Save dense
        dense_batch, nodes_to_include = to_dense_batch(x, batch=None, max_num_nodes=self.graph_size_limit)
        adj = to_dense_adj(edge_index, max_num_nodes=self.graph_size_limit, edge_attr=edge_attr)
        # Ensure correct format
        dense_batch = dense_batch[0, :, :].permute(1,0)
        if edge_attr==None:
            adj = adj.unsqueeze(dim=3)
        adj_features = adj[0,:,:,:].permute(2,0,1)

        # Make graph object
        dense_graph = Data(x=dense_batch.unsqueeze(dim=0), y=data.y.clone())
        dense_graph.b = nodes_to_include.int().float().unsqueeze(dim=0)
        dense_graph.adj = adj_features.sum(dim=0, keepdim=True).unsqueeze(dim=0)
        dense_graph.e = adj_features.unsqueeze(dim=0)

        return dense_graph
