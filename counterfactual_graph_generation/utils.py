import logging

logger = logging.getLogger(__name__)

from pathlib import Path
import torch
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse, to_dense_adj, to_dense_batch
import networkx as nx
import pandas as pd

def save_df_as_csv(df: pd.DataFrame, path: str):
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath)

def assert_valid_loss(loss, msg : str):
    assert not torch.isinf(loss), f'Reconstruction loss is infinite: {msg}'
    return

# Saved
def save_tensor_list(tensor_list, folder : str, filename: str):
    tensor_list = [t.cpu() for t in tensor_list]
    path = './data/' + folder + filename + ".pt"
    torch.save(tensor_list, path)
    logger.info(f'Saved tensors at location: {path}')
    return

# Load all outputs:
def load_tensor_list(folder : str, filename : str):
    path = './data/' + folder + filename + ".pt"
    logger.info(f'Loading tensors at location: {path}')
    return torch.load('./data/' + folder + filename + ".pt", map_location=torch.device('cpu'))

def dense_graph_to_data_object(F: Tensor, B: Tensor, A: Tensor, E: Tensor, y = None, **kwargs):
    x = F.T[B.bool().view(-1),:]
    flatten_adj = A.view(-1, A.size(-1))
    flatten_adj = flatten_adj[B.bool().view(-1)]
    flatten_adj = flatten_adj.T[B.bool().view(-1)]
    edge_index = flatten_adj.nonzero().t()
    E_edge_index = A.view(-1, A.size(-1)).nonzero().t()
    edge_attr = E[:, E_edge_index[0], E_edge_index[1]].T
    graph = Data(x=x.clone(),
                 edge_index=edge_index.clone(),
                 edge_attr=edge_attr.clone(),
                 y=y.clone() if y != None else None
    )
    # print((A.sum(dim=0)==E.sum(dim=0)).sum())
    return graph

def batched_dense_graph_to_batched_data_object(F: Tensor, B: Tensor, A: Tensor, E: Tensor, y = None, **kwargs):
    graph_list = [dense_graph_to_data_object(F[i], B[i], A[i], E[i], y = y[i] if y != None else None) for i in range(len(F))]
    batch = Batch.from_data_list(graph_list)
    batch.num_nodes = len(batch.x)
    batch.validate(raise_on_error=True)
    return batch

def data_object_to_dense_graph(graph, graph_size_limit=30):
    dense_batch, nodes_to_include = to_dense_batch(graph.x, batch=None, max_num_nodes=graph_size_limit)
    adj = to_dense_adj(graph.edge_index, max_num_nodes=graph_size_limit, edge_attr=graph.edge_attr)
    dense_batch = dense_batch[0, :, :].permute(1,0)
    if graph.edge_attr==None:
        adj = adj.unsqueeze(dim=3)
    adj_features = adj[0,:,:,:].permute(2,0,1)
    return dense_batch, nodes_to_include.int().float(), adj_features.sum(dim=0, keepdim=True), adj_features, graph.y

def data_object_to_rdkit_molecule():
    return

def torch_to_numpy(t: Tensor):
    return t.cpu().numpy()

def remove_isolated_nodes_from_nx_graph(nx_graph):
    return nx_graph.remove_nodes_from(list(nx.isolates(nx_graph)))
