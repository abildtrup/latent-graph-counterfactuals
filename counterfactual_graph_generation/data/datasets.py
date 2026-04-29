import os

import torch
from torch.nn.functional import one_hot
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.datasets import TUDataset, QM9
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.transforms import Compose
from ogb.utils.features import allowable_features
from ogb.graphproppred import PygGraphPropPredDataset

import counterfactual_graph_generation.data.dataset_utils as cgf_utils

class GraphDataset(Dataset):
    def __init__(self, root, graph_size_limit, graph_count, num_classes, dense_data_representation=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.root = root
        self.graph_size_limit = graph_size_limit
        self.dense_representation = dense_data_representation
        self.graph_count = graph_count
        self.num_classes = num_classes

        super(GraphDataset, self).__init__(root, transform, pre_transform)

    @property
    def processed_file_names(self):
        """ If these files are found in processed_dir, processing is skipped"""
        if self.dense_representation:
            return [f'data_dense_{i}.pt' for i in range(self.graph_count)]
        else:
            return [f'data_{i}.pt' for i in range(self.graph_count)]

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        return []

    def one_hot_from_label(self, labels):
        return one_hot(labels, num_classes=self.num_classes)

    def label_from_one_hot(self, one_hot):
        return torch.argmax(one_hot, dim=1)

    def num_classes(self,):
        return self.num_classes

    def len(self):
        return self.graph_count

    def download(self):
        pass

    def get_tudataset(self,):
        pass

    def get_idx_split(self,):
        pass

    def process(self):
        graphs = self.get_tudataset()
        common_nodes = cgf_utils.find_common_nodes(graphs=graphs)
        count = 0
        for _, graph in enumerate(graphs):
            if cgf_utils.contains_only_common_nodes(graph=graph, common_nodes=common_nodes) and graph.x.shape[0] <= self.graph_size_limit:
                # Save graph object
                x = graph.x[:, common_nodes].clone()
                edge_attr= graph.edge_attr.clone() if graph.edge_attr != None else None
                data = Data(edge_index=graph.edge_index.clone(),
                    edge_attr=edge_attr,
                    y=graph.y,
                    x=x,
                    num_nodes=graph.num_nodes,
                )
                torch.save(data, os.path.join(self.processed_dir, f'data_{count}.pt'))
                # Save dense
                dense_batch, nodes_to_include = to_dense_batch(x, batch=None, max_num_nodes=self.graph_size_limit)
                adj = to_dense_adj(graph.edge_index, max_num_nodes=self.graph_size_limit, edge_attr=graph.edge_attr)
                # Ensure correct format
                dense_batch = dense_batch[0, :, :].permute(1,0)
                if graph.edge_attr==None:
                    adj = adj.unsqueeze(dim=3)
                adj_features = adj[0,:,:,:].permute(2,0,1)
                torch.save([dense_batch, nodes_to_include.int().float(), adj_features.sum(dim=0, keepdim=True), adj_features, graph.y], os.path.join(self.processed_dir, f'data_dense_{count}.pt'))
                count += 1

        print(f"There are {count} graphs!")

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.dense_representation:
            data = torch.load(os.path.join(self.processed_dir,
                                        f'data_dense_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir,
                                        f'data_{idx}.pt'))
        return data


class AIDS(GraphDataset):
    def __init__(self, root, graph_size_limit, dense_data_representation=False, transform=None, pre_transform=None):
        graph_count = 1635
        num_classes = 9
        super(AIDS, self).__init__(
            root=root,
            graph_size_limit=graph_size_limit,
            graph_count=graph_count,
            num_classes=num_classes,
            dense_data_representation=dense_data_representation
        )

    def get_tudataset(self):
        return TUDataset(root=os.path.join(self.root, 'raw/'), name='AIDS')


class Mutagenicity(GraphDataset):
    def __init__(self, root, graph_size_limit, dense_data_representation=False, transform=None, pre_transform=None):
        graph_count = 3935
        num_classes = 10
        super(Mutagenicity, self).__init__(
            root=root,
            graph_size_limit=graph_size_limit,
            graph_count=graph_count,
            num_classes=num_classes,
            dense_data_representation=dense_data_representation
        )

    def get_tudataset(self):
        return TUDataset(root=os.path.join(self.root, 'raw/'), name='Mutagenicity')


class NCI1(GraphDataset):
    def __init__(self, root, graph_size_limit, dense_data_representation=False, transform=None, pre_transform=None):
        graph_count = 3678
        num_classes = 10
        super(NCI1, self).__init__(
            root=root,
            graph_size_limit=graph_size_limit,
            graph_count=graph_count,
            num_classes=num_classes,
            dense_data_representation=dense_data_representation
        )

    def get_tudataset(self):
        return TUDataset(root=os.path.join(self.root, 'raw/'), name='NCI1')


class PROTEINS(GraphDataset):
    def __init__(self, root, graph_size_limit, dense_data_representation=False, transform=None, pre_transform=None):
        graph_count = 1101
        num_classes = 3
        super(PROTEINS, self).__init__(
            root=root,
            graph_size_limit=graph_size_limit,
            graph_count=graph_count,
            num_classes=num_classes,
            dense_data_representation=dense_data_representation
        )

    def get_tudataset(self):
        return TUDataset(root=os.path.join(self.root, 'raw/'), name='PROTEINS')


### Datasets directly inherited from pytorch geometric:
class custom_QM9(QM9):
    def __init__(self, root, graph_size_limit, dense_data_representation=False, transform=None, pre_transform=None):
        self.root = os.path.join(root, 'dense') if dense_data_representation else os.path.join(root, 'sparse')
        self.graph_size_limit = graph_size_limit
        self.dense_representation = dense_data_representation
        self.graph_count = 133885

        # Define transforms
        transforms = cgf_utils.HydrogenRemover()
        if dense_data_representation:
            transforms = Compose([cgf_utils.HydrogenRemover(), cgf_utils.ToDense(graph_size_limit=graph_size_limit)])

        # Initialize base class
        super(custom_QM9, self).__init__(
            root=self.root,
            transform = None, # Transformation at runtime
            pre_transform = transforms, # Transformations before saving to disk: Remove hydragen & make dense
            #pre_filter = cgf_utils.remove_large_graphs(limit=graph_size_limit), # Filters before saving to disk,
        )

        def __getitem__(self, idx):
            item = super(QM9, self).__getitem(idx)
            if self.dense_representation:
                item = item.x, item.b, item.adj, item.e, item.y
            return item

    def get_idx_split(self,):
            pass


class OgbMolHiv(Dataset):
    def __init__(self, root, graph_size_limit, dense_data_representation=False, transform=None, pre_transform=None, filter_transform=None):
        self.root = os.path.join(root, 'dense') if dense_data_representation else os.path.join(root, 'sparse')
        self.graph_size_limit = graph_size_limit
        self.dense_representation = dense_data_representation
        self.possible_node_features = allowable_features['possible_atomic_num_list']
        self.possible_edge_features = allowable_features['possible_bond_type_list']

        # Define transforms
        if pre_transform == None and dense_data_representation:
            pre_transform = Compose([cgf_utils.ToOneHot(len(self.possible_node_features), len(self.possible_edge_features))])

        if transform == None:
            transform = None

        if filter_transform == None:
            filter_transform = Compose([cgf_utils.NodeLimitFilterTransform(graph_size_limit)])

        self.dense_transform = Compose([cgf_utils.ToDense(graph_size_limit=graph_size_limit)])

        # Get molecular dataset
        self.ogb_dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root=self.root, transform=transform, pre_transform=pre_transform)

        # Filter dataset
        self.indices = torch.Tensor([i for i, data in enumerate(self.ogb_dataset) if filter_transform(data)]).long()

    def __len__(self,):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        item = self.ogb_dataset[real_idx]
        if self.dense_representation:
            item = self.dense_transform(item)
            item = item.x.squeeze(0), item.b.squeeze(0), item.adj.squeeze(0), item.e.squeeze(0), item.y.squeeze(0)
        return item

    def len(self,):
        return self.__len__()

    def get(self, idx):
        return self.__getitem__(idx)

    def get_idx_split(self,):
        idx_split = self.ogb_dataset.get_idx_split()
        filtered_split = {'train':None, 'valid':None, 'test':None}
        for item in idx_split.items():
            key, value = item
            mask = torch.isin(self.indices, value)
            filtered_split[key] = torch.nonzero(mask).squeeze()
        return filtered_split
