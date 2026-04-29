from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
import stochman.nnj as nnj
from torch_geometric.nn import GCNConv, global_max_pool

import counterfactual_graph_generation.models.equivariant_linear_layers as eqll

class DenseGraphClassifier(torch.nn.Module):
    def __init__(self, node_features, edge_features, num_classes=2, num_layers=3, filter_channels=20, ff_channels=200):
        super(DenseGraphClassifier, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.filter_channels = filter_channels

        # First layer only considers node feature information and maps to GRAPH
        self.layer1 = nn.Sequential(
            eqll.EquiLinear1to2(in_channels= 1 + node_features, out_channels=self.filter_channels), nn.SiLU(), nn.BatchNorm2d(num_features=self.filter_channels),
            nn.Conv2d(in_channels=self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nn.SiLU(), nn.BatchNorm2d(num_features=self.filter_channels),
            nn.Conv2d(in_channels=self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nn.SiLU(), nn.BatchNorm2d(num_features=self.filter_channels),
        )

        # First equivariant layer
        self.layer2 = nn.Sequential(
            eqll.EquiLinear2to2(in_channels= 1 + edge_features+ self.filter_channels, out_channels=self.filter_channels), nn.SiLU(), nn.BatchNorm2d(num_features=self.filter_channels),
            nn.Conv2d(in_channels=self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nn.LeakyReLU(), nn.BatchNorm2d(num_features=self.filter_channels),
            nn.Conv2d(in_channels=self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nn.LeakyReLU(), nn.BatchNorm2d(num_features=self.filter_channels),
        )

        # Second equivariant layer
        self.layer3 = nn.Sequential(
            eqll.EquiLinear2to2(in_channels= self.filter_channels, out_channels=self.filter_channels), nn.SiLU(), nn.BatchNorm2d(num_features=self.filter_channels),
            nn.Conv2d(in_channels=self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nn.SiLU(), nn.BatchNorm2d(num_features=self.filter_channels),
            nn.Conv2d(in_channels=self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nn.SiLU(), nn.BatchNorm2d(num_features=self.filter_channels),
        )

        # Invariant layer
        self.layer4 = nn.Sequential(
            eqll.EquiLinear2to1(in_channels=self.filter_channels, out_channels=ff_channels), nn.LeakyReLU(), nn.BatchNorm1d(num_features=ff_channels),
        )

        # Fully connected layer.
        self.fc = torch.nn.Linear(ff_channels, num_classes)

    def forward(self, inpt_batch):
        F, B, A, E, _ = inpt_batch

        # Pooling and FCs.
        B_and_F = torch.cat([F, B], dim=1)
        out = self.layer1(B_and_F)
        out = torch.concatenate([out, A, E], dim=1)
        out = self.layer2(out)
        out = self.layer3(out)
        node_embeddings = self.layer4(out)
        graph_embedding = torch.nn.functional.max_pool1d(node_embeddings, node_embeddings.shape[-1]).squeeze(dim=2)
        logits = self.fc(graph_embedding)
        return node_embeddings, graph_embedding, logits


class BinaryDenseGraphClassifier(torch.nn.Module):
    def __init__(self, num_classes=2, num_layers=3, filter_channels=20, ff_channels=200):
        super(BinaryDenseGraphClassifier, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.filter_channels = filter_channels
        self.ff_channels = ff_channels

        # Second equivariant layer
        self.embeddings = nn.Sequential(
            eqll.EquiLinear2to2(in_channels= 1, out_channels=self.filter_channels), nn.SiLU(),
            nn.Conv2d(in_channels=self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nn.LeakyReLU(), nn.BatchNorm2d(num_features=self.filter_channels),
            nn.Conv2d(in_channels=self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nn.LeakyReLU(), nn.BatchNorm2d(num_features=self.filter_channels),
            eqll.EquiLinear2to2(in_channels=self.filter_channels, out_channels=self.filter_channels), nn.SiLU(),
            nn.Conv2d(in_channels=self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nn.LeakyReLU(), nn.BatchNorm2d(num_features=self.filter_channels),
            nn.Conv2d(in_channels=self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nn.LeakyReLU(), nn.BatchNorm2d(num_features=self.filter_channels)

        )

        self.node_embeddings = nn.Sequential(
            eqll.EquiLinear2to1(in_channels=self.filter_channels, out_channels=ff_channels), nn.LeakyReLU(), nn.BatchNorm1d(num_features=ff_channels),
        )

        # Fully connected layer.
        self.fc = torch.nn.Linear(ff_channels, num_classes)

    def forward(self, inpt_batch):
        _, B, A, _, _ = inpt_batch
        B_outer = (B.permute(0,2,1) @ B).unsqueeze(dim=1)
        B_A = torch.concatenate([B_outer, A], dim=1)

        #  Pooling and FCs.
        embeddings = self.embeddings(B_A)
        node_emebddings = self.node_embeddings(embeddings)
        graph_embedding = torch.nn.functional.max_pool1d(node_emebddings, node_emebddings.shape[-1]).squeeze()
        logits = self.fc(graph_embedding)
        return None, None, logits


class RiemannianDenseGraphClassifier(nn.Module):
    def __init__(self, node_features, edge_features, graph_size_limit, num_classes=2, num_layers=3, filter_channels=20, ff_channels=200):
        super(RiemannianDenseGraphClassifier, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.filter_channels = filter_channels
        self.ff_channels = ff_channels
        self.graph_size_limit = graph_size_limit

        # First layer only considers node feature information and maps to GRAPH
        self.outerproduct = nnj.Sequential(
            eqll.OuterProductLayer()
        )

        # Second part of node embedding:
        self.node_embedding = nnj.Sequential(
            nnj.Conv2d(in_channels = (1 + node_features) + (1 + edge_features), out_channels=self.filter_channels, kernel_size=1), nnj.ELU(), nnj.BatchNorm2d(num_features=self.filter_channels),
            nnj.Conv2d(in_channels = self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nnj.ELU(), nnj.BatchNorm2d(num_features=self.filter_channels),
            nnj.MaxPool2d(kernel_size=(1, self.graph_size_limit)),
            nnj.Conv2d(in_channels=self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nnj.ELU(), nnj.BatchNorm2d(num_features=self.filter_channels),
            nnj.Conv2d(in_channels=self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nnj.ELU(), nnj.BatchNorm2d(num_features=self.filter_channels),
            eqll.OuterProductLayer(),
            nnj.Conv2d(in_channels = self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nnj.ELU(), nnj.BatchNorm2d(num_features=self.filter_channels),
            nnj.Conv2d(in_channels = self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nnj.ELU(), nnj.BatchNorm2d(num_features=self.filter_channels),
            nnj.MaxPool2d(kernel_size=(1, self.graph_size_limit)),
            nnj.Conv2d(in_channels=self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nnj.ELU(), nnj.BatchNorm2d(num_features=self.filter_channels),
            nnj.Conv2d(in_channels=self.filter_channels, out_channels=self.ff_channels, kernel_size=1), nnj.ELU(), nnj.BatchNorm2d(num_features=self.ff_channels),
        )

        # Invariant graph embedding
        self.graph_embedding = nnj.Sequential(
            nnj.MaxPool2d(kernel_size=(self.graph_size_limit, 1)),
            nnj.Flatten()
        )

        # Fully connected layer.
        self.fc = nnj.Linear(ff_channels, num_classes)

    def forward(self, inpt_batch, jacobian=False):
        F, B, A, E, _ = inpt_batch
        B_and_F = torch.cat([F, B], dim=1)
        if jacobian:
            # Forward:
            outer, outer_jac_b_f = self.outerproduct(B_and_F, jacobian)
            graph = torch.concatenate([outer, A, E], dim=1)
            node_embeddings, ne_jac_graph  = self.node_embedding(graph, jacobian)
            graph_embedding, ge_jac_ne = self.graph_embedding(node_embeddings, jacobian)
            logits, out_jac_ge = self.fc(graph_embedding, jacobian)
            # Forward jacobians:
            graph_jac_b_f = outer_jac_b_f
            ne_jac_b_f = torch.einsum('bijnmp, bnmpvw -> bijvw', ne_jac_graph, graph_jac_b_f)
            ge_jac_b_f = torch.einsum('binm, bnmvw -> bivw', ge_jac_ne, ne_jac_b_f)
            out_jac_b_f = torch.einsum('bin, bnvw -> bivw', out_jac_ge, ge_jac_b_f)
            graph_jac_a_e = torch.concatenate([nnj.identity(A), nnj.identity(E)], dim=1)
            ne_jac_a_e = torch.einsum('bijnmp, bnmpvwz -> bijvwz', ne_jac_graph, graph_jac_a_e)
            ge_jac_a_e = torch.einsum('binm, bnmvwz -> bivwz', ge_jac_ne, ne_jac_a_e)
            out_jac_a_e = torch.einsum('bin, bnvwz -> bivwz', out_jac_ge, ge_jac_a_e)
            # Total
            out_jac_b = out_jac_b_f[:,:1,:,:]
            out_jac_f = out_jac_b_f[:,1:,:,:]
            out_jac_a = out_jac_a_e[:,:1,:,:,:]
            out_jac_e = out_jac_a_e[:,1:,:,:,:]
            return (node_embeddings, graph_embedding, logits), (out_jac_b, out_jac_f, out_jac_a, out_jac_e)
        else:
            # Pooling and FCs.
            outer = self.outerproduct(B_and_F)
            graph = torch.concatenate([outer, A, E], dim=1)
            node_embeddings = self.node_embedding(graph)
            graph_embedding = self.graph_embedding(node_embeddings)
            logits = self.fc(graph_embedding)

        return node_embeddings, graph_embedding, logits


class GraphClassifier(torch.nn.Module):
    def __init__(self, num_features, num_classes=2, num_layers=3, dim=20, dropout=0.0):
        super(GraphClassifier, self).__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dim = dim
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.ns = torch.nn.ModuleList()

        # First GCN layer.
        self.convs.append(GCNConv(num_features, dim))
        self.ns.append(torch.nn.BatchNorm1d(dim))

        # Follow-up GCN layers.
        for i in range(self.num_layers - 1):
            self.convs.append(GCNConv(dim, dim))
            self.ns.append(torch.nn.BatchNorm1d(dim))

        # Fully connected layer.
        self.fc = torch.nn.Linear(dim, num_classes)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, GCNConv):
                m.reset_parameters()
            elif isinstance(m, torch.nn.BatchNorm1d):
                m.reset_parameters()
            elif isinstance(m, torch.nn.Linear):
                m.reset_parameters()

    def forward(self, data, edge_weight=None):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        # GCNs.
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.ns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout after every layer.

        # Pooling and FCs.
        node_embeddings = x
        graph_embedding = global_max_pool(node_embeddings, batch)
        logits = self.fc(graph_embedding)

        return node_embeddings, graph_embedding, logits


class PossibleGraphClassifiersEnum(Enum):
    RIEMANNIAN = RiemannianDenseGraphClassifier
    DENSE = DenseGraphClassifier
    BINARY = BinaryDenseGraphClassifier
    SPARSE = GraphClassifier
