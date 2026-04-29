import torch
import torch.nn.functional as Fn
from torch import nn
from torch.distributions import Categorical, OneHotCategorical
import stochman.nnj as nnj

import counterfactual_graph_generation.models.equivariant_linear_layers as eqll
from counterfactual_graph_generation.models.gumbel_softmax import WeightedGumbelSoftmaxNodesBool, WeightedGumbelSoftmaxNodesAttr, WeightedGumbelSoftmaxEdgesBool, WeightedGumbelSoftmaxEdgesAttr
#from counterfactual_graph_generation.models.rbf import RBF

class GNNDecoder(torch.nn.Module):
    def __init__(self, node_features: int, edge_features: int, latent_dim: int, filter_channels: int, node_threshold: float = 0.86, edge_threshold: float = 0.1):
        """
        We factor the output distribution as p(F, E | z) = p(F | z) p(E | z, E)
        """
        super(GNNDecoder, self).__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.filter_channels = filter_channels

        self.node_threshold = node_threshold
        self.edge_threshold = edge_threshold

        self.one_zero_node_decoder = nn.Sequential(
            eqll.EquiLinear1to1(in_channels=latent_dim, out_channels=self.filter_channels), nn.LeakyReLU(),
            nn.Conv1d(in_channels=self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nn.LeakyReLU(), nn.InstanceNorm1d(num_features=self.filter_channels),
            nn.Conv1d(in_channels=self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nn.LeakyReLU(), nn.InstanceNorm1d(num_features=self.filter_channels),
            nn.Conv1d(in_channels=self.filter_channels, out_channels=1, kernel_size=1)
        )

        self.node_decoder = nn.Sequential(
            eqll.EquiLinear1to1(in_channels= 1 + latent_dim, out_channels=self.filter_channels), nn.LeakyReLU(),
            nn.Conv1d(in_channels=self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nn.LeakyReLU(), nn.InstanceNorm1d(num_features=self.filter_channels),
            nn.Conv1d(in_channels=self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nn.LeakyReLU(), nn.InstanceNorm1d(num_features=self.filter_channels),
            nn.Conv1d(in_channels=self.filter_channels, out_channels=node_features, kernel_size=1)
        )

        self.adjacency_decoder = nn.Sequential(
            eqll.EquiLinear2to2(in_channels=latent_dim, out_channels=self.filter_channels), nn.LeakyReLU(),
            nn.Conv2d(in_channels=self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nn.LeakyReLU(), nn.InstanceNorm2d(num_features=self.filter_channels),
            nn.Conv2d(in_channels=self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nn.LeakyReLU(), nn.InstanceNorm2d(num_features=self.filter_channels),
            eqll.EquiLinear2to2(in_channels=self.filter_channels, out_channels=self.filter_channels), nn.LeakyReLU(),
            nn.Conv2d(in_channels=self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nn.LeakyReLU(), nn.InstanceNorm2d(num_features=self.filter_channels),
            nn.Conv2d(in_channels=self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nn.LeakyReLU(), nn.InstanceNorm2d(num_features=self.filter_channels),
            nn.Conv2d(in_channels=self.filter_channels, out_channels= 1, kernel_size=1)
        )

        self.edge_attr_decoder = nn.Sequential(
            eqll.EquiLinear2to2(in_channels= 1 + latent_dim, out_channels=self.filter_channels), nn.LeakyReLU(),
            nn.Conv2d(in_channels=self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nn.LeakyReLU(), nn.InstanceNorm2d(num_features=self.filter_channels),
            nn.Conv2d(in_channels=self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nn.LeakyReLU(), nn.InstanceNorm2d(num_features=self.filter_channels),
            eqll.EquiLinear2to2(in_channels=self.filter_channels, out_channels=self.filter_channels), nn.LeakyReLU(),
            nn.Conv2d(in_channels=self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nn.LeakyReLU(), nn.InstanceNorm2d(num_features=self.filter_channels),
            nn.Conv2d(in_channels=self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nn.LeakyReLU(), nn.InstanceNorm2d(num_features=self.filter_channels),
            nn.Conv2d(in_channels=self.filter_channels, out_channels=edge_features, kernel_size=1)
        )

    def forward(self, z, B, F, A):
        z_B = torch.concatenate([z, B], dim=1)
        z_B_outer = (z_B.permute(0,2,1) @ z_B).unsqueeze(dim=1)
        z_B_A = torch.concatenate([z_B_outer, A], dim=1)
        # Number of nodes
        B_new = self.one_zero_node_decoder(z)
        # Adjacency matrix
        A_new = self.adjacency_decoder(z_B_outer)
        A_new = (0.5 * (A_new + A_new.permute(0,1,3,2))) #* (B.permute(0,2,1) @ B).unsqueeze(dim=1) # Ensures only connection between real edges can be considered edges
        # Node classes
        F_new = self.node_decoder(z_B) # Ensures only real nodes are given a class
        # Edge classes
        E_new = self.edge_attr_decoder(z_B_A)
        E_new = (0.5 * (E_new + E_new.permute(0,1,3,2))) # Ensures only edges are given a class
        return F_new, B_new, A_new, E_new

    def decode(self, z):
        # Decode a boolean tensor
        B_new = self.one_zero_node_decoder(z).sigmoid()

        # z and B_new
        # Get max likelihood F_new
        z_B = torch.concatenate([z, B_new], dim=1)
        F_raw = self.node_decoder(z_B) # B x C x V
        F_new = F_raw.softmax(dim=1)

        # Adjacency
        z_B_outer = (z_B.permute(0,2,1) @ z_B).unsqueeze(dim=1)
        A_intermediate = self.adjacency_decoder(z_B_outer)
        A_new = A_intermediate[:,0,:,:].unsqueeze(dim=1)
        A_new = (0.5 * (A_new + A_new.permute(0,1,3,2))).sigmoid()

        # Edge classes
        z_B_A = torch.concatenate([z_B_outer, A_new], dim=1)
        E_raw = self.edge_attr_decoder(z_B_A)
        E_new = (0.5 * (E_raw + E_raw.permute(0,1,3,2))).softmax(dim=1)

        return F_new, B_new, A_new, E_new

    def decode_discrete_graph(self, z, tau=1):
        # Decode a boolean tensor
        B_pos_logits = self.one_zero_node_decoder(z)
        B_neg_logits = torch.zeros_like(B_pos_logits)
        B_logits = torch.cat([B_neg_logits, B_pos_logits], dim=1)
        B_new = Fn.gumbel_softmax(B_logits, tau=tau, hard=True, dim=1)[:,1,:].unsqueeze(dim=1)

        # Get max likelihood F_new
        z_B = torch.concatenate([z, B_new], dim=1)
        F_raw = self.node_decoder(z_B) # B x C x V
        F_new = Fn.gumbel_softmax(F_raw, tau=tau, hard=True, dim=1) * B_new

        # Adjacency
        z_B_outer = (z_B.permute(0,2,1) @ z_B).unsqueeze(dim=1)
        A_intermediate = self.adjacency_decoder(z_B_outer)
        A_new = A_intermediate[:,0,:,:].unsqueeze(dim=1)
        A_pos_logits = (0.5 * (A_new + A_new.permute(0,1,3,2)))
        A_neg_logits = torch.zeros_like(A_pos_logits)
        A_logits = torch.cat([A_neg_logits, A_pos_logits], dim=1)
        A_one_hot = Fn.gumbel_softmax(A_logits, tau=tau, hard=True, dim=1)[:,1,:,:].unsqueeze(dim=1)
        A_tril = A_one_hot.tril(-1) # N x 1 x V x V
        A_new = (A_tril + A_tril.permute(0, 1, 3, 2)) * (B_new.permute(0,2,1) @ B_new).unsqueeze(dim=1)

        # Edge classes
        z_B_A = torch.concatenate([z_B_outer, A_new], dim=1)
        E_raw = self.edge_attr_decoder(z_B_A)
        E_logits = 0.5 * (E_raw + E_raw.permute(0,1,3,2))
        E_tril = Fn.gumbel_softmax(E_logits, tau=tau, hard=True, dim=1).tril(-1)
        E_new = (E_tril + E_tril.permute(0, 1, 3, 2)) * A_new

        return F_new, B_new, A_new, E_new

    def sample_B(self, z):
        B_dist = self.one_zero_node_decoder(z).sigmoid()
        return torch.bernoulli(B_dist).view(z.shape[0], 1, -1) # N x 1 x V

    def sample_F(self, z, B):
        z_B = torch.concatenate([z, B], dim=1)
        F_probs = self.node_decoder(z_B).softmax(dim=1).permute(0,2,1) # N x V x C
        F_dist = Categorical(probs=F_probs)
        F = torch.nn.functional.one_hot(F_dist.sample(), num_classes=self.node_features).permute(0,2,1) * B # N x C x V
        return F

    def sample_A(self, z, B):
        z_B = torch.concatenate([z, B], dim=1)
        z_B_outer = (z_B.permute(0,2,1) @ z_B).unsqueeze(dim=1)
        A_partial = self.adjacency_decoder(z_B_outer)
        A_probs = (0.5 * (A_partial + A_partial.permute(0,1,3,2))).sigmoid() # N x 1 x V x V
        A_tril = torch.tril(torch.bernoulli(A_probs), -1) # N x 1 x V x V
        A = (A_tril + A_tril.permute(0, 1, 3, 2)) * (B.permute(0,2,1) @ B).unsqueeze(dim=1)
        return A

    def sample_E(self, z, B, A):
        z_B = torch.concatenate([z, B], dim=1)
        z_B_outer = (z_B.permute(0,2,1) @ z_B).unsqueeze(dim=1)
        z_B_A = torch.concatenate([z_B_outer, A], dim=1)
        E_probs = self.edge_attr_decoder(z_B_A).softmax(dim=1).permute(0,2,3,1)
        E_dist = Categorical(probs=E_probs)
        E_tril = torch.nn.functional.one_hot(E_dist.sample(), num_classes=self.edge_features).permute(0,3,1,2) * torch.tril(A) # N x C x V x V
        E = (E_tril + E_tril.permute(0, 1, 3, 2))
        return E

    def sample(self, z):
        B = self.sample_B(z)
        F = self.sample_F(z, B)
        A = self.sample_A(z, B)
        E = self.sample_E(z, B, A)
        return F, B, A, E

    def log_prob(self, x, z):
        # The logprob of p(x|z)
        return

    def delete_last_layer(self,):
        self.one_zero_node_decoder = self.one_zero_node_decoder[:-1]
        self.node_decoder = self.node_decoder[:-1]
        self.adjacency_decoder = self.adjacency_decoder[:-1]
        self.edge_attr_decoder = self.edge_attr_decoder[:-1]


class RiemannianGNNDecoder(GNNDecoder):
    def __init__(self, node_features: int, edge_features: int, latent_dim: int, filter_channels: int, graph_size_limit: int, num_points: int = 10, node_threshold: float = 0.86, edge_threshold: float = 0.1):
        super(RiemannianGNNDecoder, self).__init__(node_features, edge_features, latent_dim, filter_channels)
        self.node_features = node_features
        self.edge_features = edge_features
        self.filter_channels = filter_channels
        self.latent_dim = latent_dim

        self.node_threshold = node_threshold
        self.edge_threshold = edge_threshold

        self.one_zero_node_decoder = nnj.Sequential( # TODO: add equivariant linear layer
            eqll.nnjEquiLinear1to1(in_channels=self.latent_dim, out_channels=self.filter_channels), nnj.ELU(),
            nnj.Conv1d(in_channels=self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nnj.ELU(), nnj.BatchNorm1d(num_features=self.filter_channels),
            nnj.Conv1d(in_channels=self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nnj.ELU(), nnj.BatchNorm1d(num_features=self.filter_channels),
            nnj.Conv1d(in_channels=self.filter_channels, out_channels=1, kernel_size=1), nnj.Sigmoid()
        )

        self.node_decoder = nnj.Sequential(
            eqll.nnjEquiLinear1to1(in_channels=latent_dim + 1, out_channels=self.filter_channels), nnj.ELU(),
            nnj.Conv1d(in_channels=self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nnj.ELU(), nnj.BatchNorm1d(num_features=self.filter_channels),
            nnj.Conv1d(in_channels=self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nnj.ELU(), nnj.BatchNorm1d(num_features=self.filter_channels),
            nnj.Conv1d(in_channels=self.filter_channels, out_channels=node_features, kernel_size=1)
        )

        self.outer_product_layer = nnj.Sequential(eqll.OuterProductLayer(), nnj.ELU())

        self.adjacency_decoder = nnj.Sequential(
            nnj.Conv2d(in_channels=self.latent_dim + 1 + self.node_features, out_channels=self.filter_channels, kernel_size=1), nnj.ELU(), nnj.BatchNorm2d(num_features=self.filter_channels),
            nnj.MaxPool2d(kernel_size=(1, graph_size_limit)),
            nnj.Conv2d(in_channels=self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nnj.ELU(), nnj.BatchNorm2d(num_features=self.filter_channels),
            nnj.Conv2d(in_channels=self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nnj.ELU(), nnj.BatchNorm2d(num_features=self.filter_channels),
            eqll.OuterProductLayer(),
            nnj.Conv2d(in_channels=self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nnj.ELU(), nnj.BatchNorm2d(num_features=self.filter_channels),
            nnj.Conv2d(in_channels=self.filter_channels, out_channels=1, kernel_size=1), nnj.Sigmoid()
        )

        self.edge_attr_decoder = nnj.Sequential(
            nnj.Conv2d(in_channels=self.latent_dim + 2 + self.node_features, out_channels=self.filter_channels, kernel_size=1), nnj.ELU(), nnj.BatchNorm2d(num_features=self.filter_channels),
            nnj.MaxPool2d(kernel_size=(1, graph_size_limit)),
            nnj.Conv2d(in_channels=self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nnj.ELU(), nnj.BatchNorm2d(num_features=self.filter_channels),
            nnj.Conv2d(in_channels=self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nnj.ELU(), nnj.BatchNorm2d(num_features=self.filter_channels),
            eqll.OuterProductLayer(),
            nnj.Conv2d(in_channels=self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nnj.ELU(), nnj.BatchNorm2d(num_features=self.filter_channels),
            nnj.Conv2d(in_channels=self.filter_channels, out_channels=self.edge_features, kernel_size=1)
        )

        # Tau predicter:
        #self.temperature_predicter = nnj.Sequential(
        #    RBF(dim=graph_size_limit, num_points=num_points),
        #    nnj.PosLinear(num_points, 1, bias=False),
        #    nnj.Reciprocal(b=1e-08),
        #)

        # Categorical decoders:
        self.gs_b = WeightedGumbelSoftmaxNodesBool()
        self.gs_f = WeightedGumbelSoftmaxNodesAttr()
        self.gs_a = WeightedGumbelSoftmaxEdgesBool()
        self.gs_e = WeightedGumbelSoftmaxEdgesAttr()

    def forward(self, z, B=None, F=None, A=None, jacobian=False):
        if jacobian and None in [B, F, A]:
            return self.decode_discrete_graph(z, jacobian=jacobian)
        else:
            # Number of nodes
            B_new = self.one_zero_node_decoder(z)
            # Node classes
            z_and_B = torch.concatenate([z, B], dim=1)
            F_new = self.node_decoder(z_and_B) * B # Ensures only real nodes are given a class
            # Outer product
            z_and_B_and_F = torch.concatenate([z, B, F], dim=1)
            outer = self.outer_product_layer(z_and_B_and_F)
            # Adjacency
            A_new = self.adjacency_decoder(outer) * (B.permute(0,2,1) @ B).unsqueeze(dim=1)
            # Edge classes
            z_and_B_and_F_and_A = torch.concatenate([outer, A], dim=1)
            E_new = self.edge_attr_decoder(z_and_B_and_F_and_A) # Ensures only edges are given a class
            E_new = (0.5 * (E_new + E_new.permute(0,1,3,2))) * A

            return F_new, B_new, A_new, E_new

    def jacobian(self, batch):
        return self.forward(batch, jacobian=True)

    def decode_discrete_graph(self, z, jacobian=False):
        if jacobian:
            # Find tau weight:
            tau, tau_jac_z = self.temperature_predicter(z.squeeze(), jacobian)
            latent_dim = z.shape[1]
            n = z.shape[2]
            tau_nodes, tau_nodes_jac_z = tau.view(-1, 1, 1).tile([1,1,n]), tau_jac_z.view(-1, 1, 1, latent_dim, n).tile([1,1,n,1,1])
            tau_edges, tau_edges_jac_z = tau.view(-1, 1, 1, 1).tile([1,1,n,n]), tau_jac_z.view(-1, 1, 1, 1, latent_dim, n).tile([1,1,n,n,1,1])

            ###### Node based ######
            # d_B: z -> (z, B)
            B_pos_logits, B_pos_jac_z = self.one_zero_node_decoder(z, jacobian) # batchsize x latent_dim x n_z x latent_dim x n_i
            tau_B_pos_logits, tau_B_pos_logits_jac_z = torch.concatenate([tau_nodes, B_pos_logits], dim=1),  torch.concatenate([tau_nodes_jac_z, B_pos_jac_z], dim=1)
            B, B_jac_tau_b_pos = self.gs_b(tau_B_pos_logits, jacobian)
            B_jac_z = torch.einsum('bijvw, bvwnm -> bijnm', B_jac_tau_b_pos, tau_B_pos_logits_jac_z)
            id_jac_z = nnj.identity(z)
            z_and_B, id_jac_z_b = torch.concatenate([z, B], dim=1), torch.concatenate([id_jac_z, B_jac_z], dim=1) # B x (latent_dim + 1) x n_z x latent_dim x n_i

            # d_F: (z, B) -> (z, B, F)
            F_raw, F_raw_jac_z_b = self.node_decoder(z_and_B, jacobian) # B x node_features x n_o x (latent_dim + 1) x n_z,
            F_raw_jac_z = torch.einsum('bijvw, bvwnm -> bijnm',F_raw_jac_z_b, id_jac_z_b)
            tau_F_raw, tau_F_raw_jac_z = torch.concatenate([tau_nodes, F_raw], dim=1),  torch.concatenate([tau_nodes_jac_z, F_raw_jac_z], dim=1)
            F_, F_jac_tau_f_raw = self.gs_f(tau_F_raw, jacobian)
            F_jac_z = torch.einsum('bijvw, bvwnm -> bijnm', F_jac_tau_f_raw, tau_F_raw_jac_z)
            B_ = B.tile(1, F_.shape[1], 1)
            F = F_ * B_
            F_jac_z = F_jac_z * B_[..., None, None] + F_[..., None, None] * B_jac_z.tile(1, F_.shape[1],1,1,1)
            z_and_B_and_F, id_jac_z_b_f = torch.concatenate([z, B, F], dim=1), torch.concatenate([id_jac_z_b, F_jac_z], dim=1)

            ###### Graph based #####
            # d: (z, B, F) -> (z, B, F)^T (z, B, F)
            outer, outer_jac_z_b_f = self.outer_product_layer(z_and_B_and_F, jacobian) # B x (latent dim + 1 + nodesfeatures) x z x z x (latent dim + 1 + nodesfeatures) x z
            outer_jac_z = torch.einsum('bijkvw, bvwnm -> bijknm', outer_jac_z_b_f, id_jac_z_b_f)

            # d_A: (z, B, F) -> (z, B, F, A)
            A_raw, A_jac_z_b_f_sq = self.adjacency_decoder(outer, jacobian)
            A_raw_jac_z = torch.einsum('bijkvwz, bvwznm -> bijknm' , A_jac_z_b_f_sq, outer_jac_z)
            tau_A_raw, tau_A_raw_jac_z = torch.concatenate([tau_edges, A_raw], dim=1),  torch.concatenate([tau_edges_jac_z, A_raw_jac_z], dim=1)
            A_, A_jac_tau_a_raw = self.gs_a(tau_A_raw, jacobian)
            A_jac_z = torch.einsum('bijkvwz, bvwznm -> bijknm', A_jac_tau_a_raw, tau_A_raw_jac_z)
            B_sq = outer[:,self.latent_dim:(self.latent_dim+1),:,:].tile(1, A_.shape[1], 1, 1)
            A = A_ * B_sq
            A_jac_z = A_jac_z * B_sq[..., None, None] + A_[..., None, None] * outer_jac_z[:,self.latent_dim:(self.latent_dim+1),:,:].tile(1, A_.shape[1],1,1,1,1)
            z_and_B_and_F_and_A, id_jac_z_b_f_a = torch.concatenate([outer, A], dim=1), torch.concatenate([outer_jac_z, A_jac_z], dim=1)

            # d_A: (z, B, F, A) -> (z, B, F, E)
            E_raw, E_jac_z_b_f_a = self.edge_attr_decoder(z_and_B_and_F_and_A, jacobian)
            E_raw_jac_z = torch.einsum('bijkvwz, bvwznm -> bijknm', E_jac_z_b_f_a, id_jac_z_b_f_a)
            tau_E_raw, tau_E_raw_jac_z = torch.concatenate([tau_edges, E_raw], dim=1),  torch.concatenate([tau_edges_jac_z, E_raw_jac_z], dim=1)
            E_, E_jac_tau_e_raw = self.gs_e(tau_E_raw, jacobian)
            E_jac_z = torch.einsum('bijkvwz, bvwznm -> bijknm', E_jac_tau_e_raw, tau_E_raw_jac_z)
            A_ = A_.tile(1, E_.shape[1], 1, 1)
            E = E_ * A_
            E_jac_z = E_jac_z * A_[..., None, None] + E[..., None, None] * A_jac_z.tile(1, E_.shape[1],1,1,1,1)

            return (F, B, A, E), (B_jac_z, F_jac_z, A_jac_z, E_jac_z)
        else:
            B_pos_logits = self.node_decoder(z)
            B_new = B_pos_logits

            z_and_B = torch.concatenate([z, B_new], dim=1)
            F_raw = self.node_decoder(z_and_B)
            F_new = F_raw # Add
            return F_new, B_new
