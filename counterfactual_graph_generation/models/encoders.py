import torch
from torch import nn

import counterfactual_graph_generation.models.equivariant_linear_layers as eqll

class GNNEncoder(torch.nn.Module):
    def __init__(self, node_features: int, edge_features: int, latent_dim: int, filter_channels: int):
        super(GNNEncoder, self).__init__()

        self.filter_channels = filter_channels
        # First layer only considers node feature information and maps to GRAPH
        self.layer1 = nn.Sequential(
            eqll.EquiLinear1to2(in_channels= 1 + node_features, out_channels=self.filter_channels), nn.LeakyReLU(),
            nn.Conv2d(in_channels=self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nn.LeakyReLU(), nn.InstanceNorm2d(num_features=self.filter_channels),
            nn.Conv2d(in_channels=self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nn.LeakyReLU(), nn.InstanceNorm2d(num_features=self.filter_channels),
        )

        # One equivariant second layer
        self.layer2 = nn.Sequential(
            eqll.EquiLinear2to2(in_channels= self.filter_channels + edge_features, out_channels=self.filter_channels), nn.LeakyReLU(),
            nn.Conv2d(in_channels=self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nn.LeakyReLU(), nn.InstanceNorm2d(num_features=self.filter_channels),
            nn.Conv2d(in_channels=self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nn.LeakyReLU(), nn.InstanceNorm2d(num_features=self.filter_channels),
        )

        self.layer3 = nn.Sequential(
            eqll.EquiLinear2to2(in_channels= self.filter_channels, out_channels=self.filter_channels), nn.LeakyReLU(),
            nn.Conv2d(in_channels=self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nn.LeakyReLU(), nn.InstanceNorm2d(num_features=self.filter_channels),
            nn.Conv2d(in_channels=self.filter_channels, out_channels=self.filter_channels, kernel_size=1), nn.LeakyReLU(), nn.InstanceNorm2d(num_features=self.filter_channels),
        )

        # Last layer shares equivariant layer but has different convolutions
        self.mu_mlp = nn.Sequential(nn.Sequential(eqll.EquiLinear2to1(in_channels=self.filter_channels, out_channels=self.filter_channels), nn.LeakyReLU()), nn.Conv1d(in_channels=self.filter_channels, out_channels=latent_dim, kernel_size=1))
        self.log_var_mlp = nn.Sequential(nn.Sequential(eqll.EquiLinear2to1(in_channels=self.filter_channels, out_channels=self.filter_channels), nn.LeakyReLU()), nn.Conv1d(in_channels=self.filter_channels, out_channels=latent_dim, kernel_size=1))


    @staticmethod
    def reparameterization(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, F, B, A, E):
         """
         F: The node feature matrix, N x C x V
         E: The edge feature matrix, N x C x V x V
         B: A boolean matrix indicating which nodes are padded and which are not, N x 1 x V
         """
         return self.encode(F, B, A, E)

    def encode(self, F, B, A, E):
        B_and_F = torch.cat([F, B], dim=1)
        out = self.layer1(B_and_F)
        out = torch.concatenate([out, E], dim=1)
        out = self.layer2(out)
        out = self.layer3(out)
        mu = self.mu_mlp(out)
        log_var = self.log_var_mlp(out)
        return mu, log_var

    def log_prob(self, x=None, mu_e=None, log_var_e=None, z=None):
        # Should return the variational posterior q(z|x)
        return

    def sample(self, mu, log_var):
        return self.reparameterization(mu, log_var)
