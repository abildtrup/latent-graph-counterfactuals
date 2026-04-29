import torch
from torch import nn

class Adapter(torch.nn.Module):
    def __init__(self, config, riemannian=False):
        super().__init__()
        self.bool_features_in = config.bool_features_in
        self.node_features_in = config.node_features_in
        self.adj_feature_in = config.adj_features_in
        self.edge_features_in = config.edge_features_in
        self.bool_features_out = config.bool_features_out
        self.node_features_out = config.node_features_out
        self.adj_feature_out = config.adj_features_out
        self.edge_features_out = config.edge_features_out
        latent_filter_channels = config.latent_filter_channels
        self.latent_filter_channels = latent_filter_channels

        # Node feature layer
        self.layer_bool_node = nn.Sequential(
            nn.Conv1d(in_channels=self.bool_features_in, out_channels=latent_filter_channels, kernel_size=1), nn.LeakyReLU(),
            nn.Conv1d(in_channels=latent_filter_channels, out_channels=latent_filter_channels, kernel_size=1), nn.LeakyReLU(),
            nn.Conv1d(in_channels=latent_filter_channels, out_channels= self.bool_features_out, kernel_size=1),
        )

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=self.node_features_in, out_channels=latent_filter_channels, kernel_size=1), nn.LeakyReLU(),
            nn.Conv1d(in_channels=latent_filter_channels, out_channels=latent_filter_channels, kernel_size=1), nn.LeakyReLU(),
            nn.Conv1d(in_channels=latent_filter_channels, out_channels=self.node_features_out, kernel_size=1),
        )

        # Edge feature layer
        self.layer_bool_edge = nn.Sequential(
            nn.Conv2d(in_channels=self.adj_feature_in, out_channels=latent_filter_channels, kernel_size=1), nn.LeakyReLU(),
            nn.Conv2d(in_channels=latent_filter_channels, out_channels=latent_filter_channels, kernel_size=1), nn.LeakyReLU(),
            nn.Conv2d(in_channels=latent_filter_channels, out_channels=self.adj_feature_out, kernel_size=1),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=self.edge_features_in, out_channels=latent_filter_channels, kernel_size=1), nn.LeakyReLU(),
            nn.Conv2d(in_channels=latent_filter_channels, out_channels=latent_filter_channels, kernel_size=1), nn.LeakyReLU(),
            nn.Conv2d(in_channels=latent_filter_channels, out_channels=self.edge_features_out, kernel_size=1),
        )

    def forward(self, F, B, A, E):
        B_new = self.layer_bool_node(B)
        F_new = self.layer1(F)
        A_new = self.layer_bool_edge(A)
        E_new = self.layer2(E)
        return F_new, B_new, A_new, E_new
