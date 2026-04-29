import torch

from counterfactual_graph_generation.models.encoders import GNNEncoder
from counterfactual_graph_generation.models.decoders import GNNDecoder, RiemannianGNNDecoder


class PEGVAE(torch.nn.Module):
    def __init__(self, encoder_config, decoder_config, riemannian=False):
        super(PEGVAE, self).__init__()

        self.encoder = GNNEncoder(**encoder_config)
        if riemannian:
            self.decoder = RiemannianGNNDecoder(**decoder_config)
        else:
            self.decoder = GNNDecoder(**decoder_config)
        self.prior = None

    def forward(self, F, B, A, E):
        mu_e, log_var_e = self.encoder.encode(F, B, A, E)
        z = self.encoder.sample(mu=mu_e, log_var=log_var_e)
        return self.decoder(z, B, F, A), (mu_e, log_var_e)

    def encode(self, F, B, A, E):
        mu_e, log_var_e = self.encoder.encode(F, B, A, E)
        return mu_e, log_var_e

    def sample(self, batch_size=16):
        #z = self.prior.sample(batch_size=batch_size)
        z = torch.randn(batch_size, 1, self.decoder.node_features)
        return self.decoder.sample(z)


class AdaptedEncoder(torch.nn.Module):
    def __init__(self, encoder, encoder_adapter):
        super().__init__()
        self.encoder = encoder
        self.encoder_adapter = encoder_adapter

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
        F_new, B_new, A_new, E_new = self.encoder_adapter(F, B, A, E)
        return self.encoder(F_new, B_new, A_new, E_new)

    def log_prob(self, x=None, mu_e=None, log_var_e=None, z=None):
        # Should return the variational posterior q(z|x)
        return

    def sample(self, mu, log_var):
        return self.reparameterization(mu, log_var)


class AdaptedDecoder(torch.nn.Module):
    def __init__(self, decoder, decoder_adapter):
        super().__init__()
        self.unadapted_decoder = decoder
        self.decoder_adapter = decoder_adapter
        self.node_features = decoder_adapter.node_features_out
        self.edge_features = decoder_adapter.edge_features_out

    def forward(self, z, B, F, A):
        F_new, B_new, A_new, E_new = self.unadapted_decoder(z, B, F, A)
        F_new, B_new, A_new, E_new = self.decoder_adapter(F_new, B_new, A_new, E_new)
        return F_new, B_new, A_new, E_new

    def decode(self, z):
        # TODO after loss check
        return

    def decode_discrete_graph(self, z, tau=1):
        # TODO after loss check
        return

    def sample_B(self, z):
        # TODO after loss check
        return

    def sample_F(self, z, B):
        # TODO after loss check
        return

    def sample_A(self, z, B, F):
        # TODO after loss check
        return

    def sample_E(self, z, B, F, A):
        # TODO after loss check
        return

    def sample(self, z):
        # TODO after loss check
        return


class AdaptedPEGVAE(torch.nn.Module):
    def __init__(self, model, encoder_adapter, decoder_adapter):
        super().__init__()
        self.encoder = AdaptedEncoder(model.encoder, encoder_adapter)
        self.decoder = AdaptedDecoder(model.decoder, decoder_adapter)

    def forward(self, F, B, A, E):
        mu_e, log_var_e = self.encoder.encode(F, B, A, E)
        z = self.encoder.sample(mu=mu_e, log_var=log_var_e)
        return self.decoder(z, B, F, A), (mu_e, log_var_e)

    def sample(self, batch_size=16):
        #z = self.prior.sample(batch_size=batch_size)
        z = torch.randn(batch_size, 1, self.decoder.node_features)
        return self.decoder.sample(z)
