import torch
import torch.nn as nn
import stochman.nnj as nnj

import counterfactual_graph_generation.models.equivariant_linear_layers as eqll
from counterfactual_graph_generation.models.decoders import RiemannianGNNDecoder, GNNDecoder
from counterfactual_graph_generation.models.graphClassifier import RiemannianDenseGraphClassifier

class testModule(nn.Module):
    def __init__(self, in_, out_):
        super().__init__()
        self.linear = nnj.Linear(in_, out_)

    def forward(self, batch, jacobian=False):
        if jacobian:
            out, jac = self.linear(batch, jacobian=True)
            return out, jac
        else:
            return self.linear(batch, jacobian=False)

    def _jacobian_wrt_input_mult_left_vec(self, x, val, jac_in):
        return self.linear._jacobian_wrt_input_mult_left_vec(x, val, jac_in)


class testModuleSeq(nnj.AbstractJacobian, nn.Module):
    def __init__(self, in_, out_):
        super().__init__()
        self.linear = nnj.Sequential(testModule(2,3))

    def forward(self, batch):
        return self.linear(batch)

#model = nnj.Sequential(testModule(2,3), testModule(3,2))

# Parameters
n = 3
batch_size = 5
node_features = 6
edge_features = 7
latent_dim = 1
filter_channels = 4

# Data
z = torch.randn(batch_size, latent_dim, n)
F = torch.randn(batch_size, node_features, n)
B = torch.randn(batch_size, 1, n)
A = torch.randn(batch_size, 1, n, n)
E = torch.randn(batch_size, edge_features, n, n)

### VAE models
#model = RiemannianGNNDecoder(node_features=node_features, edge_features=edge_features, latent_dim=latent_dim, filter_channels=filter_channels, graph_size_limit=n)
#model_vanilla = GNNDecoder(node_features=node_features, edge_features=edge_features, latent_dim=latent_dim, filter_channels=filter_channels)
# Check normal forward pass
#model_vanilla(z, B, F, A)
# Check riemannian forward pass:
#model(z, B, F, A)
# Check riemannian forward pass jacobian True:
#(F_new, B_new, A_new, E_new), (B_jac_z, F_jac_z, A_jac_z, E_jac_z) = model(z, jacobian=True)

### Classifier models:
classifier = RiemannianDenseGraphClassifier(node_features=node_features, edge_features=edge_features, graph_size_limit=n, num_classes=2)
# Forward pass normal:
batch = (F, B, A, E, None)
classifier(batch)
# Compute jacobian:
out = classifier(batch, jacobian=True)

# Print whether the test is succesfull
print("succesfull test")
