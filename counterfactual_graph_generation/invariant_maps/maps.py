import numpy as np


def invariant_channel_sort(z, dim=1):
    # Assert that the shape is B x C x N
    z_sort = np.sort(z, axis=2)
    z_permutation = np.argsort(z, axis=2)
    z_inv_permutation = np.argsort(z_permutation, axis=2)
    return z_sort, z_permutation, z_inv_permutation

# Suggestions:
# - Implement Max pooling,
# - Implememnt Sum pooling,
# - Implement prealigning to some random graph
