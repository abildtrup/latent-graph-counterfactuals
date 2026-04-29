import torch

def euclidean_grad(p, d_p):
    return d_p

def euclidean_retraction(p, d_p, lr): # Update rule for rsgd (keep as is)
    """
    The (euclidean) update rule for rsgd. When the gradient d_p is tangent to a manifold, then this Euclidean update rule will
    approximate updating along the manifold, when it is decently smooth, and when the learning rate is small enough.

    p: The point which the gradient is computed with respect to, i.e. the input.
    d_p: The gradient computed with repect to p.
    lr: The learning rate

    """
    p.data.add_(d_p, alpha=-lr)

def riemannian_grad(model, p, d_p):
    """
    Return the gradient tangent induced by the riemannian pull-back metric.

    model: The stochman model to be considered. The model must implement a method for computing jacobians.
    p: The point which the gradient is computed with respect to, i.e. the input.
    d_p: The gradient computed with repect to p.

    """
    J = model.jacobians(p)
    G = pullback_metric(J)
    inv_G = torch.inverse(G)
    d_p = torch.matmul(inv_G, d_p.unsqueeze(-1)).squeeze(-1)
    return d_p

def pullback_metric(J, A=None):
    """
    Returns a square metric specifying the riemannian pull-back metric

    J: Jacobian.
    A: A positive definite square matrix specifying a rimannian metric.
    """

    J_T = J.transpose(1, 2)
    A_J = J
    if A != None:
        A_J = torch.matmul(A, J)
    return torch.matmul(J_T, A_J)


"""
def riemannian_grad(model, classifier, p, d_p): # used for computing the riemannian gradient -
    J_mu, J_sigma = model.jacobians(p)
    if classifier is not None:
        x_mu, x_sigma = model.decode(p)
        A = classifier.metric(x_mu) # If classifier is none then this is just the identity
        #A = 0.0001*A
        #A = 99999*A
        # I = torch.eye(A.shape[-1])
        # I = I.reshape((1, A.shape[1], A.shape[2]))
        # A = I.repeat(A.shape[0], 1, 1)
        # print(A.shape, A.max(), A.min(), J_sigma.max())
        G = counterfactual_metric(J_mu, J_sigma, A)
        #breakpoint()
    else:
        G = model.metric(p)
    inv_G = torch.inverse(G)
    d_p = torch.matmul(inv_G, d_p.unsqueeze(-1)).squeeze(-1)
    return d_p

def counterfactual_metric(J_mu, J_sigma, A):
    G_mu = torch.matmul(J_mu.transpose(1, 2), torch.matmul(A, J_mu))
    G_sigma = torch.matmul(J_sigma.transpose(1, 2), torch.matmul(A, J_sigma))
    G = G_mu + G_sigma
    return G

def counterfactual_log_volume(J_mu, J_sigma, A):
    G_mu = torch.matmul(J_mu.transpose(1, 2), torch.matmul(A, J_mu))
    G_sigma = torch.matmul(J_sigma.transpose(1, 2), torch.matmul(A, J_sigma))
    G = G_mu + G_sigma
    log_vol = 0.5 * G.slogdet()[1]
    return log_vol
    """
