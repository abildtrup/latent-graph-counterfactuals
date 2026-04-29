import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.autograd.functional import jacobian as func_jacobian
import stochman.nnj as nnj
from torch import Tensor

# Gumbel softmax for the B tensor
class WeightedGumbelSoftmaxNodesBool(nnj.AbstractJacobian, nn.Module):
    def __init__(self,) -> None:
        super(WeightedGumbelSoftmaxNodesBool, self).__init__()

    def forward(self, batch, jacobian=False):
        tau = batch[:,0:1,...]
        batch = batch[:,1:,...]
        B_neg_logits = torch.zeros_like(batch)
        B_logits = torch.cat([B_neg_logits, batch], dim=1)
        B = Fn.gumbel_softmax(B_logits, tau=tau, hard=True, dim=1)[:,1,:].unsqueeze(dim=1)
        return B

    def _jacobian(self, x, val):
        jac = func_jacobian(self.forward, x, create_graph=False).diagonal(dim1=0, dim2=3).movedim(-1, 0)
        return jac

    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        jac_wrt_input = self._jacobian(x, val)
        jac_out = torch.einsum('bijnm, bnmvw -> bijvw', jac_wrt_input, jac_in)
        return jac_out

# Gumbel softmax for the F tensor
class WeightedGumbelSoftmaxNodesAttr(nnj.AbstractJacobian, nn.Module):
    def __init__(self,) -> None:
        super(WeightedGumbelSoftmaxNodesAttr, self).__init__()

    def forward(self, batch):
        tau = batch[:,0:1,...]
        batch = batch[:,1:,...]
        F = Fn.gumbel_softmax(batch, tau=tau, hard=True, dim=1)
        return F

    def _jacobian(self, x, val):
        jac = func_jacobian(self.forward, x, create_graph=False).diagonal(dim1=0, dim2=3).movedim(-1, 0)
        return jac

    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        jac_wrt_input = self._jacobian(x, val)
        jac_out = torch.einsum('bijnm, bnmvw -> bijvw', jac_wrt_input, jac_in)
        return jac_out

# Gumbel softmax for the A tensor
class WeightedGumbelSoftmaxEdgesBool(nnj.AbstractJacobian, nn.Module):
    def __init__(self,) -> None:
        super(WeightedGumbelSoftmaxEdgesBool, self).__init__()

    def forward(self, batch):
        tau = batch[:,0:1,...]
        batch = batch[:,1:,...]
        A_pos_logits = batch
        A_neg_logits = torch.zeros_like(A_pos_logits)
        A_logits = torch.cat([A_neg_logits, A_pos_logits], dim=1)
        A_one_hot = Fn.gumbel_softmax(A_logits, tau=tau, hard=True, dim=1)[:,1,:,:].unsqueeze(dim=1)
        A_tril = A_one_hot.tril(-1) # N x 1 x V x V
        A = (A_tril + A_tril.permute(0, 1, 3, 2))
        return A

    def _jacobian(self, x, val):
        jac = func_jacobian(self.forward, x, create_graph=False).diagonal(dim1=0, dim2=4).movedim(-1, 0)
        return jac

    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        jac_wrt_input = self._jacobian(x, val)
        jac_out = torch.einsum('bijnm, bnmvw -> bijvw', jac_wrt_input, jac_in)
        return jac_out

# Gumbel softmax for the E tensor
class WeightedGumbelSoftmaxEdgesAttr(nnj.AbstractJacobian, nn.Module):
    def __init__(self,) -> None:
        super(WeightedGumbelSoftmaxEdgesAttr, self).__init__()

    def forward(self, batch):
        tau = batch[:,0:1,...]
        batch = batch[:,1:,...]
        E_tril = Fn.gumbel_softmax(batch, tau=tau, hard=True, dim=1).tril(-1)
        E = (E_tril + E_tril.permute(0, 1, 3, 2))
        return E

    def _jacobian(self, x, val):
        jac = func_jacobian(self.forward, x, create_graph=False).diagonal(dim1=0, dim2=4).movedim(-1, 0)
        return jac

    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        jac_wrt_input = self._jacobian(x, val)
        jac_out = torch.einsum('bijnm, bnmvw -> bijvw', jac_wrt_input, jac_in)
        return jac_out
