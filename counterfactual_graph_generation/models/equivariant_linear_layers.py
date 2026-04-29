
import math

import torch
import stochman.nnj as nnj
from torch import Tensor, nn
from torch_geometric.nn import inits
from torch.autograd.functional import jacobian as func_jacobian
'''

Equivariant linear layer (graph-to-graph)

'''
class EquiLinear2to2(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(EquiLinear2to2, self).__init__()
        '''
        B: Batchsize
        N: Number of nodes in graph
        b(*): Bell number
        W: weights. These should have dimension b(4) x d x d'
        '''
        self.basis_dimension = 15
        self.inv_basis_dim = 1/self.basis_dimension
        self.in_channels = in_channels
        self.out_channels= out_channels
        self.weight = nn.Parameter(torch.Tensor(self.basis_dimension, out_channels, in_channels)) # 15 x d' x d

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        inits.kaiming_uniform(self.weight, fan=self.in_channels, a=math.sqrt(5))
        if self.bias is None:
            pass
        else:
            inits.uniform(self.in_channels, self.bias)

    def forward(self, E): # A: B x F_e x N x N
        ops_E_to_E = self.ops_2_to_2(E) # B x 15 x d x N x N
        output = torch.einsum('bmd, qbdwe -> qmwe', self.weight, ops_E_to_E) # B x d' x N x N
        return self.inv_basis_dim * output

    def ops_2_to_2(self, E: Tensor): # B x F_e x N x N
        ops = [None]*15
        dim = E.shape[-2]

        # Edit: Changed sum to mean to control output size.
        diag_part = E.diagonal(dim1=-2, dim2=-1) # B x F_e x N
        sum_diag_part = diag_part.mean(dim=2, keepdim=True) # B x F_e x N
        sum_of_rows = E.mean(dim=3) # B x F_e x N
        sum_of_cols = E.mean(dim=2) # B x F_e x N
        sum_all = E.mean(dim=[2,3]) # B x F_e

        # op1 - (1234) - extract diag
        ops[0] = (torch.diag_embed(diag_part)) # B x F_e x N x N
        ops[1] = (torch.diag_embed(torch.tile(sum_diag_part, dims=[1, 1, dim]))) # B x F_e x N x N
        ops[2] = (torch.diag_embed(sum_of_rows)) # B x F_e x N x N
        ops[3] = (torch.diag_embed(sum_of_cols)) # B x F_e x N x N
        ops[4] = (torch.diag_embed(torch.tile(sum_all.unsqueeze(dim=2), dims=[1,1,dim]))) # B x F_e x N x N
        ops[5] = (sum_of_cols.unsqueeze(dim=3).tile(dims=[1,1,1,dim])) # B x F_e x N x N
        ops[6] = (sum_of_rows.unsqueeze(dim=3).tile(dims=[1,1,1,dim])) # B x F_e x N x N
        ops[7] = (sum_of_cols.unsqueeze(dim=2).tile(dims=[1,1,dim,1])) # B x F_e x N x N
        ops[8] = (sum_of_rows.unsqueeze(dim=2).tile(dims=[1,1,dim,1])) # B x F_e x N x N
        ops[9] = (E) # B x F_e x N x N
        ops[10] = (E.transpose(dim0=2, dim1=3))  # B x F_e x N x N
        ops[11] = (diag_part.unsqueeze(dim=3).tile(dims=[1,1,1,dim])) # B x F_e x N x N
        ops[12] = (diag_part.unsqueeze(dim=2).tile(dims=[1,1,dim,1])) # B x F_e x N x N
        ops[13] = (sum_diag_part.unsqueeze(dim=3).tile(dims=[1,1,dim,dim])) # B x F_e x N x N
        ops[14] = (sum_all.unsqueeze(dim=2).unsqueeze(dim=3).tile(dims=[1,1,dim,dim])) # B x F_e x N x N

        return torch.stack(ops, dim=1) # B x 15 x  d' x N x N


class nnjEquiLinear2to2(EquiLinear2to2):
    def __init__(self, in_channels, out_channels, bias=True):
        super(nnjEquiLinear2to2, self).__init__(in_channels, out_channels, bias=bias)

    def forward(self, E, jacobian=False): # A: B x F_e x N x N
        ops_E_to_E = self.ops_2_to_2(E) # B x 15 x d x N x N
        output = torch.einsum('bmd, qbdwe -> qmwe', self.weight, ops_E_to_E) # B x d' x N x N
        output = self.inv_basis_dim * output
        if jacobian:
            jac = func_jacobian(self.forward, E, create_graph=False)
            return output, jac
        return output

    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor):
        # jac_in: b x channels_in_in x z_in_in x z_in_in x channels_in x z_in x z_in
        jac_wrt_input = func_jacobian(self.forward, x, create_graph=False).diagonal(dim1=0, dim2=4).movedim(-1, 0) # b x channels_out x z_o x z_o x channels_in x z_in x z_in
        jac_out = torch.einsum('bijknmp, bvwznmp -> bijkvwz', jac_wrt_input, jac_in) # b x channels_out x z_o x z_o x channels_in_in x z_in_in x z_in_in
        return jac_out


'''

Equivariant linear layer (nodes-to-graph)

'''
class EquiLinear1to2(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(EquiLinear1to2, self).__init__()
        '''
        B: Batchsize
        N: Number of nodes in graph
        b(*): Bell number
        W: weights. These should have dimension b x d x d'
        '''
        self.basis_dimension = 5
        self.inv_basis_dim = 1/self.basis_dimension
        self.in_channels = in_channels
        self.out_channels= out_channels
        self.weight = nn.Parameter(torch.Tensor(self.basis_dimension, out_channels, in_channels)) # 5 x d' x d

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    def reset_parameters(self) -> None:
        inits.kaiming_uniform(self.weight, fan=self.in_channels, a=math.sqrt(5))
        if self.bias is None:
            pass
        else:
            inits.uniform(self.in_channels, self.bias)

    def forward(self, F): # F: B x F_n x N
        ops_F_to_E = self.ops_1_to_2(F) # B x 5 x d x N x N
        output = torch.einsum('bmd, qbdwe -> qmwe', self.weight, ops_F_to_E) # B x N x N x d'
        return self.inv_basis_dim * output

    def ops_1_to_2(self, F: Tensor): # F: B x F_n x N
        ops = [None]*5
        dim = F.shape[-1]

        # Usefull variables:
        # Edit: Changed sum to mean to control output size.
        sum_all = torch.mean(F, dim=2, keepdim=True) # B x F_n x 1

        ### Operations ###
        ops[0] = (torch.diag_embed(F, offset=0, dim1=-2, dim2=-1)) # B x F_n x N x N
        ops[1] = (torch.diag_embed(torch.tile(sum_all, dims=[1,1,dim]), offset=0, dim1=-2, dim2=-1)) # B x F_n x N x N
        ops[2] = (torch.tile(torch.unsqueeze(F, dim=2), dims=[1, 1, dim, 1])) # B x F_n x N x N
        ops[3] = (torch.tile(torch.unsqueeze(F, dim=3), dims=[1, 1, 1, dim])) # B x F_n x N x N
        ops[4] = (torch.tile(torch.unsqueeze(sum_all, dim=3), dims=[1, 1, dim, dim])) # B x F_n x N x N

        return torch.stack(ops, dim=1) # B x 5 x N x N x F_n


class nnjEquiLinear1to2(EquiLinear1to2):
    def __init__(self, in_channels, out_channels, bias=True):
        super(nnjEquiLinear1to2, self).__init__(in_channels, out_channels, bias=bias)

    def forward(self, F, jacobian=False): # F: B x F_n x N
        ops_F_to_E = self.ops_1_to_2(F) # B x 5 x d x N x N
        output = torch.einsum('bmd, qbdwe -> qmwe', self.weight, ops_F_to_E) # B x N x N x d'
        output = self.inv_basis_dim * output
        if jacobian:
            jac = func_jacobian(self.forward, F, create_graph=False)
            return output, jac
        return output

    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor):
        # jac_in: b x channels_in_in x z_in_in x channels_in x z_in
        jac_wrt_input = func_jacobian(self.forward, x, create_graph=False).diagonal(dim1=0, dim2=4).movedim(-1, 0) # b x channels_out x z_o x z_o x channels_in x z_in
        jac_out = torch.einsum('bijknm, bvwnm -> bijkvw', jac_wrt_input, jac_in) # b x channels_out x z_o x z_o x channels_in_in x z_in_in
        return jac_out


'''

Equivariant linear layer (graph-to-nodes)

'''
class EquiLinear2to1(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(EquiLinear2to1, self).__init__()
        '''
        B: Batchsize
        N: Number of nodes in graph
        b(*): Bell number
        W: weights. These should have dimension b x d x d'
        '''
        self.basis_dimension = 5
        self.inv_basis_dim = 1/self.basis_dimension
        self.in_channels = in_channels
        self.out_channels= out_channels
        self.weight = nn.Parameter(torch.Tensor(self.basis_dimension, out_channels, in_channels)) # 5 x d x d'

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        inits.kaiming_uniform(self.weight, fan=self.in_channels, a=math.sqrt(5))
        if self.bias is None:
            pass
        else:
            inits.uniform(self.in_channels, self.bias)

    def forward(self, E): # E: B x F_e x N x N
        ops_E_to_F = self.ops_2_to_1(E) # B x 5 x d' x N
        output = torch.einsum('bmd, qbdw -> qmw', self.weight, ops_E_to_F) # B x d x N
        return self.inv_basis_dim * output

    def ops_2_to_1(self, E: Tensor): # E: B x F_e x N x N
        ops = [None]*5
        dim = E.shape[-1]

        # Usefull variables:
        # Edit: Changed sum to mean to control output size.
        diag_part = E.diagonal(dim1=-2, dim2=-1) # B x F_e x N
        sum_diag_part = diag_part.mean(dim=2, keepdim=True) # B x F_e x 1

        ### Operations ###
        ops[0] = (diag_part) # B x N x F_e
        ops[1] = (torch.tile(sum_diag_part, dims=[1, 1, dim])) # B x F_e x N
        ops[2] = (E.sum(dim=2)) # B x F_e x N
        ops[3] = (E.sum(dim=3)) # B x F_e x N
        ops[4] = (torch.tile(E.sum(dim=[2,3]).unsqueeze(dim=2), dims=[1, 1, dim])) # B x F_e x N

        return torch.stack(ops, dim=1) # B x 5 x F_e x N


class nnjEquiLinear2to1(EquiLinear2to1):
    def __init__(self, in_channels, out_channels, bias=True):
        super(nnjEquiLinear2to1, self).__init__(in_channels, out_channels, bias=bias)

    def forward(self, E, jacobian=False): # E: B x F_e x N x N
        ops_E_to_F = self.ops_2_to_1(E) # B x 5 x d' x N
        output = torch.einsum('bmd, qbdw -> qmw', self.weight, ops_E_to_F) # B x d x N
        output = self.inv_basis_dim * output
        if jacobian:
            jac = func_jacobian(self.forward, E, create_graph=False)
            return output, jac
        return output

    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor):
        # jac_in: b x channels_in_in x z_in_in x z_in_in x channels_in x z_in x z_in
        jac_wrt_input = func_jacobian(self.forward, x, create_graph=False).diagonal(dim1=0, dim2=3).movedim(-1, 0) # b x channels_out x z_o x channels_in x z_in x z_in
        jac_out = torch.einsum('bijnmp, bvwznmp -> bijvwz', jac_wrt_input, jac_in) # b x channels_out x z_o x channels_in_in x z_in_in x z_in_in
        return jac_out

'''

Equivariant linear layer (nodes-to-nodes)

'''
class EquiLinear1to1(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(EquiLinear1to1, self).__init__()
        '''
        B: Batchsize
        N: Number of nodes in graph
        b(*): Bell number
        W: weights. These should have dimension b x d x d'
        '''
        self.basis_dimension = 2
        self.inv_basis_dim = 1/self.basis_dimension
        self.in_channels = in_channels
        self.out_channels= out_channels
        self.weight = nn.Parameter(torch.Tensor(self.basis_dimension, out_channels, in_channels)) # 2 x d x d'

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        inits.kaiming_uniform(self.weight, fan=self.in_channels, a=math.sqrt(5))
        if self.bias is None:
            pass
        else:
            inits.uniform(self.in_channels, self.bias)

    def forward(self, F): # F: B x N x F_n
        ops_F_to_F = self.ops_1_to_1(F) # B x 2 x d' x N
        output = torch.einsum('bmd, qbdw -> qmw', self.weight, ops_F_to_F) # B x d x N
        return self.inv_basis_dim * output

    def ops_1_to_1(self, F: Tensor): # F: B x F_n x N
        ops = [None]*2
        dim = F.shape[-1]

        # Usefull variables:
        sum_all = F.sum(dim=2, keepdim=True) # B x 1 x F_n

        ### Operations ###
        ops[0] = F
        ops[1] = torch.tile(sum_all, [1, 1, dim]) # B x F_n x N
        return torch.stack(ops, dim=1) # B x 5 x F_e x N


class nnjEquiLinear1to1(EquiLinear1to1):
    def __init__(self, in_channels, out_channels, bias=True):
        super(nnjEquiLinear1to1, self).__init__(in_channels, out_channels, bias=bias)

    def forward(self, F, jacobian=False): # F: B x N x F_n
        ops_F_to_F = self.ops_1_to_1(F) # B x 2 x d' x N
        output = torch.einsum('bmd, qbdw -> qmw', self.weight, ops_F_to_F) # B x d x N
        output = self.inv_basis_dim * output
        if jacobian:
            jac = func_jacobian(self.forward, F, create_graph=False)
            return output, jac
        return output

    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor):
        # jac_in: b x channels_in_in x z_in_in x channels_in x z_in
        jac_wrt_input = func_jacobian(self.forward, x, create_graph=False).diagonal(dim1=0, dim2=3).movedim(-1, 0) # b x channels_out x z_o x channels_in x z_in
        jac_out = torch.einsum('bijnm, bnmvw -> bijvw', jac_wrt_input, jac_in) # b x channels_out x z_o x channels_in_in x z_in_in
        return jac_out

### Outer product decoder:
class OuterProductLayer(nn.Module):
    def __init__(self):
        super(OuterProductLayer, self).__init__()

    def forward(self, F, jacobian=False): # F: B x F_e x N
        F_ = F.squeeze()
        E = torch.einsum('bcj, bci -> bcji', F_, F_)
        if jacobian:
            jac = func_jacobian(self.forward, F, create_graph=False)
            return E, jac
        return E

    def jacobian(self, x):
        return self.forward(x, jacobian=True)

    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor):
        # jac_in: b x channels_in_in x z_in_in x channels_in x z_in
        #jac_wrt_input = func_jacobian(self.jacobian, x, create_graph=False).diagonal(dim1=0, dim2=3).movedim(-1, 0) # b x channels_out x z_o x z_o x channels_in x z_in
        jac_wrt_input = self.jacobian(x)[1].diagonal(dim1=0, dim2=4).movedim(-1, 0).squeeze()
        if jac_in.squeeze().dim() == 5:
            jac_out = torch.einsum('bijknm, bnmvw -> bijkvw', jac_wrt_input, jac_in.squeeze())
        elif jac_in.squeeze().dim() == 6:
            jac_out = torch.einsum('bijknm, bnmvwz -> bijkvwz', jac_wrt_input, jac_in.squeeze()) # b x channels_out x z_o x z_o x channels_in_in x z_in_in
        return jac_out
