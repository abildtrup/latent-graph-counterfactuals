import torch
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F

from counterfactual_graph_generation.optimizers.riemannian_utility import riemannian_grad, euclidean_retraction

class RiemannianSGD(Optimizer):
    def __init__(self,
                 params,
                 model,
                 rgrad=riemannian_grad,
                 retraction=euclidean_retraction,
                 normalize: bool = True,
                 lr: float = 1e-3):

        defaults = dict(model=model,
                        rgrad=rgrad,
                        retraction=retraction,
                        normalize=normalize,
                        lr=lr)

        super(RiemannianSGD, self).__init__(params, defaults)

    def step(self, lr=None):
        loss = None
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if lr is None:
                    lr = group['lr']
                # compute riemannian gradient
                d_p = group['rgrad'](group['model'], p, d_p)
                if group['normalize']:
                    d_p = F.normalize(d_p, p=2.0, dim=1, eps=1e-12, out=None)
                # euclidian retraction
                group['retraction'](p, d_p, lr)
        return loss


"""
class RiemannianSGD(Optimizer):
    def __init__(self,
                 params,
                 model,
                 classifier,
                 rgrad=riemannian_grad,
                 retraction=euclidean_retraction,
                 normalize: bool = True,
                 lr: float = 1e-3):

        defaults = dict(model=model,
                        classifier=classifier,
                        rgrad=rgrad,
                        retraction=retraction,
                        normalize=normalize,
                        lr=lr)

        super(RiemannianSGD, self).__init__(params, defaults)

    def step(self, lr=None):
        loss = None
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if lr is None:
                    lr = group['lr']
                # compute riemannian gradient
                d_p = group['rgrad'](group['model'], group['classifier'], p, d_p)
                if group['normalize']:
                    d_p = F.normalize(d_p, p=2.0, dim=1, eps=1e-12, out=None)
                # euclidian retraction
                group['retraction'](p, d_p, lr)
        return loss

class RiemannianSGDwithMomentum(Optimizer):
    def __init__(
        self,
        params,
        model,
        classifier,
        rgrad=riemannian_grad,
        retraction=euclidean_retraction,
        normalize: bool = True,
        lr: float = 1e-3,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        defaults = dict(model=model,
                        classifier=classifier,
                        rgrad=rgrad,
                        retraction=retraction,
                        normalize=normalize,
                        lr=lr,
                        momentum=momentum,
                        dampening=dampening,
                        weight_decay=weight_decay,
                        nesterov=nesterov)

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening"
            )
        super(RiemannianSGDwithMomentum, self).__init__(params, defaults)

    def step(self):
        loss = None
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if lr is None:
                    lr = group['lr']

                # compute riemannian gradient
                d_p = group['rgrad'](group['model'], group['classifier'], p, d_p)
                if group['normalize']:
                    d_p = F.normalize(d_p, p=2.0, dim=1, eps=1e-12, out=None)
                # euclidian retraction
                group['retraction'](p, d_p, lr)

                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(
                            d_p
                        ).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                # apply momentum
                p.data.add_(d_p, alpha=-group["lr"])
                # apply weight decay
                if weight_decay != 0:
                    p.data.add_(weight_decay, alpha=-group["lr"])
        return loss
"""
