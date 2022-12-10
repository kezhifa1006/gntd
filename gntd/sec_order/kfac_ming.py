import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
import torch.distributed as dist

class KFAC(Optimizer):

    def __init__(self, net, damping, sua=False, pi=False, update_freq=1,
                 alpha=1.0, constraint_norm=False):
        """ K-FAC Preconditionner for Linear and Conv2d layers.

        Computes the K-FAC of the second moment of the gradients.
        It works for Linear and Conv2d layers and silently skip other layers.

        Args:
            net (torch.nn.Module): Network to precondition.
            damping (float): Tikhonov regularization parameter for the inverses.
            sua (bool): Applies SUA approximation.
            pi (bool): Computes pi correction for Tikhonov regularization.
            update_freq (int): Perform inverses every update_freq updates.
            alpha (float): Running average parameter (if == 1, no r. ave.).
            constraint_norm (bool): Scale the gradients by the squared
                fisher norm.
        """
        self.damping = damping
        self.sua = sua
        self.pi = pi
        self.update_freq = update_freq
        self.alpha = alpha
        self.constraint_norm = constraint_norm
        self.params = []
        self.iteration_counter = 0
        for mod in net.modules():
            mod_class = mod.__class__.__name__
            if mod_class in ['LinearEx', 'Conv2dEx']:
                params = [mod.weight]
                if mod.bias is not None:
                    params.append(mod.bias)
                d = {'params': params, 'mod': mod, 'layer_type': mod_class}
                self.params.append(d)
        super(KFAC, self).__init__(self.params, {})

    def compute_all_covs(self):
        isdist = _is_dist()
        if isdist:
            world_size = dist.get_world_size()
            this_rank = torch.distributed.get_rank()
        async_handles = []

        for group in self.param_groups:
            mod = group['mod']
            state = self.state[group['params'][0]]
            x = mod.last_input.detach()
            # loss function is averaged per batch, so multiply it with batch size per process
            gy = mod.last_output.grad.detach() * mod.last_output.grad.size(0)
            self._compute_covs(group, state, x, gy)
            if isdist:
                def all_avg(key):
                    state[key] = state[key].contiguous() / world_size
                    handle = dist.all_reduce(state[key], dist.ReduceOp.SUM, async_op=True)
                    async_handles.append(handle)
                all_avg('xxt')
                all_avg('ggt')
        for handle in async_handles:
            handle.wait()


    def step(self):
        """Performs one step of preconditioning."""
        fisher_norm = 0.
        for i, group in enumerate(self.param_groups):
            state = self.state[group['params'][0]]
            # Update inverses
            if self.iteration_counter % self.update_freq == 0:
                ixxt, iggt = self._inv_covs(state['xxt'], state['ggt'])
                state['ixxt'] = ixxt.contiguous()
                state['iggt'] = iggt.contiguous()
        # if update_params:
        for group in self.param_groups:
            # Getting parameters
            if len(group['params']) == 2:
                weight, bias = group['params']
            else:
                weight = group['params'][0]
                bias = None
            state = self.state[weight]
            # Preconditionning
            gw, gb = self._precond(weight, bias, group, state)
            # Updating gradients
            if self.constraint_norm:
                fisher_norm += (weight.grad * gw).sum()
            weight.grad.data = gw
            if bias is not None:
                if self.constraint_norm:
                    fisher_norm += (bias.grad * gb).sum()
                bias.grad.data = gb

        # Eventually scale the norm of the gradients
        if self.constraint_norm:
            scale = (1. / fisher_norm) ** 0.5
            for group in self.param_groups:
                for param in group['params']:
                    param.grad.data *= scale
        self.iteration_counter += 1

    def _precond(self, weight, bias, group, state):
        """Applies preconditioning."""
        if group['layer_type'] == 'Conv2dEx' and self.sua:
            return self._precond_sua(weight, bias, group, state)
        ixxt = state['ixxt']
        iggt = state['iggt']
        g = weight.grad.detach()
        s = g.shape
        if group['layer_type'] == 'Conv2dEx':
            g = g.contiguous().view(s[0], s[1]*s[2]*s[3])
        if bias is not None:
            gb = bias.grad.detach()
            g = torch.cat([g, gb.view(gb.shape[0], 1)], dim=1)
        g = torch.mm(torch.mm(iggt, g), ixxt)
        if bias is not None:
            gb = g[:, -1].contiguous().view(*bias.shape)
            g = g[:, :-1]
        else:
            gb = None
        g = g.contiguous().view(*s)
        return g, gb

    def _precond_sua(self, weight, bias, group, state):
        """Preconditioning for KFAC SUA."""
        ixxt = state['ixxt']
        iggt = state['iggt']
        g = weight.grad.detach()
        s = g.shape
        mod = group['mod']
        g = g.permute(1, 0, 2, 3).contiguous()
        if bias is not None:
            gb = bias.grad.view(1, -1, 1, 1).expand(1, -1, s[2], s[3])
            g = torch.cat([g, gb], dim=0)
        g = torch.mm(ixxt, g.contiguous().view(-1, s[0]*s[2]*s[3]))
        g = g.view(-1, s[0], s[2], s[3]).permute(1, 0, 2, 3).contiguous()
        g = torch.mm(iggt, g.view(s[0], -1)).view(s[0], -1, s[2], s[3])
        if bias is not None:
            gb = g[:, -1, s[2]//2, s[3]//2]
            g = g[:, :-1]
        else:
            gb = None
        return g, gb

    def _compute_covs(self, group, state, x, gy):
        """Computes the covariances."""
        mod = group['mod']

        # Computation of ggt
        if group['layer_type'] == 'Conv2dEx':
            gy = gy.detach().permute(1, 0, 2, 3)
            num_locations = gy.size(2) * gy.size(3)
            gy = gy.contiguous().view(gy.size(0), -1)
        else:
            gy = gy.detach().t()
            num_locations = 1
        normalizer = num_locations**0.5
        if self.iteration_counter == 0:
            state['ggt'] = torch.mm(gy, gy.t()) * (normalizer / gy.size(1))
        else:
            state['ggt'].addmm_(mat1=gy, mat2=gy.t(),
                                beta=(1. - self.alpha),
                                alpha=self.alpha * normalizer / gy.size(1))

        # Computation of xxt
        if group['layer_type'] == 'Conv2dEx':
            if not self.sua:
                x = F.unfold(x, mod.kernel_size, padding=mod.padding,
                             stride=mod.stride)
            else:
                x = x.view(x.size(0), x.size(1), -1)
            x = x.detach().permute(1, 0, 2).contiguous().view(x.size(1), -1)
        else:
            x = x.detach().t()
        if mod.bias is not None:
            ones = torch.ones_like(x[:1])
            x = torch.cat([x, ones], dim=0)
        if self.iteration_counter == 0:
            state['xxt'] = torch.mm(x, x.t()) * (normalizer / gy.size(1))
        else:
            state['xxt'].addmm_(mat1=x, mat2=x.t(),
                                beta=(1. - self.alpha),
                                alpha=self.alpha * normalizer / float(x.size(1)))

    def _inv_covs(self, xxt, ggt):
        """Inverses the covariances."""
        # Computes pi
        pi = 1.0
        if self.pi:
            tx = torch.trace(xxt) * ggt.shape[0]
            tg = torch.trace(ggt) * xxt.shape[0]
            pi = (tx / tg)
        # Regularizes and inverse
        damping = self.damping
        ixxt = _diag_add(xxt, (damping * pi)**0.5).inverse().contiguous()
        iggt = _diag_add(ggt, (damping / pi)**0.5).inverse().contiguous()
        return ixxt, iggt

def _is_dist():
    return hasattr(torch.distributed, 'is_initialized') and torch.distributed.is_initialized()

def _diag_add(mat_in, diag_elem, inplace=False):
    mat_out = mat_in
    if not inplace:
        mat_out = mat_out.clone()
    mat_out.diagonal().add_(diag_elem)
    return mat_out
