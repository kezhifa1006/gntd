import math

import torch
import torch.optim as optim

# from _utils import (ComputeCovA, ComputeCovG)
# from _utils import update_running_stat
from sec_order import _utils


class TDKFAC(optim.Optimizer):
    def __init__(self,
                 model,
                 lr=0.001,
                 stat_decay=0.9,
                 damping=0.01,
                 TCov=5,
                 TInv=50,
                 pi=False,
                 constraint_norm=False,
                 batch_averaged=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        # TODO (CW): KFAC optimizer now only support model as input
        defaults = dict(lr=lr, damping=damping)
        super(TDKFAC, self).__init__(model.parameters(), defaults)
        self.lr = lr
        self.CovAHandler = _utils.ComputeCovA()
        self.CovGHandler = _utils.ComputeCovG()
        self.damping = damping
        self.pi = pi
        self.constraint_norm = constraint_norm
        self.batch_averaged = batch_averaged

        self.known_modules = {'Linear', 'Conv2d'}

        self.modules = []
        self.grad_outputs = {}

        self.model = model
        self._prepare_model()

        self.steps = 0

        self.m_aa, self.m_gg = {}, {}
        self.inv_aa, self.inv_gg = {}, {}
        self.stat_decay = stat_decay

        self.TCov = TCov
        self.TInv = TInv

        self.acc_stats = True

    def _save_input(self, module, input):
        if torch.is_grad_enabled() and self.steps % self.TCov == 0:
            aa = self.CovAHandler(input[0].data, module)
            # Initialize buffers
            if self.steps == 0:
                self.m_aa[module] = torch.diag(aa.new(aa.size(0)).fill_(1))
            _utils.update_running_stat(aa, self.m_aa[module], self.stat_decay)

    def _save_grad_output(self, module, grad_input, grad_output):
        # Accumulate statistics for Fisher matrices
        if self.acc_stats and self.steps % self.TCov == 0:
            gg = self.CovGHandler(grad_output[0].data, module, self.batch_averaged)
            # Initialize buffers
            if self.steps == 0:
                self.m_gg[module] = torch.diag(gg.new(gg.size(0)).fill_(1))
            _utils.update_running_stat(gg, self.m_gg[module], self.stat_decay)

    def _prepare_model(self):
        count = 0
        print(self.model)
        print("=> We keep following layers in KFAC. ")
        for module in self.model.modules():
            classname = module.__class__.__name__
            # print('=> We keep following layers in KFAC. <=')
            if classname in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)
                print('(%s): %s' % (count, module))
                count += 1

    def _update_inv(self, m):
        """Do eigen decomposition for computing inverse of the ~ fisher.
        :param m: The layer
        :return: no returns.
        """
        pi = 1.0
        if self.pi:
            ta = torch.trace(self.m_aa[m]) * self.m_gg[m].shape[0]
            tg = torch.trace(self.m_gg[m]) * self.m_aa[m].shape[0]
            pi = (ta / tg)
        # Regularizes and inverse
        damping = self.damping
        self.inv_aa[m] = _diag_add(self.m_aa[m], (damping * pi)**0.5).inverse().contiguous()
        self.inv_gg[m] = _diag_add(self.m_gg[m], (damping / pi)**0.5).inverse().contiguous()

    @staticmethod
    def _get_matrix_form_grad(m, classname):
        """
        :param m: the layer
        :param classname: the class name of the layer
        :return: a matrix form of the gradient. it should be a [output_dim, input_dim] matrix.
        """
        if classname == 'Conv2d':
            p_grad_mat = m.weight.grad.data.view(m.weight.grad.data.size(0), -1)  # n_filters * (in_c * kw * kh)
        else:
            p_grad_mat = m.weight.grad.data
        if m.bias is not None:
            p_grad_mat = torch.cat([p_grad_mat, m.bias.grad.data.view(-1, 1)], 1)
        return p_grad_mat

    def _get_natural_grad(self, m, p_grad_mat, damping):
        """
        :param m:  the layer
        :param p_grad_mat: the gradients in matrix form
        :return: a list of gradients w.r.t to the parameters in `m`
        """
        # p_grad_mat is of output_dim * input_dim
        # inv((ss')) p_grad_mat inv(aa') = [ Q_g (1/R_g) Q_g^T ] @ p_grad_mat @ [Q_a (1/R_a) Q_a^T]
        v = torch.mm(torch.mm(self.inv_gg[m], p_grad_mat), self.inv_aa[m])
        # e_, v_ = torch.eig(self.inv_gg[m])
        if m.bias is not None:
            # we always put gradient w.r.t weight in [0]
            # and w.r.t bias in [1]
            v = [v[:, :-1], v[:, -1:]]
            v[0] = v[0].view(m.weight.grad.data.size())
            v[1] = v[1].view(m.bias.grad.data.size())
        else:
            v = [v.view(m.weight.grad.data.size())]

        return v
    
    def constraint_norm_clip(self, updates, params=None):
        fisher_norm = 0.0
        for i, p in enumerate(params['params']):
            if p.grad == None:
                continue
            fisher_norm += (p.grad * updates[i]).sum()
        
        fisher_norm = max(fisher_norm, 1e-6)
        scale = (1.0 / fisher_norm) ** 0.5
        
        return scale     
    
    def step(self, optimizer=None, updates_tmp=None, closure=None):
        self_params = self.param_groups[0]
        params = optimizer.param_groups[0]
        lr = self.lr
        damping = self.damping
        updates = {}
        
        cnt = 0
        
        for m in self.modules:
            if self.steps % self.TInv == 0:
                    self._update_inv(m)
            if m.__class__.__name__ == 'Linear':
                if updates_tmp is None:
                    grad, bias_grad = params['params'][cnt].grad.detach(), None
                    grad_shape = grad.shape
                    cnt += 1
                    if m.bias is not None:
                        bias_grad = params['params'][cnt].grad.detach()
                        cnt += 1
                        grad = torch.cat([grad, bias_grad.view(bias_grad.shape[0], 1)], dim=1)
                else:
                    grad, bias_grad = updates_tmp[cnt], None
                    grad_shape = grad.shape
                    cnt += 1
                    if m.bias is not None:
                        bias_grad = updates_tmp[cnt]
                        cnt += 1
                        grad = torch.cat([grad, bias_grad.view(bias_grad.shape[0], 1)], dim=1) 
                v = self._get_natural_grad(m, grad, damping)               
                if m.bias is not None:
                    updates[cnt-2] = v[0]
                    updates[cnt-1] = v[1]
                else:
                    grad = grad.contiguous().view(*grad_shape)
                    updates[cnt-1] = v[0]
            elif m.__class__.__name__ == 'Conv2d':
                cnt += 1
        
        scale = 1.0
        if self.constraint_norm:
            # Not use updates_tmp
            scale = self.constraint_norm_clip(updates, params)
  
        if updates_tmp is None:      
            for i, p_ in enumerate(params['params']):
                if p_.grad == None:
                    continue
                p_.grad = updates[i] * scale
        else:
            for i, p in enumerate(self_params['params']):
                if p.grad is None:
                    continue
                d_p = updates[i]
                p.data.add_(-lr, d_p)
                    
        self.steps += 1


def _diag_add(mat_in, diag_elem, inplace=False):
    mat_out = mat_in
    if not inplace:
        mat_out = mat_out.clone()
    mat_out.diagonal().add_(diag_elem)
    return mat_out

