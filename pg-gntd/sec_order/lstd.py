import math
from pickle import NONE
from sys import modules
from matplotlib.pyplot import axes
import copy

import torch
import torch.optim as optim
from sec_order._utils import ComputeA, ComputeG

from sec_order import _utils

class LSTD(optim.Optimizer):
    def __init__(self,
                 model,
                 lr=0.001,
                 stat_decay=0.9,
                 damping=0.05,
                 td_error=None,
                 TCov=5,
                 TInv=50):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, momentum=0.0, 
                        weight_decay=0.0)
        super(LSTD, self).__init__(model.parameters(), defaults)
        self.damping = damping
        
        self.MatAHandler = _utils.ComputeA()
        self.MatGHandler = _utils.ComputeG()
        
        self.known_modules = {'Linear', 'Conv2d'}
        
        self.modules = []
        self.grads = {}
        self.m_aa, self.m_gg = {}, {}
        self.m_inv = {}

        self.model = model
        self.stat_decay = stat_decay
        
        self._prepare_model()
        
        self.td_error = td_error
        
        self.steps = 0
        self.TCov = TCov
        self.TInv = TInv
        
    
    def _save_input(self, module, input):
        if torch.is_grad_enabled() and self.steps % self.TCov == 0:
            aa = self.MatAHandler(input[0].data, module)
            # Initialize buffers
            if self.steps == 0:
                # self.m_aa[module] = torch.diag(aa.new(aa.size(0)).fill_(1))
                self.m_aa[module] = copy.deepcopy(aa)
            _utils.update_running_stat(aa, self.m_aa[module], self.stat_decay)

    def _save_grad_output(self, module, grad_input, grad_output):
        # Accumulate statistics for Fisher matrices
        if self.steps % self.TCov == 0:
            gg = self.MatGHandler(grad_output[0].data, module, True)
            # Initialize buffers
            if self.steps == 0:
                # self.m_gg[module] = torch.diag(gg.new(gg.size(0)).fill_(1))
                self.m_gg[module] = copy.deepcopy(gg)
            _utils.update_running_stat(gg, self.m_gg[module], self.stat_decay)

    def _prepare_model(self):
        count = 0
        print(self.model)
        print("=> We keep following layers in LSTD. ")
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_full_backward_hook(self._save_grad_output)
                print('(%s): %s' % (count, module))
                count += 1
    
    def _update_inv(self, m):
        bs = self.m_aa[m].shape[0]
        sca = bs / bs**0.5
        mat_A = torch.bmm(self.m_gg[m].unsqueeze(2).mul(sca), self.m_aa[m].unsqueeze(2).transpose(1, 2))
        # mat_A = torch.bmm(self.m_gg[m].unsqueeze(2), self.m_aa[m].unsqueeze(2).transpose(1, 2))
        if self.td_error is None:
            self.grads[m] = mat_A
        else:
            self.grads[m] = torch.mean(mat_A * torch.reshape(self.td_error, [-1, 1, 1]), axes=0)
        # print("mat_A shape is ", mat_A.shape)
        mat_A = torch.reshape(mat_A, [bs, -1])
        mat_A = mat_A @ mat_A.t() + self.damping * torch.eye(bs)
        self.m_inv[m] = torch.inverse(mat_A)
        
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
    
    def step(self, optimizer=None, closure=None):
        updates = {}
        
        cnt = 0
        params = optimizer.param_groups[0]
        
        for m in self.modules:
            if self.steps % self.TInv == 0:
                self._update_inv(m)
            # F^{-1}g = g/k - U' (kI+UU')^{-1} U g/k
            if m.__class__.__name__ == 'Linear':
                grad, bias_grad = params['params'][cnt].grad.detach(), None
                grad_shape = grad.shape
                cnt += 1
                if m.bias is not None:
                    bias_grad = params['params'][cnt].grad.detach()
                    # print(bias_grad.shape)
                    cnt += 1
                    grad = torch.cat([grad, bias_grad.view(bias_grad.shape[0], 1)], dim=1)
                orig_shape = grad.size()
                grad = grad.view(-1)
                grad.div_(self.damping)
                U = torch.reshape(self.grads[m], [self.grads[m].shape[0], -1])
                tmp = U @ grad
                tmp = self.m_inv[m] @ tmp
                tmp = U.t() @ tmp
                grad.sub_(tmp)
                grad = grad.view(orig_shape)
                if m.bias is not None:
                    bias_grad = grad[:, -1].contiguous().view(*bias_grad.shape)
                    grad = grad[:, :-1].contiguous().view(*grad_shape)
                    updates[cnt-2] = grad
                    updates[cnt-1] = bias_grad
                else:
                    grad = grad.contiguous().view(*grad_shape)
                    updates[cnt-1] = grad
            elif m.__class__.__name__ == 'Conv2d':
                cnt += 1
        
        for i, p in enumerate(params['params']):
            if p.grad == None:
                continue
            p.grad = updates[i]
        
        self.steps += 1
    
    


    
    
    
    



    
    
    
    