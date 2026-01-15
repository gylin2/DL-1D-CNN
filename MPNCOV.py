'''
@file: MPNCOV.py
@author: Jiangtao Xie
@author: Peihua Li

Copyright (C) 2018 Peihua Li and Jiangtao Xie

All rights reserved.
'''
import torch
import numpy as np
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F


class Covpool(Function):
     @staticmethod
     def forward(ctx, input):
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         M = x.data.shape[2]
         h = x.data.shape[2]
         w = x.data.shape[3]
         M = h*w
         x = x.reshape(batchSize,dim,M)
         I_hat = (-1./M/M)*torch.ones(M,M,device = x.device) + (1./M)*torch.eye(M,M,device = x.device)
         I_hat = I_hat.view(1,M,M).repeat(batchSize,1,1).type(x.dtype)
         y = x.bmm(I_hat).bmm(x.transpose(1,2))
         ctx.save_for_backward(input,I_hat)
         return y
     @staticmethod
     def backward(ctx, grad_output):
         input,I_hat = ctx.saved_tensors
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         h = x.data.shape[2]
         w = x.data.shape[3]
         M = h*w
         x = x.reshape(batchSize,dim,M)
         grad_input = grad_output + grad_output.transpose(1,2)
         grad_input = grad_input.bmm(x).bmm(I_hat)
         grad_input = grad_input.reshape(batchSize,dim,h,w)
         return grad_input

class Sqrtm(Function):
     @staticmethod
     def forward(ctx, input, iterN):
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         I3 = 3.0*torch.eye(dim,dim,device = x.device).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
         normA = (1.0/3.0)*x.mul(I3).sum(dim=1).sum(dim=1)
         A = x.div(normA.view(batchSize,1,1).expand_as(x))
         Y = torch.zeros(batchSize, iterN, dim, dim, requires_grad = False, device = x.device)
         Z = torch.eye(dim,dim,device = x.device).view(1,dim,dim).repeat(batchSize,iterN,1,1)
         if iterN < 2:
            ZY = 0.5*(I3 - A)
            Y[:,0,:,:] = A.bmm(ZY)
         else:
            ZY = 0.5*(I3 - A)
            Y[:,0,:,:] = A.bmm(ZY)
            Z[:,0,:,:] = ZY
            for i in range(1, iterN-1):
               ZY = 0.5*(I3 - Z[:,i-1,:,:].bmm(Y[:,i-1,:,:]))
               Y[:,i,:,:] = Y[:,i-1,:,:].bmm(ZY)
               Z[:,i,:,:] = ZY.bmm(Z[:,i-1,:,:])
            ZY = 0.5*Y[:,iterN-2,:,:].bmm(I3 - Z[:,iterN-2,:,:].bmm(Y[:,iterN-2,:,:]))
         y = ZY*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
         ctx.save_for_backward(input, A, ZY, normA, Y, Z)
         ctx.iterN = iterN
         return y
     @staticmethod
     def backward(ctx, grad_output):
         input, A, ZY, normA, Y, Z = ctx.saved_tensors
         iterN = ctx.iterN
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         der_postCom = grad_output*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
         der_postComAux = (grad_output*ZY).sum(dim=1).sum(dim=1).div(2*torch.sqrt(normA))
         I3 = 3.0*torch.eye(dim,dim,device = x.device).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
         if iterN < 2:
            der_NSiter = 0.5*(der_postCom.bmm(I3 - A) - A.bmm(der_postCom))
         else:
            dldY = 0.5*(der_postCom.bmm(I3 - Y[:,iterN-2,:,:].bmm(Z[:,iterN-2,:,:])) -
                          Z[:,iterN-2,:,:].bmm(Y[:,iterN-2,:,:]).bmm(der_postCom))
            dldZ = -0.5*Y[:,iterN-2,:,:].bmm(der_postCom).bmm(Y[:,iterN-2,:,:])
            for i in range(iterN-3, -1, -1):
               YZ = I3 - Y[:,i,:,:].bmm(Z[:,i,:,:])
               ZY = Z[:,i,:,:].bmm(Y[:,i,:,:])
               dldY_ = 0.5*(dldY.bmm(YZ) - 
                         Z[:,i,:,:].bmm(dldZ).bmm(Z[:,i,:,:]) - 
                             ZY.bmm(dldY))
               dldZ_ = 0.5*(YZ.bmm(dldZ) - 
                         Y[:,i,:,:].bmm(dldY).bmm(Y[:,i,:,:]) -
                            dldZ.bmm(ZY))
               dldY = dldY_
               dldZ = dldZ_
            der_NSiter = 0.5*(dldY.bmm(I3 - A) - dldZ - A.bmm(dldY))
         grad_input = der_NSiter.div(normA.view(batchSize,1,1).expand_as(x))
         grad_aux = der_NSiter.mul(x).sum(dim=1).sum(dim=1)
         for i in range(batchSize):
             grad_input[i,:,:] += (der_postComAux[i] \
                                   - grad_aux[i] / (normA[i] * normA[i])) \
                                   *torch.ones(dim,device = x.device).diag()
         return grad_input, None

class Triuvec(Function):
     @staticmethod
     def forward(ctx, input):
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         x = x.reshape(batchSize, dim*dim)
         I = torch.ones(dim,dim).triu().reshape(dim*dim)
         index = I.nonzero()
         y = torch.zeros(batchSize,int(dim*(dim+1)/2),device = x.device)
         y = x[:,index]
         ctx.save_for_backward(input,index)
         return y
     @staticmethod
     def backward(ctx, grad_output):
         input,index = ctx.saved_tensors
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         grad_input = torch.zeros(batchSize,dim*dim,device = x.device,requires_grad=False)
         grad_input[:,index] = grad_output
         grad_input = grad_input.reshape(batchSize,dim,dim)
         return grad_input

def CovpoolLayer(var):
    return Covpool.apply(var)

def SqrtmLayer(var, iterN):
    return Sqrtm.apply(var, iterN)

def TriuvecLayer(var):
    return Triuvec.apply(var)


from mpmath import *
import numpy as np
import torch

mp.dps = 32
one = mpf(1)
mp.pretty = True

def f(x):
    return sqrt(one-x)

# Derive the taylor and pade' coefficients for MTP, MPA
a = taylor(f, 0, 10)
pade_p, pade_q = pade(a, 5, 5)
a = torch.from_numpy(np.array(a).astype(float))
pade_p = torch.from_numpy(np.array(pade_p).astype(float))
pade_q = torch.from_numpy(np.array(pade_q).astype(float))

def matrix_taylor_polynomial(p, I):
    p_sqrt= I
    p_app = I - p
    p_hat = p_app
    for i in range(10):
      p_sqrt += a[i+1]*p_hat
      p_hat = p_hat.bmm(p_app)
    return p_sqrt

def matrix_pade_approximant(p,I):
    p_sqrt = pade_p[0]*I
    q_sqrt = pade_q[0]*I
    p_app = I - p
    p_hat = p_app
    for i in range(5):
        p_sqrt += pade_p[i+1]*p_hat
        q_sqrt += pade_q[i+1]*p_hat
        p_hat = p_hat.bmm(p_app)
    #There are 4 options to compute the MPA: comput Matrix Inverse or Matrix Linear System on CPU/GPU;
    #It seems that single matrix is faster on CPU and batched matrices are faster on GPU
    #Please check which one is faster before running the code;
    return torch.linalg.solve(q_sqrt, p_sqrt)
    #return torch.linalg.solve(q_sqrt.cpu(), p_sqrt.cpu()).cuda()
    #return torch.linalg.inv(q_sqrt).mm(p_sqrt)
    #return torch.linalg.inv(q_sqrt.cpu()).cuda().bmm(p_sqrt)

def matrix_pade_approximant_inverse(p,I):
    p_sqrt = pade_p[0]*I
    q_sqrt = pade_q[0]*I
    p_app = I - p
    p_hat = p_app
    for i in range(5):
        p_sqrt += pade_p[i+1]*p_hat
        q_sqrt += pade_q[i+1]*p_hat
        p_hat = p_hat.bmm(p_app)
    #There are 4 options to compute the MPA_inverse: comput Matrix Inverse or Matrix Linear System on CPU/GPU;
    #It seems that single matrix is faster on CPU and batched matrices are faster on GPU
    #Please check which one is faster before running the code;
    return torch.linalg.solve(p_sqrt, q_sqrt)
    #return torch.linalg.solve(p_sqrt.cpu(), q_sqrt.cpu()).cuda()
    #return torch.linalg.inv(p_sqrt).mm(q_sqrt)
    #return torch.linalg.inv(p_sqrt.cpu()).cuda().bmm(q_sqrt)

#Differentiable Matrix Square Root by MPA_Lya
class MPA_Lya(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M):
        normM = torch.norm(M,dim=[1,2]).reshape(M.size(0),1,1)
        I = torch.eye(M.size(1), requires_grad=False, device=M.device).reshape(1,M.size(1),M.size(1)).repeat(M.size(0),1,1)
        #This is for MTP calculation
        #M_sqrt = matrix_taylor_polynomial(M/normM,I)
        M_sqrt = matrix_pade_approximant(M / normM, I)
        M_sqrt = M_sqrt * torch.sqrt(normM)
        ctx.save_for_backward(M, M_sqrt, normM,  I)
        return M_sqrt

    @staticmethod
    def backward(ctx, grad_output):
        M, M_sqrt, normM,  I = ctx.saved_tensors
        b = M_sqrt / torch.sqrt(normM)
        c = grad_output / torch.sqrt(normM)
        for i in range(8):
            #In case you might terminate the iteration by checking convergence
            #if th.norm(b-I)<1e-4:
            #    break
            b_2 = b.bmm(b)
            c = 0.5 * (c.bmm(3.0*I-b_2)-b_2.bmm(c)+b.bmm(c).bmm(b))
            b = 0.5 * b.bmm(3.0 * I - b_2)
        grad_input = 0.5 * c
        return grad_input

#Differentiable Inverse Square Root by MPA_Lya_Inv
class MPA_Lya_Inv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M):
        normM = torch.norm(M,dim=[1,2]).reshape(M.size(0),1,1)
        I = torch.eye(M.size(1), requires_grad=False, device=M.device).reshape(1,M.size(1),M.size(1)).repeat(M.size(0),1,1)
        #M_sqrt = matrix_taylor_polynomial(M/normM,I)
        M_sqrt_inv = matrix_pade_approximant_inverse(M / normM, I)
        M_sqrt_inv = M_sqrt_inv / torch.sqrt(normM)
        ctx.save_for_backward(M, M_sqrt_inv,  I)
        return M_sqrt_inv

    @staticmethod
    def backward(ctx, grad_output):
        M, M_sqrt_inv,  I = ctx.saved_tensors
        M_inv = M_sqrt_inv.bmm(M_sqrt_inv)
        grad_lya = - M_inv.bmm(grad_output).bmm(M_inv)
        norm_sqrt_inv = torch.norm(M_sqrt_inv)
        b = M_sqrt_inv / norm_sqrt_inv
        c = grad_lya / norm_sqrt_inv
        for i in range(8):
            #In case you might terminate the iteration by checking convergence
            #if th.norm(b-I)<1e-4:
            #    break
            b_2 = b.bmm(b)
            c = 0.5 * (c.bmm(3.0 * I - b_2) - b_2.bmm(c) + b.bmm(c).bmm(b))
            b = 0.5 * b.bmm(3.0 * I - b_2)
        grad_input = 0.5 * c
        return grad_input

def FastMatSqrt (var):
    return MPA_Lya.apply(var)

def FastInvSqrt (var):
    return MPA_Lya_Inv.apply(var)


class Covariance(nn.Module):
    def __init__(self, 
                cov_type='norm',
                remove_mean=True,
                dimension_reduction=None,
                input_dim=2048,
        ):
        super(Covariance, self).__init__()
        self.cov_type = cov_type
        self.remove_mean = remove_mean
        self.dr = dimension_reduction
        if self.dr is not None:
            if self.cov_type == 'norm':
                self.conv_dr_block = nn.Sequential(
                    nn.Conv2d(input_dim, self.dr[0], kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(self.dr[0]),
                    nn.ReLU(inplace=True)
                )
            elif self.cov_type == 'cross':
                self.conv_dr_block = nn.Sequential(
                    nn.Sequential(
                        nn.Conv2d(input_dim, self.dr[0], kernel_size=1, stride=1, bias=False),
                        nn.BatchNorm2d(self.dr[0]),
                        nn.ReLU(inplace=True)
                    ),
                    nn.Sequential(
                        nn.Conv2d(input_dim, self.dr[1], kernel_size=1, stride=1, bias=False),
                        nn.BatchNorm2d(self.dr[1]),
                        nn.ReLU(inplace=True)
                    )
                )
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    

    def _remove_mean(self, x):
        _mean = F.adaptive_avg_pool2d(x, (1,1))
        x = x - _mean
        return x

    def _cov(self, x):
        # channel
        batchsize, d, h, w = x.size()
        N = h*w
        x = x.view(batchsize, d, N)
        y = (1. / N ) * (x.bmm(x.transpose(1, 2)))
        return y
    
    def _cross_cov(self, x1, x2):
        # channel
        batchsize1, d1, h1, w1 = x1.size()
        batchsize2, d2, h2, w2 = x2.size()
        N1 = h1*w1
        N2 = h2*w2
        assert batchsize1 == batchsize2
        assert N1 == N2
        x1 = x1.view(batchsize1, d1, N1)
        x2 = x2.view(batchsize2, d2, N2)

        y = (1. / N1) * (x1.bmm(x2.transpose(1, 2)))
        return y
    
    def forward(self, x, y=None):
        #import pdb;pdb.set_trace()
        if self.dr is not None:
            if self.cov_type == 'norm':
                x = self.conv_dr_block(x)
            elif self.cov_type == 'cross':
                if y is not None:
                    x = self.conv_dr_block[0](x)
                    y = self.conv_dr_block[1](y)
                else:
                    ori = x
                    x = self.conv_dr_block[0](ori)
                    y = self.conv_dr_block[1](ori)
        if self.remove_mean:
            x = self._remove_mean(x)
            if y is not None:
                y = self._remove_mean(y)               
        if y is not None:
            x = self._cross_cov(x, y)
        else:
            x = self._cov(x)
        return x

class Triuvec(Function):
     @staticmethod
     def forward(ctx, input):
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         x = x.reshape(batchSize, dim*dim)
         I = torch.ones(dim,dim).triu().reshape(dim*dim)
         index = I.nonzero(as_tuple=False)
         y = torch.zeros(batchSize,int(dim*(dim+1)/2),device = x.device).type(dtype)
         y = x[:,index]
         ctx.save_for_backward(input,index)
         return y
     @staticmethod
     def backward(ctx, grad_output):
         input,index = ctx.saved_tensors
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = grad_output.dtype
         grad_input = torch.zeros(batchSize,dim*dim,device = x.device,requires_grad=False).type(dtype)
         grad_input[:,index] = grad_output
         grad_input = grad_input.reshape(batchSize,dim,dim)
         return grad_input
