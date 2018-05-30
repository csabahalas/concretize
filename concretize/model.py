import torch
import torch.autograd as ag
import torch.nn as nn
import torch.nn.functional as F

class StraightThroughBernoulli(ag.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.bernoulli(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

ste_bernoulli = StraightThroughBernoulli.apply

def binary_concrete(log_alpha, beta=2/3, gamma=-0.1, zeta=1.1, hard=False, training=False, return_unif=False):
    u = torch.Tensor(*log_alpha.size()).uniform_().to(log_alpha.device)
    if training:
        s = u.log().add_((1 - u).log_().mul_(-1))
        s = (s + log_alpha) / beta
        s = s.sigmoid()
    else:
        s = log_alpha.sigmoid()
    if hard:
        s = (zeta - gamma) * s + gamma
        s = s.clamp(0, 1)
    return (s, u) if return_unif else s

def quantize_concrete(*args, **kwargs):
    return 2 * binary_concrete(*args, **kwargs) - 1

class ConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_weight = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(32, 1, 5, 5)))
        self.conv1_bias = nn.Parameter(torch.zeros(32))
        self.conv2_weight = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(64, 32, 5, 5)))
        self.conv2_bias = nn.Parameter(torch.zeros(64))
        self.mp = nn.MaxPool2d(2)
        self.lin = nn.Linear(1024, 10)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.beta = 2/3

    def forward(self, x):
        fn = quantize_concrete
        x = self.mp(F.relu(F.conv2d(x, fn(self.conv1_weight, beta=self.beta), bias=fn(self.conv1_bias, beta=self.beta))))
        x = self.bn1(x)
        x = self.mp(F.relu(F.conv2d(x, fn(self.conv2_weight, beta=self.beta), bias=fn(self.conv2_bias, beta=self.beta))))
        x = self.bn2(x)
        return self.lin(x.view(x.size(0), -1))

class MultiTuckerExampleModel(nn.Module):
    def __init__(self, t=(0.49,)):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.Tensor(len(t)).normal_(0, 0.01))
        self.register_buffer("t", torch.Tensor(t))
        self.criterion = nn.MSELoss()

    def mse(self, batch=1):
        alpha = binary_concrete(self.log_alpha.unsqueeze(0).repeat(batch, 1), training=self.training)
        if not self.training:
            alpha = alpha.bernoulli()
        t = self.t.unsqueeze(0).repeat(batch, 1)
        if self.training:
            t = t.bernoulli()
        else:
            t = t.round()
        return self.criterion(alpha, t)
