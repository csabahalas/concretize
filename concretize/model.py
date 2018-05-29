import torch
import torch.nn as nn

def binary_concrete(log_alpha, beta=1/3, gamma=-0.1, zeta=1.1, hard=False, training=False):
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
    return s

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
