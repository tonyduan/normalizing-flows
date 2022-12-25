import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


class NormalizingFlowModel(nn.Module):

    def __init__(self, dim, flows):
        super().__init__()
        self.prior = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
        self.flows = nn.ModuleList(flows)

    def forward(self, x):
        bsz, _ = x.shape
        log_det = torch.zeros(bsz)
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
        z, prior_logprob = x, self.prior.log_prob(x)
        return z, prior_logprob, log_det

    def inverse(self, z):
        bsz, _ = z.shape
        log_det = torch.zeros(bsz)
        for flow in self.flows[::-1]:
            z, ld = flow.inverse(z)
            log_det += ld
        x = z
        return x, log_det

    def sample(self, n_samples):
        z = self.prior.sample((n_samples,))
        x, _ = self.inverse(z)
        return x
