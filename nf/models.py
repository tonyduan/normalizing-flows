import torch
import torch.nn as nn


class NormalizingFlowModel(nn.Module):

    def __init__(self, prior, flows):
        super().__init__()
        self.prior = prior
        self.flows = nn.ModuleList(flows)

    def forward(self, x):
        """
        Returns
        -------
        """
        m, n = x.shape
        log_det = torch.zeros(m)
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
        z, prior_logprob = x, self.prior.log_prob(x)
        return z, prior_logprob, log_det

    def backward(self, z):
        prior_logprob = self.prior.log_prob(z)
        return x, prior_logprob

    def sample(self, n_samples):
        z = self.prior.sample((n_samples,))
        x, _, _ = self.backward(z)
        return x

