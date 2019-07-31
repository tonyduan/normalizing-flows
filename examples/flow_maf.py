import numpy as np
import logging
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn.functional as F
from argparse import ArgumentParser
from torch.distributions import MultivariateNormal

from nf.flows import MAF
from nf.models import NormalizingFlowModel


def gen_data(n=512):
    return np.r_[np.random.randn(n // 3, 2) + np.array([0, 6]),
                 np.random.randn(n // 3, 2) + np.array([2.5, 3]),
                 np.random.randn(n // 3, 2) + np.array([-2.5, 3])]


def gen_mixture_data(n=512):
    return np.r_[np.random.randn(n // 2, 2) + np.array([5, 3]),
                 np.random.randn(n // 2, 2) + np.array([-5, 3])]



def plot_data(x, **kwargs):
    plt.scatter(x[:,0], x[:,1], marker="x", **kwargs)
    plt.xlim((-10, 10))
    plt.ylim((-5, 12))


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--n", default=512, type=int)
    argparser.add_argument("--flows", default=5, type=int)
    argparser.add_argument("--iterations", default=300, type=int)
    argparser.add_argument("--use-mixture", action="store_true")
    args = argparser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    flows = [MAF(dim=2, hidden_dim=16) for _ in range(args.flows)]
    prior = MultivariateNormal(torch.zeros(2), torch.eye(2))
    model = NormalizingFlowModel(prior, flows)

    optimizer = optim.Adam(model.parameters(), lr=0.005)
    if args.use_mixture:
        x = torch.Tensor(gen_mixture_data(args.n))
    else:
        x = torch.Tensor(gen_data(args.n))

    for i in range(args.iterations):
        optimizer.zero_grad()
        z, prior_logprob, log_det = model(x)
        logprob = prior_logprob + log_det
        loss = -torch.mean(prior_logprob + log_det)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            logger.info(f"Iter: {i}\t" +
                        f"Logprob: {logprob.mean().data:.2f}\t" +
                        f"Prior: {prior_logprob.mean().data:.2f}\t" +
                        f"LogDet: {log_det.mean().data:.2f}")

    plot_data(x, color="grey")
    plot_data(z.data, color="black", alpha=0.5)
    plt.title("Latent space")
    plt.show()

    samples = model.sample(500).data
    plot_data(samples, color="black", alpha=0.5)
    plt.title("Reconstructed sample")
    plt.show()

    for f in flows:
        x = f(x)[0].data
        plot_data(x, color="black", alpha=0.5)
        plt.show()

