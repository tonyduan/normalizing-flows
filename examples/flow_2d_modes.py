import numpy as np
import logging
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn.functional as F
from argparse import ArgumentParser
from torch.distributions import MultivariateNormal

from nf.flows import *
from nf.models import NormalizingFlowModel


def gen_mixture_data(n=512):
    return np.r_[np.c_[np.ones(n // 2) + np.random.randn(n // 2) * 0.05, np.zeros(n // 2)],
                 np.c_[-np.ones(n // 2) + np.random.randn(n // 2) * 0.05, np.zeros(n // 2)]]

def plot_data(x, **kwargs):
    plt.scatter(x[:,0], x[:,1], marker="x", **kwargs)
    plt.xlim((-10, 10))
    plt.ylim((-5, 12))


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--n", default=512, type=int)
    argparser.add_argument("--flows", default=5, type=int)
    argparser.add_argument("--flow", default="MAF", type=str)
    argparser.add_argument("--iterations", default=300, type=int)
    args = argparser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    flow = eval(args.flow)
    flows = [flow(dim=2) for _ in range(args.flows)]
    prior = MultivariateNormal(torch.zeros(2), torch.eye(2))
    model = NormalizingFlowModel(prior, flows)

    optimizer = optim.Adam(model.parameters(), lr=0.005)
    x = torch.Tensor(gen_mixture_data(args.n))

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

    plt.figure(figsize=(8, 3))
    plt.subplot(1, 3, 1)
    plt.hist(z.data[:,0])
    plt.subplot(1, 3, 2)
    plt.hist(z.data[:,1])
    plt.show()

    samples = model.sample(500).data
    plot_data(samples, color="black", alpha=0.5)
    plt.title("Reconstructed sample")
    plt.show()

    for f in flows:
        x = f(x)[0].data
        plot_data(x, color="black", alpha=0.5)
        plt.show()

