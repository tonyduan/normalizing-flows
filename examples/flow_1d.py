import numpy as np
import scipy as sp
import scipy.stats
import itertools
import logging
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn.functional as F
from argparse import ArgumentParser
from torch.distributions import MultivariateNormal

from nf.flows import *
from nf.models import NormalizingFlowModel


def gen_data(n=512):
    return np.r_[np.random.randn(n // 2, 1) + np.array([2]),
                 np.random.randn(n // 2, 1) + np.array([-2])]

def plot_data(x, bandwidth = 0.2, **kwargs):
    kde = sp.stats.gaussian_kde(x[:,0])
    x_axis = np.linspace(-5, 5, 200)
    plt.plot(x_axis, kde(x_axis), **kwargs)


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--n", default=512, type=int)
    argparser.add_argument("--flows", default=2, type=int)
    argparser.add_argument("--flow", default="NSF_AR", type=str)
    argparser.add_argument("--iterations", default=500, type=int)
    args = argparser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    flow = eval(args.flow)
    flows = [flow(dim=1) for _ in range(args.flows)]
    prior = MultivariateNormal(torch.zeros(1), torch.eye(1))
    model = NormalizingFlowModel(prior, flows)

    optimizer = optim.Adam(model.parameters(), lr=0.005)
    x = torch.Tensor(gen_data(args.n))

    plot_data(x, color = "black")
    plt.show()

    for i in range(x.shape[1]):
        x[:,i] = (x[:,i] - torch.mean(x[:,i])) / torch.std(x[:,i])

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

    plt.figure(figsize=(8, 3))
    plt.subplot(1, 3, 1)
    plot_data(x, color="black", alpha=0.5)
    plt.title("Training data")
    plt.subplot(1, 3, 2)
    plot_data(z.data, color="darkblue", alpha=0.5)
    plt.title("Latent space")
    plt.subplot(1, 3, 3)
    samples = model.sample(500).data
    plot_data(samples, color="black", alpha=0.5)
    plt.title("Generated samples")
    plt.savefig("./examples/ex_1d.png")
    plt.show()
