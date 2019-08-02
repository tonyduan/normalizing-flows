import numpy as np
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
    return np.r_[np.random.randn(n // 3, 2) + np.array([0, 6]),
                 np.random.randn(n // 3, 2) + np.array([2.5, 3]),
                 np.random.randn(n // 3, 2) + np.array([-2.5, 3])]


def gen_mixture_data(n=512):
    return np.r_[np.random.randn(n // 2, 2) + np.array([5, 3]),
                 np.random.randn(n // 2, 2) + np.array([-5, 3])]



def plot_data(x, **kwargs):
    plt.scatter(x[:,0], x[:,1], marker="x", **kwargs)
    plt.xlim((-3, 3))
    plt.ylim((-3, 3))


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--n", default=512, type=int)
    argparser.add_argument("--flows", default=2, type=int)
    argparser.add_argument("--flow", default="NSF_CL", type=str)
    argparser.add_argument("--iterations", default=500, type=int)
    argparser.add_argument("--use-mixture", action="store_true")
    argparser.add_argument("--convolve", action="store_true")
    args = argparser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    flow = eval(args.flow)
    flows = [flow(dim=2) for _ in range(args.flows)]
    if args.convolve:
        convs = [OneByOneConv(dim=2) for _ in range(args.flows)]
        flows = list(itertools.chain(*zip(convs, flows)))
    prior = MultivariateNormal(torch.zeros(2), torch.eye(2))
    model = NormalizingFlowModel(prior, flows)

    optimizer = optim.Adam(model.parameters(), lr=0.005)
    if args.use_mixture:
        x = torch.Tensor(gen_mixture_data(args.n))
    else:
        x = torch.Tensor(gen_data(args.n))

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
    plt.savefig("./examples/ex.png")
    plt.show()

    for f in flows:
        x = f(x)[0].data
        plot_data(x, color="black", alpha=0.5)
        plt.show()

