import numpy as np
import logging
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn.functional as F
from argparse import ArgumentParser
from torch.distributions import MultivariateNormal

from hybrid_models.flows import Planar
from hybrid_models.models import NormalizingFlowModel


def gen_data(n=512):
    x1 = np.random.randn(n) * 2
    x2 = np.random.randn(n) + 0.25 * x1 ** 2
    return np.c_[x1, x2]

def gen_mixture_data(n=512):
    return np.r_[np.random.randn(n, 2) + np.array([5, 3]),
                 np.random.randn(n, 2) + np.array([-5, 5])]

def plot_data(x, color="grey"):
    plt.scatter(x[:,0], x[:,1], marker="x", color=color)
    plt.xlim((-10, 10))
    plt.ylim((-5, 12))


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--n", default=512, type=int)
    argparser.add_argument("--flows", default=2, type=int)
    argparser.add_argument("--iterations", default=1000, type=int)
    argparser.add_argument("--use-mixture", action="store_true")
    args = argparser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    flows = [Planar(dim=2, nonlinearity=torch.tanh) for _ in range(args.flows)]
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
        loss = -torch.mean(prior_logprob + log_det)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            logger.info(f"Iter: {i}\t" +
                        f"Prior: {prior_logprob.mean().data:.2f}\t" +
                        f"LogDet: {log_det.mean().data:.2f}")

    plot_data(x, color="grey")
    plot_data(z.data, color="black")
    plt.show()

    for f in flows:
        x = f(x)[0].data
        plot_data(x)
        plt.show()
