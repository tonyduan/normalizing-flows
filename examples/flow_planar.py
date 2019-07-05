import numpy as np
import logging
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn.functional as F
from argparse import ArgumentParser
from torch.distributions import MultivariateNormal

from nf.flows import Planar
from nf.models import NormalizingFlowModel


def gen_mixture_data(n=512):
    return np.r_[np.random.randn(n // 2, 2) + np.array([5, 3]),
                 np.random.randn(n // 2, 2) + np.array([-5, 5])]


def plot_data(x, color="grey", hist=False):
    if hist:
        plt.hist2d(x[:,0].numpy(), x[:,1].numpy(), bins=100,
                   range=np.array([(-10, 10), (-5, 12)]))
        plt.axis("off")
    else:
        plt.scatter(x[:,0], x[:,1], marker="x", color=color)
        plt.xlim((-10, 10))
        plt.ylim((-5, 12))


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--n", default=512, type=int)
    argparser.add_argument("--flows", default=10, type=int)
    argparser.add_argument("--iterations", default=1000, type=int)
    args = argparser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    flows = [Planar(dim=2, nonlinearity=torch.tanh) for _ in range(args.flows)]
    prior = MultivariateNormal(torch.zeros(2), torch.eye(2))
    model = NormalizingFlowModel(prior, flows)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    x = torch.Tensor(gen_mixture_data(args.n))

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

    plot_data(x, hist=True)
    plt.show()
    for f in flows:
        x = f(x)[0].data
        plot_data(x, hist=True)
        plt.show()
