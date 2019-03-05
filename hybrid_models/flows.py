import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


# supported non-linearities: note that the function must be invertible
functional_derivatives = {
    torch.tanh: lambda x: 1 - torch.pow(torch.tanh(x), 2),
    F.leaky_relu: lambda x: (x > 0).type(torch.FloatTensor) + \
                            (x < 0).type(torch.FloatTensor) * -0.01,
    F.elu: lambda x: (x > 0).type(torch.FloatTensor) + \
                     (x < 0).type(torch.FloatTensor) * torch.exp(x)
}


class Planar(nn.Module):
    """
    Planar flow.

        z = f(x) = x + u h(wᵀx + b)

    [Rezende and Mohamed, 2015]

    Parameters
    ----------
    dim:
    """
    def __init__(self, dim, nonlinearity=torch.tanh):
        super().__init__()
        self.h = nonlinearity
        self.w = nn.Parameter(torch.Tensor(dim))
        self.u = nn.Parameter(torch.Tensor(dim))
        self.b = nn.Parameter(torch.Tensor(1))
        self.reset_parameters(dim)

    def reset_parameters(self, dim):
        init.uniform_(self.w, -math.sqrt(1/dim), math.sqrt(1/dim))
        init.uniform_(self.u, -math.sqrt(1/dim), math.sqrt(1/dim))
        init.uniform_(self.b, -math.sqrt(1/dim), math.sqrt(1/dim))

    def forward(self, x):
        """
        Given x, returns z and the log-determinant log|df/dx|.

        Returns
        -------
        """
        if self.h in (F.elu, F.leaky_relu):
            u = self.u
        elif self.h == torch.tanh:
            scal = torch.log(1+torch.exp(self.w @ self.u)) - self.w @ self.u - 1
            u = self.u + scal * self.w / torch.norm(self.w)
        else:
            raise NotImplementedError("Non-linearity is not supported.")
        lin = torch.unsqueeze(x @ self.w, 1) + self.b
        z = x + u * self.h(lin)
        phi = functional_derivatives[self.h](lin) * self.w
        log_det = torch.log(torch.abs(1 + phi @ u) + 1e-4)
        return z, log_det

    def backward(self, z):
        raise NotImplementedError("Planar flow has no algebraic inverse.")


class Radial(nn.Module):
    """
    Radial flow.

        z = f(x) = = x + β h(α, r)(z − z0)

    [Rezende and Mohamed 2015]

    Parameters
    ----------
    """
    def __init__(self, dim):
        super().__init__()
        self.x0 = nn.Parameter(torch.Tensor(dim))
        self.log_alpha = nn.Parameter(torch.Tensor(1))
        self.beta = nn.Parameter(torch.Tensor(1))

    def reset_parameters(dim):
        init.uniform_(self.z0, -math.sqrt(1/dim), math.sqrt(1/dim))
        init.uniform_(self.log_alpha, -math.sqrt(1/dim), math.sqrt(1/dim))
        init.uniform_(self.beta, -math.sqrt(1/dim), math.sqrt(1/dim))

    def forward(self, x):
        """
        Given x, returns z and the log-determinant log|df/dx|.
        """
        m, n = x.shape
        r = torch.norm(x - self.x0)
        h = 1 / (torch.exp(self.log_alpha) + r)
        beta = -torch.exp(self.log_alpha) + torch.log(1 + torch.exp(self.beta))
        z = x + beta * h * (x - self.x0)
        log_det = (n - 1) * torch.log(1 + beta * h) + \
                  torch.log(1 + beta * h - \
                            beta * r / (torch.exp(self.log_alpha) + r) ** 2)
        return z, log_det


class AffineCouplingLayer(nn.Module):
    """
    Non-volume preserving flow.

    Parameters
    ----------
    """
    def __init__(self, dim, t=nn.Linear, s=nn.Linear, permutation=None):
        self.t = t
        self.s = s
        self.idx = dim // 2
        self.permutation = permutation

    def forward(self, x):
        t_transform = self.t(x[:self.idx])
        s_transform = self.s(x[:self.idx])
        lower = x[:self.idx]
        upper = t_transform + x[self.idx:] * torch.exp(s_transform)
        z = torch.stack([lower, upper])

    def backward(self, z):
        pass
