### Normalizing Flows Models

Last update: December 2022.

---

Lightweight normalizing flows for generative modeling in PyTorch.

#### Setup

```math
\begin{align*}
\mathbf{x} & = f_\theta^{-1}(\mathbf{z}) & \mathbf{z} & = f_\theta(\mathbf{x}),
\end{align*}
```

where $f:\mathbb{R}^d \mapsto \mathbb{R}^d$ is an invertible function. The Change of Variables formula tells us that
```math
\begin{align*}
\underbrace{p(\mathbf{x})}_{\text{over }\mathbf{x}} &= \underbrace{p\left(f_\theta(\mathbf{x})\right)}_{\text{over }\mathbf{z}} \left|\mathrm{det}\left(\frac{\partial f_\theta \mathbf{x}}{\partial \mathbf{x}}\right)\right|
\end{align*}
```

Here $\frac{\partial f_\theta\mathbf{x}}{\partial \mathbf{x}}$ denotes the $d \times d$ Jacobian (this needs to be easy to compute).

We typically choose a simple distribution over the latent space, $p(\mathbf{z})\sim N(\mathbf{0},\mathbf{I})$.

Suppose we compose functions $f_\theta(\mathbf{x}) = f_1\circ f_2 \circ \dots f_k(\mathbf{x};\theta)$. The log-likelihood decomposes nicely.
```math
\begin{align*}
\log p(\mathbf{x}) & = \log p\left(f_\theta(\mathbf{x}\right)) + \sum_{i=1}^k\log \mathrm{det}\frac{\partial f_i(\mathbf{x};\theta)}{\partial \mathbf{x}}
\end{align*}
```
Sampling can be done easily, as long as $f_\theta^{-1}$ is tractable.

#### Implemented Flows

**Planar and radial flows** [1]. Note these have no algebraic inverse $f^{-1}(\mathbf{x})$.
```math
\begin{align*}
f(\mathbf{x}) & = \mathbf{x} + \mathbf{u}h(\mathbf{w}^\top \mathbf{z} + b)\\
f(\mathbf{x}) & = \mathbf{x} + \frac{\beta(\mathbf{x}-\mathbf{x}_0)}{\alpha + \|\mathbf{x}-\mathbf{x}_0\|}
\end{align*}
```
**Real NVP** [2]. Partition the vector $\mathbf{x}$ into components $\mathbf{x}^{(1)},\mathbf{x}^{(2)}$. Let $s,t$ be arbitrary neural networks $\mathbb{R}^d \mapsto \mathbb{R}^d$.
```math
\begin{align*}
f(\mathbf{x}^{(1)}) &= t(\mathbf{x}^{(2)}) + \mathbf{x}^{(1)}\odot \exp s(\mathbf{x}^{(2)})\\
f(\mathbf{x}^{(2)}) &= t(\mathbf{x}^{(1)}) + \mathbf{x}^{(2)}\odot \exp s(\mathbf{x}^{(1)})
\end{align*}
```
Here the diagonal of the Jacobian is simply $[\exp s(\mathbf{x}^{(2)}) \exp s(\mathbf{x}^{(1)})]$.

**Invertible 1x1 Convolution** [3].  Use an LU decomposition for computational efficiency.
```math
f(\mathbf{x})= W\mathbf{x}, \text{ where }W \text{ is square}
```
**ActNorm** [3]. Even more straightforward.
```math
f(\mathbf{x}) = W\mathbf{x} + b, \text{ where }W \text{ is diagonal}
```
**Masked Autoregressive Flow** [4]. For each dimension of $\mathbf{x}$, use a neural network to predict scalars $\mu,\alpha$.
```math
f(x_i) = (x_i - \mu(x_{< i})) / \mathrm{exp}(\alpha(x_{< i}))
```
Here the diagonal of the Jacobian is $\exp^{-1}(\alpha)$.

**Neural Spline Flow** [5]. Two versions: auto-regressive and coupling.
```math
\begin{align*}
f(x_i) & = \mathrm{RQS}_{g(x_{< i})}(x_i), \text{ (autoregressive) }\\
f(\mathbf{x}^{(1)}) & = \mathrm{RQS}_{g(\mathbf{x}^{(2)})}(\mathbf{x}^{(1)}) \text{ (coupling)}\\
f(\mathbf{x}^{(2)}) & = \mathrm{RQS}_{g(\mathbf{x}^{(1)})}(\mathbf{x}^{(2)})
\end{align*}
```
#### Examples

Below we show examples (in 1D and 2D) transforming a mixture of Gaussians into a unit Gaussian.

![](examples/ex_1d.png)

![](examples/ex_2d.png)

#### References

[1] Rezende, D. J. & Mohamed, S. Variational Inference with Normalizing Flows. in Proceedings of the 32nd International Conference on Machine Learning - Volume 37 - Volume 37 1530–1538 (JMLR.org, 2015).

[2] Dinh, L., Krueger, D., and Bengio, Y. (2014). NICE: Non-linear Independent Components Estimation.

[3] Kingma, D.P., and Dhariwal, P. (2018). Glow: Generative Flow with Invertible 1x1 Convolutions. In Advances in Neural Information Processing Systems 31, S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, eds. (Curran Associates, Inc.), pp. 10215–10224.

[4] Papamakarios, G., Pavlakou, T., and Murray, I. (2017). Masked Autoregressive Flow for Density Estimation. In Advances in Neural Information Processing Systems 30, I. Guyon, U.V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, eds. (Curran Associates, Inc.), pp. 2338–2347.

[5] Durkan, C., Bekasov, A., Murray, I., and Papamakarios, G. (2019). Neural Spline Flows.

#### License

This code is available under the MIT License.
