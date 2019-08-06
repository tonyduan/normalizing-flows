### Normalizing Flows Models

Last update: August 2019.

---

Implementations of normalizing flow models [1] for generative modeling in PyTorch.

We implement so far the following flows:
- Planar flows; $f(x) = x + u h(w^\intercal z + b)$
- Radial flows; $f(x) = x + \frac{\beta}{\alpha + |x - x_0|}(x - x_0)$
- Real NVP; affine coupling layer; $f(x^{(2)}) = t(x^{(1)}) + x^{(2)}\odot\exp s(x^{(1)}) $ [2]
- Masked Autoregressive Flow (MAF); $f(x_i) = (x_i - \mu(x_{<i})) / \exp(\alpha(x_{<i}))$ [3]
- Invertible 1x1 Convolution; $f(x) = Wx$ where $W$ is square [4]
- ActNorm; $f(x) = Wx + b$ where $W$ is diagonal and $b$ is a constant [4]
- Autoregressive Neural Spline Flow (NSF-AF); $f(x_i) = \mathrm{RQS}_{\theta(x_{<i})}(x_i)$ [5] 
- Coupling Neural Spline Flow (NSF-CL); $f(x^{(2)}) = \mathrm{RQS}_{\theta(x^{(1)})}(x^{(2)})$ [5] 

Note that planar and radial flows admit no algebraic inverse.

Below we show an example transforming a mixture of Gaussians into a unit Gaussian.

![](examples/ex_2d.png)

![](examples/ex_1d.png)

#### References

[1] Rezende, D. J. & Mohamed, S. Variational Inference with Normalizing Flows. in Proceedings of the 32nd International Conference on Machine Learning - Volume 37 - Volume 37 1530–1538 (JMLR.org, 2015).

[2] Dinh, L., Krueger, D., and Bengio, Y. (2014). NICE: Non-linear Independent Components Estimation.

[3] Papamakarios, G., Pavlakou, T., and Murray, I. (2017). Masked Autoregressive Flow for Density Estimation. In Advances in Neural Information Processing Systems 30, I. Guyon, U.V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, eds. (Curran Associates, Inc.), pp. 2338–2347.

[4] Kingma, D.P., and Dhariwal, P. (2018). Glow: Generative Flow with Invertible 1x1 Convolutions. In Advances in Neural Information Processing Systems 31, S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, eds. (Curran Associates, Inc.), pp. 10215–10224.

[5] Durkan, C., Bekasov, A., Murray, I., and Papamakarios, G. (2019). Neural Spline Flows.

#### License

This code is available under the MIT License.
