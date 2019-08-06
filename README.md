### Normalizing Flows Models

Last update: August 2019.

---

Implementations of normalizing flow models [1] for generative modeling in PyTorch.

We implement so far the following flows:
- Planar flows; <img alt="$f(x) = x + u h(w^\intercal z + b)$" src="svgs/545410cd1bceadb4b752b26bcd4583e5.svg" align="middle" width="171.05777204999998pt" height="24.65753399999998pt"/>
- Radial flows; <img alt="$f(x) = x + \frac{\beta}{\alpha + |x - x_0|}(x - x_0)$" src="svgs/60aff21144fe6991acc235d668f11af2.svg" align="middle" width="204.43170659999998pt" height="30.648287999999997pt"/>
- Real NVP; affine coupling layer; <img alt="$f(x^{(2)}) = t(x^{(1)}) + x^{(2)}\odot\exp s(x^{(1)}) $" src="svgs/1442af3a4ea68074dd58e967835d4e93.svg" align="middle" width="259.94306414999994pt" height="29.190975000000005pt"/> [2]
- Masked Autoregressive Flow (MAF); <img alt="$f(x_i) = (x_i - \mu(x_{&lt;i})) / \exp(\alpha(x_{&lt;i}))$" src="svgs/e97ff71b80919c8d61e65e9d1432bd0b.svg" align="middle" width="252.32704529999998pt" height="24.65753399999998pt"/> [3]
- Invertible 1x1 Convolution; <img alt="$f(x) = Wx$" src="svgs/55c64c795eec86709b848718c6fb1804.svg" align="middle" width="81.11871074999999pt" height="24.65753399999998pt"/> where <img alt="$W$" src="svgs/84c95f91a742c9ceb460a83f9b5090bf.svg" align="middle" width="17.80826024999999pt" height="22.465723500000017pt"/> is square [4]
- ActNorm; <img alt="$f(x) = Wx + b$" src="svgs/776c3eedb8fc8128961e94382da5b06a.svg" align="middle" width="108.26469884999997pt" height="24.65753399999998pt"/> where <img alt="$W$" src="svgs/84c95f91a742c9ceb460a83f9b5090bf.svg" align="middle" width="17.80826024999999pt" height="22.465723500000017pt"/> is diagonal and <img alt="$b$" src="svgs/4bdc8d9bcfb35e1c9bfb51fc69687dfc.svg" align="middle" width="7.054796099999991pt" height="22.831056599999986pt"/> is a constant [4]
- Autoregressive Neural Spline Flow (NSF-AF); <img alt="$f(x_i) = \mathrm{RQS}_{\theta(x_{&lt;i})}(x_i)$" src="svgs/246c6ece83158ca84d352d8e9602f92c.svg" align="middle" width="159.88035854999998pt" height="24.65753399999998pt"/> [5] 
- Coupling Neural Spline Flow (NSF-CL); <img alt="$f(x^{(2)}) = \mathrm{RQS}_{\theta(x^{(1)})}(x^{(2)})$" src="svgs/95e7fe373e2b163eb46f1aed0f278185.svg" align="middle" width="185.43989969999998pt" height="29.190975000000005pt"/> [5] 

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
