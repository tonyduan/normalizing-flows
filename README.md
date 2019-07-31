### Normalizing Flows Models

Last update: June 2019.

---

Implementations of simple normalizing flow models [1] in PyTorch.

We implement so far the following flows:
- Planar flows; <img alt="$f(x) = x + u h(w^\intercal z + b)$" src="svgs/545410cd1bceadb4b752b26bcd4583e5.svg" align="middle" width="171.05777204999998pt" height="24.65753399999998pt"/>
- Radial flows; <img alt="$f(x) = x + \frac{\beta}{\alpha + |x - x_0|}(x - x_0)$" src="svgs/60aff21144fe6991acc235d668f11af2.svg" align="middle" width="204.43170659999998pt" height="30.648287999999997pt"/>
- Real NVP; affine coupling layer; <img alt="$f(x^{(2)}) = t(x^{(1)}) + x^{(2)}\odot\exp s(x^{(1)}) $" src="svgs/1442af3a4ea68074dd58e967835d4e93.svg" align="middle" width="259.94306414999994pt" height="29.190975000000005pt"/> [2]
- Masked Autoregressive Flow (MAF); <img alt="$f(x_i) = (x_i - \mu(x_{&lt;i})) / \exp(\alpha(x_{&lt;i}))$" src="svgs/e97ff71b80919c8d61e65e9d1432bd0b.svg" align="middle" width="252.32704529999998pt" height="24.65753399999998pt"/> [3]

Note that planar and radial flows admit no algebraic inverse.

Below we show an example using MAF to transform a mixture of Gaussians into a unit Gaussian.

![](examples/ex.png)

#### References

[1] Rezende, D. J. & Mohamed, S. Variational Inference with Normalizing Flows. in Proceedings of the 32nd International Conference on Machine Learning - Volume 37 - Volume 37 1530–1538 (JMLR.org, 2015).

[2] Dinh, L., Krueger, D., and Bengio, Y. (2014). NICE: Non-linear Independent Components Estimation.

[3] Papamakarios, G., Pavlakou, T., and Murray, I. (2017). Masked Autoregressive Flow for Density Estimation. In Advances in Neural Information Processing Systems 30, I. Guyon, U.V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, eds. (Curran Associates, Inc.), pp. 2338–2347.

#### License

This code is available under the MIT License.
