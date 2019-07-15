### Normalizing Flows Models

Last update: June 2019.

---

Implementations of simple normalizing flow models [1] in PyTorch.

We implement so far the following flows:
- Planar flows; $f(x) = x + u h(w^\intercal z + b)$
- Radial flows; $f(x) = x + \frac{\beta}{\alpha + |x - x_0|}(x - x_0)$
- Real NVP; affine coupling layers [2]

Todo: autoregressive flows.

Note that planar and radial flows admit no algebraic inverse.

#### References

[1] Rezende, D. J. & Mohamed, S. Variational Inference with Normalizing Flows. in Proceedings of the 32nd International Conference on Machine Learning - Volume 37 - Volume 37 1530–1538 (JMLR.org, 2015).

[2] Dinh, L., Krueger, D., and Bengio, Y. (2014). NICE: Non-linear Independent Components Estimation.

#### License

This code is available under the MIT License.
