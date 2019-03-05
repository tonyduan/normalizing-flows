### Hybrid Models from Normalizing Flows

Last update: February 2019.

---

Work in progress implementation of hybrid models [1] in PyTorch, built on normalizing flows [2].

We implement so far the following flows:
- Planar flows; <img alt="$f(x) = x + u h(w^\intercal z + b)$" src="svgs/545410cd1bceadb4b752b26bcd4583e5.svg" align="middle" width="171.05777204999998pt" height="24.65753399999998pt"/>
- Radial flows; <img alt="$f(x) = x + \frac{\beta}{\alpha + |x - x_0|}(x - x_0)$" src="svgs/60aff21144fe6991acc235d668f11af2.svg" align="middle" width="204.43170659999998pt" height="30.648287999999997pt"/>

#### Usage

Todo.

#### References

[1] Nalisnick, E., Matsukawa, A., Teh, Y. W., Gorur, D. & Lakshminarayanan, B. Hybrid Models with Deep and Invertible Features. (2019).

[2] Rezende, D. J. & Mohamed, S. Variational Inference with Normalizing Flows. in Proceedings of the 32nd International Conference on Machine Learning - Volume 37 - Volume 37 1530–1538 (JMLR.org, 2015).

#### License

This library is available under the MIT License.
