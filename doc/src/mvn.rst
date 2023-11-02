.. _mvn:

#######################################
Multivariate Gaussian Distributions
#######################################

Defining a Gaussian process can be thought of as providing a way to create arbitrary Multivariate Guassian Distributions (aka Multivariate Normal Distributions). The steps to fit and predict with a Gaussian process can then be thought of mostly as manipulations of Multivariate Gaussians. Here we provide a brief overview of some manipulations.

Starting with a univariate Gaussian,

.. math:: \mathbf{z} \sim \mathcal{N}\left(0, 1\right)

we can add a scalar,

.. math:: \mathbf{z} + a \sim \mathcal{N}\left(a, 1\right)

multiply by a scalar,

.. math:: a \mathbf{z} \sim \mathcal{N}\left(0, a^2\right)

or add two distributions,

.. math:: \mathcal{N}\left(m_a, \sigma_a^2\right) + \mathcal{N}\left(m_b, \sigma_b^2\right) = \mathcal{N}\left(m_a + m_b, \sigma_a^2 + \sigma_b^2\right).

Similar operations exist for multivariate Gaussian distributions, we can
add a vector, :math:`\mu \in \mathcal{R}^n`

.. math:: \mathcal{N}\left(0, I_{nn}\right) + \mu = \mathcal{N}\left(\mu, I\right)

multiply by a matrix, :math:`A \in \mathbb{R}^{m, n}`

.. math:: A \mathcal{N}\left(0, I\right) = \mathcal{N}\left(0, A A^T\right)

add two distributions,

.. math:: \mathcal{N}\left(\mu_a, \Sigma_a\right) + \mathcal{N}\left(\mu_b, \Sigma_b\right) = \mathcal{N}\left(\mu_a + \mu_b, \Sigma_a + \Sigma_b\right)

Adding and multiplying distributions can be useful, but the most
important operation for Gaussian processes is the conditional
distribution. Start by splitting a multivariate Gaussian distribution
into two variables,

.. math::

   \label{eq:ab_prior}
   \left[\begin{array}{c} 
     \mathbf{a}
    \\ 
     \mathbf{b}
    \end{array}\right] \sim \mathcal{N}\left(
   \begin{bmatrix} \mu_a \\ \mu_b \end{bmatrix},
   \left[
   \begin{bmatrix} \Sigma_{aa} & \Sigma_{ab} \\ \Sigma_{ba} & \Sigma_{bb} \end{bmatrix}\right]\right)

notice that the two random variables, :math:`\mathbf{a}` and
:math:`\mathbf{b}`, are correlated with each other. The conditional
distribution, :math:`\mathbf{a}
|b`, gives us the distribution of :math:`\mathbf{a}` if we knew the
value of :math:`\mathbf{b} = b`,

.. math::

   \mathbf{a}
   |b \sim \mathcal{N}\left(\mu_a + \Sigma_{ab} \Sigma_{bb}^{-1}\left(b - \mu_b\right) \hspace{0.1cm}, \hspace{0.1cm} \Sigma_{aa} - \Sigma_{ab} \Sigma_{bb}^{-1} \Sigma_{ba} \right)

Here we’ve started with a joint prior distribution of :math:`\mathbf{a}`
and :math:`\mathbf{b}` and found the posterior distribution, :math:`\mathbf{a}|b`, of :math:`\mathbf{a}` given :math:`\mathbf{b} = b`. These identities and more can be found in the `The Matrix Cookbook <https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf>`__ which is an extremely valuable resource for linear algebra in general.
