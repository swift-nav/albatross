#################################################
Sparse Gaussian Process Implementation Details
#################################################

.. _sparse-gp-implementation:

--------------
Introduction
--------------

This class implements an approximation technique for Gaussian processes which relies on an assumption that all observations are independent (or groups of observations are independent) conditional on a set of inducing points.  The method is based off:

   [1] Sparse Gaussian Processes using Pseudo-inputs
   Edward Snelson, Zoubin Ghahramani
   http://www.gatsby.ucl.ac.uk/~snelson/SPGP_up.pdf

Though the code uses notation closer to that used in this (excellent) overview of these methods:

   [2] A Unifying View of Sparse Approximate Gaussian Process Regression
   Joaquin Quinonero-Candela, Carl Edward Rasmussen
   http://www.jmlr.org/papers/volume6/quinonero-candela05a/quinonero-candela05a.pdf

Very broadly speaking this method starts with a prior over the observations,

.. math::

   [\mathbf{f}] \sim \mathcal{N}(0, K_{ff})

where $$K_ff(i, j) = covariance_function(features[i], features[j])$$ and f represents the function value.

It then uses a set of inducing points, u, and makes some assumptions about the conditional distribution:

   [f|u] ~ N(K_fu K_uu^-1 u, K_ff - Q_ff)

Where Q_ff = K_fu K_uu^-1 K_uf represents the variance in f that is explained by u.

For FITC (Fully Independent Training Contitional) the assumption is that K_ff - Qff is diagonal, for PITC (Partially Independent Training Conditional) that it is block diagonal.  These assumptions lead to an efficient way of inferring the posterior distribution for some new location f*,

   [f*|f=y] ~ N(K_*u S K_uf^-1 A^-1 y, K_** − Q_** + K_*u S K_u*)

Where S = (K_uu + K_uf A^-1 K_fu)^-1 and A = diag(K_ff - Q_ff) and "diag" may mean diagonal or block diagonal.  Regardless we end up with O(m^2n) complexity instead of O(n^3) of direct Gaussian processes.  (Note that in [2] S is Sigma and A is lambda.)

Of course, the implementation details end up somewhat more complex in order to improve numerical stability.  A few great resources were heavily used to get those deails straight:

   - https://bwengals.github.io/pymc3-fitcvfe-implementation-notes.html
   - https://github.com/SheffieldML/GPy/blob/devel/GPy/inference/latent_function_inference/fitc.py

.. math::

  \mbox{elevation_scaling}(h) = 1. + \alpha \left(H - h\right)_{+}.

