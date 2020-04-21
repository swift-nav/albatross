#################################################
Sparse Gaussian Process Implementation Details
#################################################

.. _sparse-gp-implementation:


----------------
Introduction
----------------

Here we describe an approximation technique for Gaussian processes proposed in:

   [1] Sparse Gaussian Processes using Pseudo-inputs
   Edward Snelson, Zoubin Ghahramani
   http://www.gatsby.ucl.ac.uk/~snelson/SPGP_up.pdf

Though the code uses notation closer to that used in this (excellent) overview of similar methods:

   [2] A Unifying View of Sparse Approximate Gaussian Process Regression
   Joaquin Quinonero-Candela, Carl Edward Rasmussen
   http://www.jmlr.org/papers/volume6/quinonero-candela05a/quinonero-candela05a.pdf


We've implemented the FITC and PITC methods which rely on an assumption that all observations (or groups of observations) are independent conditional on a set of inducing points (aka common state) which allows us to factorize the training covariance matrix into a low rank term and a diagonal term which leads to significant reduction in computational complexity compared to a full Gaussian process.

The method starts with the standard Gaussian process prior over the observations,

.. math::

    [\mathbf{f}] \sim \mathcal{N}(0, K_{ff})

Where :math:`K_{ff}` is the covariance function evaluated at all the training locations. It then uses a set of inducing points, :math:`u`, and makes some assumptions about the conditional distribution:,
 
.. math::

    [\mathbf{f}|u] \sim \mathcal{N}\left(K_{fu} K_{uu}^{-1} u, K_{ff} - Q_{ff}\right)

Where :math:`Q_{ff} = K_{fu} K_{uu}^{-1} K_{uf}` represents the variance in :math:`f` that is explained by :math:`u`.

For FITC (Fully Independent Training Contitional) the assumption is that :math:`K_{ff} - Q_{ff}` is diagonal, for PITC (Partially Independent Training Conditional) that it is block diagonal.  These assumptions lead to an efficient way of inferring the posterior distribution for some new location :math:`f^*`,

.. math::

    [f^*|f=y] \sim \mathcal{N}(K_{*u} \Sigma K_{uf} \Lambda^{-1} y, K_{**} - Q_{**} + K_{*u} \Sigma K_{u*})

Where :math:`\Sigma = (K_{uu} + K_{uf} \Lambda^{-1} K_{fu})^{-1}` and :math:`\Lambda = {\bf diag}(K_{ff} - Q_{ff})` and :math:`{\bf diag}` may mean diagonal or block diagonal.  See Equation 24 in [2] for details.  Regardless we end up with :math:`O(m^2n)` complexity instead of :math:`O(n^3)` of direct Gaussian processes.

----------------------
QR Solver
----------------------

The QR approach to the Sparse Gaussian process draws on the equivalence between the normal equations and QR solutions to the least squares problem and is outlined at length in:

   [3] Stable and EfficientGaussian Process Calculations.
   Leslie Foster, Alex Waagen, Nabeela Aijaz, Michael Hurley, Apolonio Luis,Joel Rinsky, Chandrika Satyavolu, and Mailbolt Com.
   http://www.jmlr.org/papers/volume10/foster09a/foster09a.pdf

Here we'll recap the approach they describe but with some minor changes to reflect specifics of our implementation.

Least squares consists of solving for :math:`x` which minimizes :math:`\lVert A x- b\rVert` with :math:`A \in \mathbb{R}^{n, m}`, :math:`x \in \mathbb{R}^m`, :math:`b \in \mathbb{R}^n` and :math:`n > m`.  There are several approaches that can be used to solve the problem.  One approach uses the normal equations to acquire :math:`x_N = \left(A^T A\right)^{-1} A^T b`.  A theoretically equivalent, but more numerically stable, approach involves computing the QR decomposition :math:`A = QR`, after which you can solve for :math:`x_{QR} = R^{-1} Q^T b`.

----------------------
Predictive Expectation
----------------------

To see how this can be applied to prediction in the context of Sparse Gaussian processes we first look at the mean of prediction distribution,

.. math::
	
	    y^{*} &= Q_{*f}\left(\Lambda + Q_{ff}\right)^{-1}y  \\
	     &= K_{*u} K_{uu}^{-1} K_{uf} \left(\Lambda + K_{fu} K_{uu}^{-1} K_{uf}\right)^{-1} y \\
	     &= K_{*u} \left(K_{uu} + K_{uf} \Lambda^{-1} K_{fu}\right)^{-1} K_{uf} \Lambda^{-1} y

We can break apart the inverted term into an inner product of an augmented tall skinny matrix,

.. math::
	
	    B &= \begin{bmatrix} \Lambda^{-1/2} K_{fu} \\ L_{uu}^T \end{bmatrix},	

where :math:`K_{uu} = L_{uu} L_{uu}^T`. Then we see that substituting :math:`B` back into the predicted mean gives,

.. math::
	
	    y^* &= K_{*u} \left(K_{uu} + K_{uf} \Lambda^{-1} K_{fu}\right)^{-1} K_{uf} \Lambda^{-1} y \\
	    &= K_{*u} \left(B^T B\right) B^T \Lambda^{-1/2} y \\
	    &= K_{*u} v

Where :math:`v = \left(B^T B\right) B^T \Lambda^{-1/2} y` is the information vector which is in the form of a normal equation solution to the system,

.. math::
	
	    \left\lVert B v - \begin{bmatrix}\Lambda^{-1/2} y \\ 0 \end{bmatrix} \right\rVert

Instead we can then compute the QR decomposition of :math:`B` which, when using a column pivoting scheme gives :math:`B = QRP^T`.  In its full form the QR decomposition of :math:`B \in \mathbb{R}^{k, m}` consists of the orthonormal matrix :math:`Q \in \mathbb{R}^{k, m}`, the upper triangular matrix :math:`R \in \mathbb{R}^{m, m}` and a permutation matrix :math:`P \in \mathbb{R}^{m, m}` which, by recognizing that the lower portion of the right hand side in our least squares problem is zero, can be broken into parts and used to solve for :math:`v`,

.. math::
	
	    B &= \begin{bmatrix} Q_1 \\ Q_2 \end{bmatrix} R P^T \\
	    v &= P R^{-1} Q_1^T \Lambda^{-1/2} y

----------------------
Predictive Covariance
----------------------

Turning back to the predictive distribution, we see we now have

.. math::
	
	[f^*|f=y] &\sim \mathcal{N}(K_{*u} v, K_{**} - Q_{**} + K_{*u} \Sigma K_{u*}) \\
	&\sim \mathcal{N}(K_{*u} v, K_{**} - E_{**})

Where we use :math:`E_{**}` to represent the covariance which is explained via :math:`[u|y]`.  The two terms involved can be interpreted as the law of total variance or in other words, :math:`E_{**}` consists of the covariance that would be explained if you knew the inducing points perfectly, :math:`Q_{**} = K_{*u} K_{uu}^{-1} K_{u*}`, minus the uncertainty of the inducing points, :math:`\Sigma`, mapped to the predictions, :math:`K_{*u} \Sigma K_{u*}`. Refactoring we have,

.. math::
	
	E_{**} &=  Q_{**} - K_{*u} \Sigma K_{u*} \\
	&= K_{*u} K_{uu}^{-1} K_{u*} - K_{*u} \Sigma K_{u*} \\
	&=  K_{*u} L_{uu}^{-T} L_{uu}^{-1} K_{u*} - K_{*u} \left(B^T B\right)^{-1} K_{u*} \\
	&= \left(L_{uu}^{-1} K_{u*}\right)^T \left(L_{uu}^{-1} K_{u*}\right) - K_{*u} \left(P R^T Q^T Q R P^T\right)^{-1} K_{u*}  \\
	&= \left(L_{uu}^{-1} K_{u*}\right)^T \left(L_{uu}^{-1} K_{u*}\right) - \left(R^{-T} P^T K_{u*}\right)^T \left(R^{-T} P^T K_{u*}\right) \\
	&= V_{a}^T V_{a} - V_{b}^T V_{b}
	


this has the nice property of being composed entirely of inner products which will ensure that the resulting posterior covariance will always be symmetric.

----------------
Summary
----------------

Putting this all together we have the following algorithm for the QR approach to a Sparse Gaussian Process.

First we compute the square root of :math:`\Lambda`.  When :math:`\Lambda` is a pure diagonal matrix this is simple,

.. math::

    \Lambda^{-1/2}_{ii} = 1 / \sqrt{\Lambda_{ii}}.

When :math:`\Lambda` is a block diagonal matrix we use the LDLT decomposition of each block, :math:`\Lambda_{b} = P^T L D L^T P` which we can use to get,

.. math::
	
    \Lambda_{b}^{-1/2} = D_b^{-1/2} L_b^{-1} P_b
	


where :math:`D_b^{-1/2}` is the square root of a diagonal matrix and :math:`P_b` is the permutation matrix computed when performing the LDLT with pivoting.

Next we can compute the square root of :math:`K_{uu}` which above we represented by :math:`L_{uu}`, but when implementing we'll use the LDLT again which gives us,

.. math::
	
	K_{uu}^{-1/2} = D_{uu}^{-1/2} L_{uu}^{-1} P_{uu}
	

We can now form the matrix :math:`B` and compute its QR decomposition,

.. math::
	
	    B &= \begin{bmatrix} \Lambda^{-1/2} K_{fu} \\ L_{uu}^T \end{bmatrix} \\
	    &= \begin{bmatrix} Q_1 \\ Q_2 \end{bmatrix} R P^T
	

which we can use to get :math:`v = P R^{-1} Q_1^T \Lambda^{-1/2} y`.
At this point we can store, :math:`v`, :math:`R`, :math:`P` and :math:`K_{uu}^{-1/2}` to use for predictions.

A prediction can then be made by computing, :math:`V_a = K_{uu}^{-1/2} K_{u*}` and :math:`V_b =  R^{-T} P^T K_{u*}` which we can then use to create,

.. math::
	
	    [f^*|f=y] \sim \mathcal{N}(K_{*u} v, K_{**} - V_a^T V_a + V_b^T V_b)
	
---------------------
Alternative Approach
---------------------

The first implementation of the Sparse Gaussiance process in albatross used an approach inspired by pymc3 and Gpy described here:

  - https://bwengals.github.io/pymc3-fitcvfe-implementation-notes.html
  - https://github.com/SheffieldML/GPy/blob/devel/GPy/inference/latent_function_inference/fitc.py

However, we found that while the posterior mean predictions were numerically stable, the posterior covariance term could not be broken into inner products which resulted in asymmetric covariance matrices which subsequently led to severe instability downstream.

