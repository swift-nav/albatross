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

For FITC (Fully Independent Training Conditional) the assumption is that :math:`K_{ff} - Q_{ff}` is diagonal, for PITC (Partially Independent Training Conditional) that it is block diagonal.  These assumptions lead to an efficient way of inferring the posterior distribution for some new location :math:`f^*`,

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

We can then compute the QR decomposition of :math:`B` which, when using a column pivoting scheme gives :math:`B = QRP^T`.  In its full form the QR decomposition of :math:`B \in \mathbb{R}^{k, m}` consists of the orthonormal matrix :math:`Q \in \mathbb{R}^{k, m}`, the upper triangular matrix :math:`R \in \mathbb{R}^{m, m}` and a permutation matrix :math:`P \in \mathbb{R}^{m, m}` which, by recognizing that the lower portion of the right hand side in our least squares problem is zero, can be broken into parts and used to solve for :math:`v`,

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

-------------------------
Adding a Group (Updating)
-------------------------

Sparse Gaussian Processes can be efficiently updated with new groups in
an online fashion. In other words this allows you to do:

::

           auto fit_model = model.fit(dataset_a);
           fit_model.update_in_place(dataset_b);    

Which will be equivalent to:

::

           auto fit_model = model.fit(concatenate(dataset_a, dataset_b));

There are some papers which describe methods for performing online
updates to sparse Gaussian processes. The paper

::

      Streaming Sparse Gaussian Process Approximations
      Thang  D  Bui,  Cuong  Nguyen,  and  Richard  E  Turner.
      https://arxiv.org/abs/1705.07131

describes a way of both adding online observations and updating the
inducing points for the the Variational Free Energy (VFE) approach
(which is closely related to FITC). And,

::

      Online sparse Gaussian process regression using FITC and PITC approximations
      Hildo Bijl, Jan-Willem van Wingerden, Thomas B. Sch ̈on, and Michel Ver-haegen.
      https://hildobijl.com/Downloads/OnlineSparseGP.pdf

describes an approach to performing online updates to FITC and PITC but
focuses on rank one updates in which the entire covariance is stored.
Here we describe how to update FITC and PITC with new batches of data.
These batches may contain a single observation (FITC) or a new group
(PITC) and are used to update the QR decomposition (rather than the full
dense covariances) used in the direct fit.

Consider the situation where we are first given a set of observations
:math:`y_a`, fit the model, then want to update the model with new
observations :math:`y_b`. The existing model will consist of,
:math:`v_a`, :math:`R_a`, :math:`P_a`, and :math:`L_{uu}` such that,

.. math::

   \begin{aligned}
           \Sigma_a^{-1} &= \left(K_{uu} + K_{ua} \Lambda_a^{-1} K_{au}\right) \\
           &= P_a R_a^T R_a P_a^T \\
           v_a &= \Sigma_a K_{ua} \Lambda_{a}^{-1} y_a \\
               &= P_a R_a^{-1} Q_a1^T \Lambda_a^{-1/2} y_a \\
           K_{uu} &= L_{uu} L_{uu}^T
   \end{aligned}

We’ll be given new observations and can use the same low rank prior,

.. math::

   \begin{aligned}
   y_b \sim \mathcal{N}\left(m_b, \Lambda_b + K_{bu} K_{uu}^{-1} K_{ub}\right).
   \end{aligned}

And we wish to produce a new :math:`\hat{v}`, :math:`\hat{R}` and
:math:`\hat{P}` which produce the same predictions as we would have
gotten if we’d fit to
:math:`\hat{y} = \begin{bmatrix} y_a \\ y_b \end{bmatrix}` directly.

To do so we start by explicitly writing out the components we would get
with all groups available. We’ll use a hat, :math:`\hat{a}`, to indicate
quantities which correspond to a full fit. Starting with
:math:`\hat{\Sigma}`,

.. math::

   \begin{aligned}
           \hat{\Sigma} &= \left(K_{uu} + K_{uf} \hat{\Lambda}^{-1} K_{fu}\right)^{-1} \\
           &=\left(K_{uu} +
                \begin{bmatrix} K_{ua} & K_{ub} \end{bmatrix}
                \begin{bmatrix} \Lambda_a & 0 \\ 0 & \Lambda_b \end{bmatrix}^{-1} \begin{bmatrix} K_{au} \\ K_{bu} \end{bmatrix}
              \right)^{-1} \\
           &= \left(K_{uu} + K_{ua} \Lambda_a^{-1} K_{au} + K_{ub} \Lambda_b^{-1} K_{bu}
              \right)^{-1} \\
           &= \left(\Sigma_a^{-1} + K_{ub} \Lambda_b^{-1} K_{bu}
              \right)^{-1}
   \end{aligned}

We can then prepare for a similar QR approach to the one we used when
fitting and find a :math:`\hat{B}` such that
:math:`\hat{\Sigma} = \left(\hat{B}^T \hat{B}\right)^{-1}`. We can see
that by setting,

.. math::

   \begin{aligned}
   \hat{B} &= \begin{bmatrix} R_a P_a^T \\ \Lambda_{b}^{-1/2} K_{bu} \end{bmatrix}
   \end{aligned}

We can recover :math:`\hat{\Sigma}` by,

.. math::

   \begin{aligned}
       \hat{\Sigma} &= \left(\hat{B}^T \hat{B}\right)^{-1} \\
           &=  \left(P_a R_a^T R_a P_a^T + K_{ub} \Lambda_b^{-1} K_{bu}
              \right)^{-1}\\
           &= \left(\Sigma_a^{-1} + K_{ub} \Lambda_b^{-1} K_{bu}
              \right)^{-1}
   \end{aligned}

So by solving for the QR decomposition,

.. math::

   \begin{aligned}
           \hat{B} &= \begin{bmatrix} R_a P_a^T \\ \Lambda_{b}^{-1/2} K_{bu} \end{bmatrix} \\
           &= \hat{Q} \hat{R} \hat{P}^T
   \end{aligned}

We get the new updated values for :math:`\hat{P}` and :math:`\hat{R}`.

Now we need to figure out how to update the information vector
:math:`\hat{v}`. If we had fit the model all at once the information
vector would be,

.. math::

   \begin{aligned}
           \hat{v} &= \left(K_{uu} + K_{uf} \hat{\Lambda}^{-1} K_{fu}\right)^{-1} K_{uf} \hat{\Lambda}^{-1} \hat{y} \\
        &= \hat{\Sigma} K_{uf}\hat{\Lambda}^{-1} \hat{y}
   \end{aligned}

which we can divide into new and old observations,

.. math::

   \begin{aligned}
      \hat{v} &= \hat{\Sigma} \begin{bmatrix} K_{ua} \Lambda_a^{-1} & K_{ub} \Lambda_b^{-1} \end{bmatrix} \begin{bmatrix} y_a \\ y_b \end{bmatrix}
   \end{aligned}

We’ll already have the QR decomposition of :math:`\hat{B}` which we can
use to compute solutions in the form,

.. math::

   \begin{aligned}
           \hat{v} &= \left(\hat{B}^T \hat{B}\right)^{-1} \hat{B}^T z \\
        &= \hat{\Sigma} \hat{B}^T z
   \end{aligned}

so if we can find a :math:`z` such that

.. math::

   \begin{aligned}
       \hat{B}^T z = K_{uf}\hat{\Lambda}^{-1} \hat{y}
   \end{aligned}

then we can use the QR decomposition of :math:`\hat{B}` to get
:math:`\hat{v}`. By again dividing into new and old vectors we see that
we need,

.. math::

   \begin{aligned}
        \begin{bmatrix} P_a R_a^T & K_{ub} \Lambda_{b}^{-1/2} \end{bmatrix} \begin{bmatrix} z_a \\ z_b \end{bmatrix} &= \begin{bmatrix} K_{ua} \Lambda_a^{-1} & K_{ub} \Lambda_b^{-1} \end{bmatrix} \begin{bmatrix} y_a \\ y_b \end{bmatrix}
   \end{aligned}

We can satisfy that equality if we set,

.. math::

   \begin{aligned}
           P_a R_a^T z_a &= K_{ua} \Lambda_a^{-1} y_a \\
        K_{ub} \Lambda_b^{-1/2} z_b &= K_{ub} \Lambda_b^{-1} y_b
   \end{aligned}

Which gives us,

.. math::

   \begin{aligned}
   z_a &= R_a^{-T} P_a^T K_{ua} \Lambda_a^{-1} y_a \\
       &= R_a^{-T} P_a^T \Sigma_a^{-1} v_a \\
       &= R_a^{-T} P_a^T P_a R_a^T R_a P_a^T v_a \\
       &= R_a P_a^T v_a
   \end{aligned}

and

.. math::

   \begin{aligned}
           z_b = \Lambda_b^{-1/2} y_b 
   \end{aligned}

Then we can plug that into the QR solution to get,

.. math::

   \begin{aligned}
   \hat{v} &= \left(\hat{B}^T \hat{B}\right)^{-1} \hat{B}^T z \\
           &=\hat{P} \hat{R}^{-1} \hat{Q}^T \begin{bmatrix}R_a P_a^T v_a \\  \Lambda_b^{-1/2} y_b \end{bmatrix}
   \end{aligned}

In summary, updating an existing sparse Gaussian process (where any
added observations are considered independent of existing ones) can be
done by,

-  Computing :math:`\Lambda_b^{-1/2}`.

-  Computing the QR decomposition,
   :math:`\hat{Q} \hat{R} \hat{P}^T = \begin{bmatrix}R_a P_a^T \\ \Lambda_b^{-1/2} K_{bu} \end{bmatrix}`.

-  Setting
   :math:`\hat{v} = \hat{P} \hat{R}^{-1}\hat{Q}^T \begin{bmatrix} R_a P_a^T v_a \\ \Lambda_b^{-1/2} y_b \end{bmatrix}`

Note: As long as the new datasets you add consist of different groups
you can continuously update the sparse Gaussian process retaining the
same model you’d get if you had fit everything at once. For FITC this is
always the case (since each observation is treated independently), for
PITC care needs to be taken that you don’t update with a dataset
containing groups which overlap with previous updates, the result would
be over-confident predictions.

------------------------
Rebasing Inducing Points
------------------------

Consider a problem which is temporal in nature and you’d like to be able
to fit a model with some observations, make predictions, then update
that model with more recent observations and repeat. You could do this
using a dense Gaussian process by simply concatenating previous and
new observations into a larger dataset, but the result would be
unbounded growth in the model size and it would quickly become too large to
manage.

Instead, you could use a sparse Gaussian process, in which case you may
fit with observations on time, :math:`t_p`, then update the model with
new observations on time :math:`t_n`, and repeat. This would keep the
computation costs bounded, but if your problem is temporal in nature,
the time associated with the inducing points may diverge from the
time of the observations, causing the inducing point approximation to
degrade.

Here we describe how to take a model which has already been fit using
inducing points valid for some previous times, :math:`p`, and then
advance them forward in time. This temporal example here is just that;
an example. These operations do not require a temporal problem, they
more generally describe how to take a model based on some previous set
of inducing points, :math:`p`, and find an equivalent model based on new
inducing points, :math:`n`. The result opens up a style of algorithms
which involve, updating a model with observations, advancing the model
forward in time, updating with new observations and repeating.

Details
-------

Consider the situation where you’ve already fit a model using
observations :math:`f` with inducing points :math:`p` and you then want
to rebase the model on inducing points :math:`n`. This means you will have
computed,

.. math::

   \begin{aligned}
   \Sigma_p &= B_p^T B_p \\
   &= K_{pp} + K_{pf} \Lambda^{-1} K_{fp}
   \end{aligned}

and would like to find the equivalent for the new inducing points,

.. math::

   \begin{aligned}
   \Sigma_n &= B_n^T B_n \\
   &= K_{nn} + K_{nf} \Lambda^{-1} K_{fn}
   \end{aligned}

We don’t have :math:`K_{nf}` because when we fit the model using
observations, :math:`f`, we didn’t know the next inducing points,
:math:`n`. But we could use the Nystrom approximation
:math:`K_{nf} = K_{np} K_{pp}^{-1} K_{pf}` which can be interpreted as
saying: the only information we can capture in the new inducing points
is information that was already captured using the previous ones.

.. math::

   \begin{aligned}
   \Sigma_n &= K_{nn} + K_{nf} \Lambda^{-1} K_{fn} \\
   &\approx K_{nn} + K_{np} K_{pp}^{-1} K_{pf} \Lambda^{-1} K_{fp} K_{pp}^{-1} K_{pn}
   \end{aligned}

Now we can add and subtract a :math:`K_{np}K_{pp}^{-1}K_{pn}` term and
do some rearranging,

.. math::

   \begin{aligned}
   \Sigma_n &= K_{nn} + K_{np} K_{pp}^{-1} K_{pf} \Lambda^{-1} K_{fp} K_{pp}^{-1} K_{pn} + K_{np}K_{pp}^{-1}K_{pn} - K_{np}K_{pp}^{-1}K_{pn} \\
   &= K_{nn} + K_{np} K_{pp}^{-1} \left(K_{pp} + K_{pf} \Lambda^{-1} K_{fp} \right) K_{pp}^{-1} K_{pn} - K_{np}K_{pp}^{-1}K_{pn}
   \end{aligned}

Remember that
:math:`P_p R_p^T R_p P_p^T = K_{pp} + K_{pf} \Lambda^{-1} K_{fp}` and if
we solve for :math:`\hat{L}_{nn}` such that
:math:`\hat{L}_{nn} \hat{L}_{nn}^T = K_{nn} - K_{np}K_{pp}^{-1}K_{pn}`,
then we have,

.. math::

   \begin{aligned}
   \Sigma_n &= \left(R_{p} P_p^T K_{pp}^{-1} K_{pn}\right)^T \left(R_{p}P_p^T K_{pp}^{-1} K_{pn}\right) + \hat{L}_{nn} \hat{L}_{nn}^T
   \end{aligned}

we can write this as the symmetric product,
:math:`\Sigma_n = \hat{B}_n^T \hat{B}_n`, where

.. math::

   \begin{aligned}
       \hat{B}_n &= \begin{bmatrix} \hat{L}_{nn}^T \\ R_{p} P_p^T K_{pp}^{-1} K_{pn} \end{bmatrix}.
   \end{aligned}

Then by solving for :math:`\hat{Q}_n \hat{R}_n = \hat{B}_n \hat{P}_n` we
have,

.. math::

   \begin{aligned}
       \Sigma_n &\approx \hat{B}_n^T \hat{B}_n \\
       &= \hat{P}_n \hat{R}_n^T \hat{R}_n \hat{P}_n^T \\
       &= K_{nn} + K_{nf} \Lambda^{-1} K_{fn}
   \end{aligned}

Summary
-------

Starting with :math:`L_{pp}` and :math:`R_p` we can rebase onto new
inducing points by,

-  Building the matrices :math:`K_{pn}` and :math:`K_{nn}`.

-  Computing :math:`A = L_{pp}^{-1}K_{pn}`.

-  Solving for :math:`\hat{L}_{nn}` such that
   :math:`\hat{L}_{nn} \hat{L}_{nn}^T= K_{nn} - A^T A`.

-  Solving for
   :math:`\hat{Q}_n\hat{R}_n = \begin{bmatrix}\hat{L}_{nn}^T \\ R_p P_p^T L_{pp}^{-T} A \end{bmatrix}`.

-  Solving for :math:`L_{nn} = \mbox{chol}(K_{nn})`.

-  Solving for :math:`v_n = K_{nn}^{-1} K_{np} v_p`

---------------------
Alternative Approach
---------------------

The first implementation of the Sparse Gaussiance process in albatross used an approach inspired by pymc3 and Gpy described here:

  - https://bwengals.github.io/pymc3-fitcvfe-implementation-notes.html
  - https://github.com/SheffieldML/GPy/blob/devel/GPy/inference/latent_function_inference/fitc.py

However, we found that while the posterior mean predictions were numerically stable, the posterior covariance term could not be broken into inner products which resulted in asymmetric covariance matrices which subsequently led to severe instability downstream.

You should be able to find this implementation and details using git history.

