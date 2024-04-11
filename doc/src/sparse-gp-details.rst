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

---------------------
Adding a Group
---------------------	

Our implementation of the Sparse Gaussian Process can be efficiently updated with new groups in an online fasion. In otherwords this allows you to do:

.. code-block:: c

        auto fit_model = model.fit(dataset_a);
        fit_model.update_in_place(dataset_b);

Which will be equivalent to:

.. code-block:: c

        fit_model == model.fit(concatenate_datasets(dataset_a, dataset_b));

There are some papers which describe methods for performing online updates to sparse gaussian processes.  The paper

   [4] Streaming Sparse Gaussian Process Approximations
   Thang  D  Bui,  Cuong  Nguyen,  and  Richard  E  Turner.
   https://arxiv.org/abs/1705.07131

describes a way of both adding online observations and updating the inducing points for the the Variational Free Energy (VFE) approach (which is closely related to FITC).

   [4] Online sparse Gaussian process regression using FITC and PITCapproximations
   Hildo Bijl, Jan-Willem van Wingerden, Thomas B. Sch ̈on, and Michel Ver-haegen.
   https://hildobijl.com/Downloads/OnlineSparseGP.pdf

describes an approach to performing online updates to FITC and PITC but focuses on rank one updates in which the entire covariance is stored.  Here we describe how to update FITC and PITC with new batches of data.  These batches may contain a single observation (FITC) or a new group (PITC) and are used to update the QR decomposition (rather than the full dense covariances) used in the direct fit.

Consider the situation where we are first given a set of observation :math:`y_a`, fit the model, then want to update the model with new observations :math:`y_b`.  The existing model will consist of, :math:`v_a`, :math:`R_a`, :math:`P_a`, and :math:`L_{uu}` such that:

.. math::
	
	    \Sigma_a^{-1} &= \left(K_{uu} + K_{ua} \Lambda_a^{-1} K_{au}\right) \\
	    &= P_a R_a^T R_a P_a^T \\
	    v_a &= \Sigma_a K_{ua} \Lambda_{a}^{-1} y_a \\
	        &= P_a R_a^{-1} Q_a1^T \Lambda_a^{-1/2} y_a \\
	    K_{uu} &= L_{uu} L_{uu}^T
	


We'll be given a new group in the form of raw observations:

.. math::
	
	    y_b \sim \mathcal{N}\left(y_b, \Lambda_b + K_{bu} K_{uu}^{-1} K_{ub}\right)


And we wish to produce a new :math:`\hat{v}`, :math:`\hat{R}` and :math:`\hat{P}` which produce the same predictions as we would have gotten if we'd fit to :math:`\hat{y} = \begin{bmatrix} y_a \\ y_b \end{bmatrix}` directly.

To do so we start by explicitly writing out the components we would get with all groups available.  We'll use a hat, :math:`\hat{a}` to indicate quantities which correspond to a full fit.  Starting with :math:`\hat{\Sigma}`,

.. math::
	
	    \hat{\Sigma} &= \left(K_{uu} + K_{uf} \hat{\Lambda}^{-1} K_{fu}\right)^{-1} \\
	    &=\left(K_{uu} +
	         \begin{bmatrix} K_{ua} & K_{ub} \end{bmatrix}
	         \begin{bmatrix} \Lambda_a & 0 \\ 0 & \Lambda_b \end{bmatrix}^{-1} \begin{bmatrix} K_{au} \\ K_{bu} \end{bmatrix}
	       \right)^{-1} \\
	    &= \left(\Sigma_a^{-1} + K_{ub} \Lambda_b^{-1} K_{bu}
	       \right)^{-1}
	


We can then find a :math:`\hat{B}` such that :math:`\hat{\Sigma} = \left(\hat{B}^T \hat{B}\right)^{-1}` using the same approach as Equation~\ref{eq:B_qr}.  In particular we can see that by setting,

.. math::
	
	    \hat{B} &= \begin{bmatrix} R_a P_a^T \\ \Lambda_{b}^{-1/2} K_{bu} \end{bmatrix}
	


We can then represent :math:`\hat{\Sigma}` by,

.. math::
	
	\hat{\Sigma} &= \left(\hat{B}^T \hat{B}\right)^{-1} \\
	    &=  \left(P_a R_a^T R_a P_a^T + K_{ub} \Lambda_b^{-1} K_{bu}
	       \right)^{-1}\\
	    &= \left(\Sigma_a^{-1} + K_{ub} \Lambda_b^{-1} K_{bu}
	       \right)^{-1}
	


We then need to update the existing QR decomposition to get :math:`\hat{P}` and :math:`\hat{R}`,

.. math::
	
	    \hat{B} &= \begin{bmatrix} R_a P_a^T \\ \Lambda_{b}^{-1/2} K_{bu} \end{bmatrix} \\
	    &= \hat{Q} \hat{R} \hat{P}^T
	


Now we need to figure out how to update the information vector :math:`\hat{v}`.  If we had fit the model all at once the information vector would take the form,

.. math::
	
	    \hat{v} &= \left(K_{uu} + K_{uf} \hat{\Lambda}^{-1} K_{fu}\right)^{-1} K_{uf} \hat{\Lambda}^{-1} \hat{y} \\
	    &= \hat{\Sigma} \begin{bmatrix} K_{ua} \Lambda_a^{-1} & K_{ub} \Lambda_b^{-1} \end{bmatrix} \begin{bmatrix} y_a \\ y_b \end{bmatrix}
	


We'll already have the QR decomposition of :math:`\hat{B}` so we can try to find the :math:`z` such that :math:`\hat{v}` is the solution to the least squares problem, :math:`\left\lVert \hat{B} \hat{v} - z\right\rVert`.  Solving this system gives us,

.. math::
	
	    \hat{v} &= \left(\hat{B}^T \hat{B}\right)^{-1} \hat{B}^T z \\
	    &= \hat{\Sigma} \begin{bmatrix} P_a R_a^T & K_{ub} \Lambda_b^{-1/2} \end{bmatrix} \begin{bmatrix} z_a \\ z_b \end{bmatrix} \\
	


From which we can see that if we set,

.. math::
	
	    P_a R_a^T z_a &= K_{ua} \Lambda_a^{-1} y_a \\
	    &= \Sigma_a^{-1} v_a \\
	    z_a &= R_a^{-T} P_a^T \Sigma_a^{-1} v_a \\
	    &= R_a^{-T} P_a^T P_a R_a^T R_a P_a^T v_a \\
	    &= R_a P_a^T v_a
	


and

.. math::
	
	    z_b = \Lambda_b^{-1/2} y_b 

Then the following QR solution will effectively update the information vector,

.. math::
	
	    \hat{v} &= \hat{P} \hat{R}^{-1} \hat{Q}^T \begin{bmatrix}R_a P_a^T v_a \\  \Lambda_b^{-1/2} y_b \end{bmatrix}
	

After an update the only term which changes in the posterior covariance in Equation~\ref{eq:posterior} is the computation of the explained covariance,

.. math::
	
	    E_{**} = Q_{**} - K_{*u} \hat{\Sigma} K_{u*}

And since we've already computed :math:`\hat{\Sigma} = \left(\hat{B}^T \hat{B}\right)^{-1}` we don't need to do any further work.

In summary, updating an existing sparse Gaussian process (where any added observations are considered independent of existing ones) can be done by,

- Computing :math:`\Lambda_b^{-1/2}`.
- Computing (or updating) the QR decomposition, :math:`\hat{Q} \hat{R} \hat{P}^T = \begin{bmatrix}R_a P_a^T \\ \Lambda_b^{-1/2} K_{bu} \end{bmatrix}`.
- Setting :math:`\hat{v} = \hat{P} \hat{R}^{-1}\hat{Q}^T \begin{bmatrix} R_a P_a^T v_a \\ \Lambda_b^{-1/2} y_b \end{bmatrix}`

Note: As long as the new datasets you add consist of different groups you can continuously update the sparse Gaussian process retaining the same model you'd get if you had fit everything at once.  For FITC (each observation is treated independently) this is always the case, for PITC care needs to be taken that you don't update with a dataset containing groups which overlap with previous updates, the result would be over-confident predictions.

---------------------
Alternative Approach
---------------------

The first implementation of the Sparse Gaussiance process in albatross used an approach inspired by pymc3 and Gpy described here:

  - https://bwengals.github.io/pymc3-fitcvfe-implementation-notes.html
  - https://github.com/SheffieldML/GPy/blob/devel/GPy/inference/latent_function_inference/fitc.py

However, we found that while the posterior mean predictions were numerically stable, the posterior covariance term could not be broken into inner products which resulted in asymmetric covariance matrices which subsequently led to severe instability downstream.

You should be able to find this implementation and details using git history.

--------------------
PIC
--------------------

The way I started partitioning the covariance term into blocks is as follows:

.. math::

            (\sigma_*^2)^{PIC} &= K_* - \tilde{\mathbf{K}}^{PIC}_{*f} \left[ \tilde{\mathbf{K}}^{PITC}_{ff} \right]^{-1} \tilde{\mathbf{K}}^{PIC}_{f*} + \sigma^2 \\
            &= K_* - \begin{bmatrix} \mathbf{Q}_{* \cancel{B}} & \mathbf{K}_{* B} \end{bmatrix} \left(\mathbf{Q}_{ff} - \mathtt{blkdiag}(\mathbf{K}_{ff} - \mathbf{Q}_{ff})\right)^{-1} \begin{bmatrix} \mathbf{Q}_{\cancel{B} *} \\ \mathbf{K}_{B *} \end{bmatrix} \\
            &= K_* - \begin{bmatrix} \mathbf{Q}_{* \cancel{B}} & \mathbf{K}_{* B} \end{bmatrix} \left(\mathbf{K}_{fu} \mathbf{K}_{uu}^{-1} \mathbf{K}_{uf} + \mathbf{\Lambda} \right)^{-1} \begin{bmatrix} \mathbf{Q}_{\cancel{B} *} \\ \mathbf{K}_{B *} \end{bmatrix} 

The problem with doing this for PIC covariance (vs. PIC mean and PITC) is that we can't left-multiply the whole thing by :math:`\mathbf{K}_{uu}^{-1} \mathbf{K}_{uf}` (which in those instances leads to applying Woodbury's lemma to reduce the inverse to the size of the number of inducing points :math:`M`) because :math:`\mathbf{K}_{*B}` is not a low-rank approximation using the inducing points.  We can instead break up the inverse term into blocks:

.. math::
            \newcommand{VV}{\mathbf{V}} 
            (\sigma_*^2)^{PIC} &= K_* -
            \begin{bmatrix} \mathbf{Q}_{* \cancel{B}} & \mathbf{K}_{* B}\end{bmatrix}
            \begin{bmatrix} \mathbf{Q}_{\cancel{B} \cancel{B}} & \mathbf{Q}_{\cancel{B} B} \\ \mathbf{Q}_{B \cancel{B}} & \mathbf{Q}_{B B} \end{bmatrix}^{-1}
            \begin{bmatrix} \mathbf{Q}_{\cancel{B} *} \\ \mathbf{K}_{B *}\end{bmatrix}

If we substitute :math:`\mathbf{K}_{* B} = \mathbf{Q}_{* B} + \VV_{* B}` as with the mean, it doesn't work out nicely:
            
.. math::
            (\sigma_*^2)^{PIC} &= K_* - 
            \begin{bmatrix} \mathbf{Q}_{* \cancel{B}} & \mathbf{Q}_{* B} + \VV_{* B} \end{bmatrix}
            \underbrace{\begin{bmatrix} \mathbf{Q}_{\cancel{B} \cancel{B}} & \mathbf{Q}_{\cancel{B} B} \\ \mathbf{Q}_{B \cancel{B}} & \mathbf{Q}_{B B} \end{bmatrix}^{-1}}_{\mathbf{S}^{-1}}
            \begin{bmatrix} \mathbf{Q}_{\cancel{B} *} \\ \mathbf{Q}_{B *} + \VV_{B *}\end{bmatrix} \\
            &= K_* - \mathbf{Q}_{**}^{PITC} - \underbrace{\mathbf{Q}_{* f} \mathbf{S}^{-1}_{f B} \VV_{B *}}_{\mathbf{U}} - \mathbf{U}^T - \mathbf{V}_{* B} \mathbf{S}^{-1}_{B B} \mathbf{V}_{B *}

Now we have 3 correction terms to apply to the posterior PITC covariance.  The best thing I can think of is to apply Woodbury's lemma, but in the opposite direction to usual:

.. math::
            \newcommand{Lam}{\mathbf{\Lambda}}
            \newcommand{Kuu}{\mathbf{K}_{uu}}
            \newcommand{Kuf}{\mathbf{K}_{uf}}
            \newcommand{Kfu}{\mathbf{K}_{fu}}
            \mathbf{S}^{-1} &= \left(\Kfu \Kuu^{-1} \Kuf + \Lam\right)^{-1} \\
            &= \Lam^{-1} - \Lam^{-1} \Kfu \left( \Kuu + \Kuf \Lam^{-1} \Kfu \right)^{-1} \Kuf \Lam^{-1}

which involves decomposing the block-diagonal matrix :math:`\Lam` with blocks of size :math:`|B|` and a matrix the size :math:`M` of the inducing point set.  In practice after we precompute terms, we have a sequence of triangular factors that we can subset as needed to pick out :math:`B` and :math:`\cancel{B}`.  (Confusingly, one of these useful decompositions is the QR decomposition of the unrelated matrix :math:`B` in the PITC derivation above.)  

.. math::
            \mathbf{U} &= \mathbf{Q}_{* f} \mathbf{S}^{-1}_{f B} \VV_{B *} \\
            &= \mathbf{Q}_{* f} \left( \Lam^{-1} - \Lam^{-1} \Kfu \left( \Kuu + \Kuf \Lam^{-1} \Kfu \right)^{-1} \Kuf \Lam^{-1} \right)_{f B} \VV_{B *}

This looks appealingly like we could just keep combining instances of the QR decomposition of :math:`B`, but that would leave out the :math:`\mathbf{K}_{uu}` part.

The open question is how to efficiently distill this into various correction terms for each group that don't require operations that scale with :math:`\cancel{B}`, since the paper promises :math:`O((|B| + M)^2)` for predictive covariances after precomputation.  In principle, using sparse vectors / matrices for the :math:`*` target components, combined with Eigen's expression templates, should bring the complexity of some of these repeated solves for mostly empty vectors (for :math:`B`) down to :math:`O(|B|)`, and likewise for inducing points.

At this point, our cross-terms are preceded by :math:`Q_{* f}`, which expands to :math:`K_{*u} K_{uu}^{-1} K_{u f}`.  So actually we should be able to precompute everything except :math:`K_{*u}`, leaving prediction-time computations to scale with :math:`M`!

So for the variance, we must precompute:

 - :math:`P^TLDL^TP = \Lam`
 - :math:`\mathbf{G} = \Lam^{-1} \Kfu`
 - :math:`L_{uu} L_{uu}^T = \Kuu`
 - :math:`QRP^T = B = \begin{bmatrix}\Lam^{-\frac{1}{2}}\Kfu \\ L_{uu} \end{bmatrix}` such that :math:`\mathbf{S}^{-1} = \Lam^{-1} - \mathbf{G} \left(B^T B\right)^{-1} \mathbf{G}^T`, and blocks :math:`\mathbf{S}^{-1}_{a b}` can be got by choosing the right rows / columns with which to do permutations and back-substitutions.
 - :math:`\mathbf{W} = \Kuu^{-1} \mathbf{K}_{u f} \mathbf{S}^{-1}`

For the mean, we compute the information vector :math:`v` as in PITC.   

then for each group :math:`B`:

 - :math:`\mathbf{Y}_B = \Kuu^{-1} \mathbf{K}_{u B}`
 - :math:`v_b = \Lam_{B B}^{-1} \left( y_B - \mathbf{K}_{B u} v \right)`

Given that we have already computed :math:`\Kfu`, we can use :math:`\mathbf{K}_{u B}` and :math:`\mathbf{K}_{u \cancel{B}}` efficiently in Eigen using sparse matrices with a single entry per nonzero row or column to be used.
   
Then at prediction time, we must compute:

 - :math:`\mathbf{K}_{* B}`, :math:`O(|B|)`
 - :math:`\mathbf{K}_{* u}`, :math:`O(M)`
 - :math:`\mathbf{Q}_{* B} = \mathbf{K}_{* u} \mathbf{Y}_B`, :math:`O(M^2)` with the existing decomposition
 - :math:`\VV_{* B} = \mathbf{K}_{* B} - \mathbf{Q}_{* B}`, :math:`O(|B|^2)`
 - :math:`\mathbf{Q}_{**}^{PITC}` as with PITC
 - :math:`\mathbf{U} = \mathbf{K}_{* u} \mathbf{W}_B \VV_{B *}`, :math:`O(M + |B|)`?
 - :math:`\VV_{* B} \mathbf{S}^{-1}_{B B} \VV_{B *}`, :math:`O(|B|^2)`
   
To compute :math:`\mathbf{V}_{* B} \mathbf{S}^{-1}_{B B} \mathbf{V}_{B *}`, we form a (mostly zero) column of :math:`\mathbf{V}` for each feature, break the two terms of :math:`\mathbf{S}^{-1}` into symmetric parts, multiply by :math:`\mathbf{V}` and subtract, here in excruciating notational detail:

.. math::
           \VV_{* B} \mathbf{S}^{-1} \VV_{B *} &= \VV_{* B} \left( \Lam^{-1} - \Lam^{-1} \Kfu \left( \Kuu + \Kuf \Lam^{-1} \Kfu \right)^{-1} \Kuf \Lam^{-1} \right)_{B B} \VV_{B *} \\
           &= \VV_{* B}  \left( \left(P_\Lam L_\Lam^{-T} D_\Lam^{-\frac{1}{2}}\right) \underbrace{\left(D_\Lam^{-\frac{1}{2}} L_\Lam^{-1} P_\Lam^T\right)}_{Z_\Lam} - \mathbf{G}^T (B^T B)^{-1} \mathbf{G} \right) \VV_{B *} \\
           &= \VV_{* B}  \left( \mathbf{Z}_\Lam^T \mathbf{Z}_\Lam - \mathbf{G}^T (P_u R_u^{-1} R_u^{-T} P_u^T) \mathbf{G} \right) \VV_{B *} \\
           &= \VV_{* B}  \left( \mathbf{Z}_\Lam^T \mathbf{Z}_\Lam - \mathbf{G}^T (P_u R_u^{-1}) \underbrace{(R_u^{-T} P_u^T) \mathbf{G}}_{\mathbf{Z}_u} \right) \VV_{B *} \\
           &= \VV_{* B}  \left( \mathbf{Z}_\Lam^T \mathbf{Z}_\Lam - \mathbf{Z}_u^T \mathbf{Z}_u \right) \VV_{B *} \\
           &= \VV_{* B} \mathbf{Z}_\Lam^T \underbrace{\mathbf{Z}_\Lam \VV_{B *}}_{\mathbf{\xi}_\Lam} - \VV_{* B} \mathbf{Z}_u^T \underbrace{\mathbf{Z}_u \VV_{B *}}_{\mathbf{\xi}_u} \\
           &= \mathbf{\xi}_\Lam^T \mathbf{\xi}_\Lam - \mathbf{\xi}_u^T \mathbf{\xi}_u \\

Note that the left-hand (subscript :math:`\Lam`) term is the decomposition of a block-diagonal matrix, so it will only contain cross-terms between features corresponding to the same local block.  This calculation can be done blockwise.  The right-hand (subscript :math:`u`) term projects the features through the local block of the training dataset and then through the inducing points, so the cross-terms are not in general sparse, and this calculation involves a lot more careful indexing.

The same breakdown of :math:`\mathbf{S}^{-1}` can be used to compute :math:`\mathbf{W}` during the fit step.

The notation :math:`\mathbf{W}_B` indicates that the relevant columns of :math:`\mathbf{W}` are used on the right-hand side.  This is mathematically equivalent to making :math:`\VV_{B *}` have dimension :math:`N \times p` and be zero outside block :math:`B`.  Computationally the right factor must be a sparse object to preserve the desired asymptotics.