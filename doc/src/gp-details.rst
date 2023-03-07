#################################################
Gaussian Process Implementation Details
#################################################

.. _gp-implementation:

----------------
Introduction
----------------

There are a number of excellent introductions to Gaussian processes (see our :ref:`references`) here we assume a general understanding of GPs and focus on describing the details of the implementations we use in `albatross`.

----------------
Notation
----------------

We (mostly) borrow the notation from Gaussian Process for Machine Learning. A process (or function) is written :math:`f(x)` where :math:`x` is some location. In ``albatross`` we call :math:`x` a feature (see :ref:`datasets`). We use bold font, :math:`\mathbf{x}` to represent a vector (or arbitrary number) of some variable. Lower case :math:`x` is used for a single variable and upper case :math:`X` to represent a matrix.

Actual measurements :math:`y` of some process :math:`f` at some location :math:`x` can be written

.. math::

    y \leftarrow f(x) + \epsilon

where :math:`\epsilon \sim \mathcal{N}(0, \sigma_\epsilon^2)` is measurement noise.  Sometimes the measurement noise is included as part of the process itself, in which case we'd just write,

.. math::

    y \leftarrow f(x)
    
which in english would be: :math:`y` is drawn from :math:`f(x)`. In albatross the output :math:`y` corresponding to some input :math:`x` is called a target (see :ref:`datasets`). If :math:`f(x)` is a Gaussian process we would write,

.. math::

    f(\mathbf{x}) \sim \mathcal{GP}\left(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x'})\right)
    
where :math:`m(\mathbf{x})` is a mean function and :math:`k(\mathbf{x}, \mathbf{x'})` a covariance function. Note that frequently the mean function is assumed zero. A set of function evaluations at known locations, :math:`\mathbf{x}`, is written with bold font, :math:`\mathbf{f(x)}` or just :math:`\mathbf{f}`.  When :math:`f(x)` is a Gaussian process we can write the prior at known locations:

.. math::

    \mathbf{f} \sim \mathcal{N}(\mathbf{m}_f, K_{ff})

Where :math:`\mathbf{m}_f` and :math:`K_{ff}` are the mean and covariance function evaluated at all the locations. We could equivalently write this for each location:

.. math::

    \mathbf{f} = \begin{bmatrix}f(x_0) \\ \vdots \\ f(x_n)\end{bmatrix} \sim \mathcal{N}\left(\begin{bmatrix}m(x_0) \\\vdots \\ m(x_n)\end{bmatrix}, \begin{bmatrix}k(x_0, x_0) && \cdots && k(x_0, x_n) \\ \vdots && && \vdots \\ k(x_n, x_0) && \cdots && k(x_n, x_n)\end{bmatrix}\right)
 
when a process is actually measured we use :math:`y` to represent the measurement. Many measurements would also take on bold font,

.. math::

    \mathbf{y} \leftarrow \mathbf{f(x)}
    
In Gaussian process regression we're typically using the process to make predictions at new locations, :math:`\mathbf{x}^*`, given measurements, :math:`\mathbf{y}` taken at some known locations.  We could write the prior distribution of the process at known and new locations in block form,

.. math::

    \begin{bmatrix}\mathbf{f} \\ \mathbf{f^*}\end{bmatrix} \sim \mathcal{N}\left(0, \begin{bmatrix} K_{ff} && K_{f*} \\ K_{*f} && K_{**} \end{bmatrix} \right)

Notice that we dropped the mean, mostly just to make the rest of the equations easier to interpret, to get from a non zero mean process to a zero mean process you can subtract off the prior mean. When we actually make observations :math:`\mathbf{y} \leftarrow \mathbf{f}` we can then ask for the posterior distribution of :math:`\mathbf{f}^*` given :math:`\mathbf{f} = \mathbf{y}`,

.. math::

    [\mathbf{f^*}|\mathbf{f} = \mathbf{y}] \sim \mathcal{N}\left(K_{*f} K_{ff}^{-1} \mathbf{y}, K_{**} - K_{*f} K_{ff}^{-1} K_{f*}\right)
    
for brevity we'll often drop the :math:`\mathbf{f} = \mathbf{y}` part and just write:

.. math::

    \mathbf{f^*}|\mathbf{y} \sim \mathcal{N}\left(K_{*f} K_{ff}^{-1} \mathbf{y}, K_{**} - K_{*f} K_{ff}^{-1} K_{f*}\right)

This would read, "the posterior distribution, :math:`\mathbf{f^*}`, at locations :math:`\mathbf{x^*}`  given :math:`\mathbf{y}` is multivariate normal with the given mean and covariance". This is the `conditional distribution of a multivariate Gaussian distribution <https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions>`_.

You may also see

.. math::

    \mathbf{y^*}|\mathbf{y} \sim \mathcal{N}\left(K_{*f} K_{ff}^{-1} \mathbf{y}, K_{**} - K_{*f} K_{ff}^{-1} K_{f*}\right)

Particularly if measurement noise is already a part of the process :math:`f`.  However, this is a bit of an abuse of notation because (in this case) :math:`\mathbf{y^*}` are not actual measurements, they're hypothetical measurements of :math:`\mathbf{f^*}|\mathbf{f} = \mathbf{y}`. The point here is that it can be important to distinguish between the process itself, :math:`\mathbf{f}` (random variable), measurements of the process :math:`\mathbf{y}` (typically a real valued vector, maybe with associated measurement noise) and the locations/information associated with the process or measurements, :math:`\mathbf{x}` (possibly a struct holding arbitrary information describing a measurement).

----------------------
Model Fit
----------------------

When you build a Gaussian process in `albatross` and fit the model,

.. code-block:: c

    auto model = gp_from_covariance(k);
    RegressionDataset<> dataset(x, y);
    auto fit_model = model.fit(dataset);

we perform some of the intense computation up front. In this case we'd be building the covariance matrix associated with the features, :math:`\mathbf{x}`, decomposing it to make subsequent inversion easier and precomputing the information vector, :math:`\mathbf{v}`,

.. math::
    
    K_{ff} &= \begin{bmatrix}k(x_0, x_0) && \cdots && k(x_0, x_n) \\ \vdots && && \vdots \\ k(x_n, x_0) && \cdots && k(x_n, x_n)\end{bmatrix} \\
    P^TLDL^TP &= K_{ff} \\
    \mathbf{v} &= K_{ff}^{-1} \mathbf{y}

We've picked the Robust Cholesky decomposition (`the LDLT decomposition <https://eigen.tuxfamily.org/dox/classEigen_1_1LDLT.html>`_) which is known to have good numerical properties (due in large part to the pivoting which results in a permutation matrix :math:`P`).

-----------------------
Predictive Distribution
-----------------------

Once we've fit a model we can use it to make a prediction at arbitrary locations (read: features), :math:`\mathbf{x}^*`,

.. math::

    \mathbf{f^*}|\mathbf{y} \sim \mathcal{N}\left(K_{*f} K_{ff}^{-1} \mathbf{y}, K_{**} - K_{*f} K_{ff}^{-1} K_{f*}\right)

We can take advantage of some of the precomputed quantities to make this prediction step more efficient.  In particular we would write this,

.. math::

    \mathbf{f^*}|\mathbf{y} & \sim \mathcal{N}\left(K_{*f} \mathbf{v}, K_{**} - K_{*f} (P^TLDL^TP)^{-1} K_{f*}\right) \\
    & \sim \mathcal{N}\left(K_{*f} \mathbf{v}, K_{**} - (K_{*f} P^T L^{-T} D^{-1/2}) (D^{-1/2}L^{-1}P K_{f*}\right) \\
    & \sim \mathcal{N}\left(K_{*f} \mathbf{v}, K_{**} - Q_{f*}^T Q_{f*}\right)
    
Where :math:`Q_{f*} = D^{-1/2}L^{-1}P K_{*f}^T`.

To make a prediction in albatross you'd first fit the model (see above), then call,

.. code-block:: c

    const auto prediction = fit_model.predict(new_features);

This is a lazy operation (nothing is actually done yet, only saving the ``new_features`` where predictions are desired). You then have some choices for the actual prediction type you'd like:

^^^^^^^^^^^^^^^^^^^
Mean Predictions
^^^^^^^^^^^^^^^^^^^

Calling:

.. code-block:: c

    const Eigen::VectorXd mean = prediction.mean();

would:

* Evaluate :math:`K_{*f}`
* Compute the mean :math:`K_{*f} \mathbf{v}`

^^^^^^^^^^^^^^^^^^^^
Marginal Predictions
^^^^^^^^^^^^^^^^^^^^

Calling:

.. code-block:: c

    const MarginalDistribution marginal = prediction.marginal();

would:

* Compute :math:`Q_{f*} = D^{-1/2}L^{-1}P K_{*f}^T`  
* Evaluate the prior variance :math:`\mbox{diag}(K_{**})`
* Compute the posterior variance :math:`\mbox{diag}(K_{**}) - \mbox{diag}(Q_{f*}^T Q_{f*})`

^^^^^^^^^^^^^^^^^^^
Joint Predictions
^^^^^^^^^^^^^^^^^^^

Calling:

.. code-block:: c

    const JointDistribution marginal = prediction.joint();

would:

* Evaluate the prior covariance :math:`K_{**}`
* Compute the posterior covariance :math:`K_{**} - Q_{f*}^T Q_{f*}`

++++++++++++++++++++++++++++++++++++++++
Cross Validation Implementation Details
++++++++++++++++++++++++++++++++++++++++

In `Gaussian Processes for Machine Learning`_ (Section 5.4.2) they describe an efficient way for making leave one out predictions, here we expand that same trick to enable making leave one group out predictions.

Consider the case where we have a set of observations, :math:`y`, and we would like to make leave one group out cross validated predictions and by groups we mean independent sets of one or more variables.

We start with our GP,

.. math::

    \mathbf{f} \sim \mathcal{N}\left(0, \Sigma \right)

Which we can then break into groups,

.. math::

    \begin{bmatrix} \mathbf{\hat{y}} \\ \mathbf{y_i} \end{bmatrix} \sim \mathcal{N}\left(\begin{bmatrix} 0 \\ 0 \end{bmatrix}, \begin{bmatrix}\hat{\Sigma} & S \\ S^T & C \end{bmatrix}\right)

Where we will be using a subset of observations, :math:`\hat{y}` to make predictions for a held out set of locations, :math:`x_i`.  We can do this directly using the Gaussian process predict formula,

.. math::

    [\mathbf{y_i}|\mathbf{\hat{y}}=\hat{y}] \sim \mathcal{N}\left(S^T \hat{\Sigma}^{-1} \hat{y}, C - S^T \hat{\Sigma}^{-1} S\right)

But doing so would require computing :math:`\hat{\Sigma}^{-1}` for every group, :math:`i`, that we hold out.  So if we're doing leave one out with :math:`n` observations we have to do the :math:`\mathcal{O}(n^3)` inversion :math:`n` times leading to :math:`\mathcal{O}(n^4)` complexity which will quickly get infeasible.

However, in the process of fitting our GP we'll need to end up computing the inverse of the full covariance, :math:`\Sigma^{-1}` as well as what we've been calling the information vector, :math:`v = \Sigma^{-1} y`.  By using block inversion we get,

.. math::

    \Sigma^{-1} = \begin{bmatrix}
    \left(\hat{\Sigma} - S C^{-1} S^T\right)^{-1} & -\left(\hat{\Sigma} - S C^{-1} S^T\right)^{-1}SC^{-1} \\
    -\left(C - S^T \hat{\Sigma}^{-1} S\right)^{-1} S^T \hat{\Sigma}^{-1} & \left(C - S^T \hat{\Sigma}^{-1} S\right)^{-1}\end{bmatrix}

And if we break up :math:`v` into :math:`[\hat{v} \hspace{8pt} v_i]` using the same partitioning as :math:`y` we see,

.. math::

    v_i & = \left[\Sigma^{-1} y\right]_i \\
    & = \begin{bmatrix}
    -\left(C - S^T \hat{\Sigma}^{-1} S\right)^{-1} S^T \hat{\Sigma}^{-1} & \left(C - S^T \hat{\Sigma}^{-1} S\right)^{-1}
    \end{bmatrix} \begin{bmatrix} \hat{y} \\ y_i \end{bmatrix} \\
    & = -\left(C - S^T \hat{\Sigma}^{-1} S\right)^{-1} S^T \hat{\Sigma}^{-1} \hat{y} + \left(C - S^T \hat{\Sigma}^{-1} S\right)^{-1} y_i \\
    & = -A S^T \hat{\Sigma}^{-1} \hat{y} + A y_i

Where :math:`A = \left(C - S^T \hat{\Sigma}^{-1} S\right)^{-1}` is the lower right corner of :math:`\Sigma^{-1}` and :math:`A^{-1}` is the leave one out prediction covariance. Notice that if we multiply :math:`v_i` through by :math:`A^{-1}` we end up with,

.. math::

    A^{-1} v_i &= - S^T \hat{\Sigma}^{-1} \hat{y} + y_i \\
    &= -\mbox{E}[\mathbf{y_i}|\hat{y}] + y_i \\
    \mbox{E}[\mathbf{y_i}|\mathbf{\hat{y}}=\hat{y}] &= y_i - A^{-1} v_i

We can then recover the leave one out predictions,

.. math::

    [\mathbf{y_i}|\mathbf{\hat{y}}=\hat{y}] \sim \mathcal{N}\left(y_i - A^{-1} v_i, A^{-1}\right)

+++++++++++++++++++++++++++++++++
Computing :math:`A`
+++++++++++++++++++++++++++++++++

Above we see that if we can compute :math:`A` then we can recover the leave one out predictions without ever directly computing :math:`\hat{\Sigma}^{-1}`.  Take the case of leave one observation out, in this case :math:`A` will be the last diagonal value of :math:`\Sigma^{-1}`.  When training a Gaussian process we'll often have a decomposition of :math:`\Sigma` laying around, typically :math:`\Sigma = LDL^T`.  To get the :math:`i^{th}` diagonal value of :math:`\Sigma^{-1}` we can first compute, :math:`q = D^{-1/2} L^{-1} e_i`, where :math:`e_i` is a vector of zeros with a one in element :math:`i`, then we find that :math:`\Sigma^{-1}_{ii} = q^T q`.  Since :math:`L` is lower triangular and :math:`D` is diagonal :math:`p` can be computed efficiently.

Similarly if we're making leave one group out predictions we can build an indexing matrix :math:`E_i` which consists of columns :math:`e_j` for each :math:`j` in group :math:`i`.  Then we find that,

.. math::

    A = Q^T Q

with

.. math::

    Q = D^{-1/2} L^{-1} E_i.

Where :math:`L^{-1} E_i` amounts to extracting columns of :math:`L^{-1}`.

.. _`Gaussian Processes for Machine Learning`: http://gaussianprocess.org/gpml/chapters/RW.pdf