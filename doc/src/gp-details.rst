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

    [\mathbf{f^*}|\mathbf{f} = \mathbf{y}] \sim \mathcal{N}\left(K_{*f} K_{ff}^{-1} \mathbf{y}, K_{**} - K_{*f} K_{ff} K_{f*}\right)
    
for brevity we'll often drop the :math:`\mathbf{f} = \mathbf{y}` part and just write:

.. math::

    \mathbf{f^*}|\mathbf{y} \sim \mathcal{N}\left(K_{*f} K_{ff}^{-1} \mathbf{y}, K_{**} - K_{*f} K_{ff} K_{f*}\right)

This would read, "the posterior distribution, :math:`\mathbf{f^*}`, at locations :math:`\mathbf{x^*}`  given :math:`\mathbf{y}` is multivariate normal with the given mean and covariance".

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
    P^TLDLP &= K_{ff} \\
    \mathbf{v} &= K_{ff}^{-1} \mathbf{y}

-----------------------
Predictive Distribution
-----------------------

Once we've fit a model we can use it to make a prediction at arbitrary locations (read: features), :math:`\mathbf{x}^*`,

.. math::

    \mathbf{f^*}|\mathbf{y} \sim \mathcal{N}\left(K_{*f} K_{ff}^{-1} \mathbf{y}, K_{**} - K_{*f} K_{ff} K_{f*}\right)

We can take advantage of some of the precomputed quantities to make this prediction step more efficient.  In particular we would write this,

.. math::

    \mathbf{f^*}|\mathbf{y} & \sim \mathcal{N}\left(K_{*f} \mathbf{v}, K_{**} - K_{*f} (P^TLDL^TP)^{-1} K_{f*}\right) \\
    & \sim \mathcal{N}\left(K_{*f} \mathbf{v}, K_{**} - (K_{*f} P^T L^{-T} D^{-1/2}) (D^{-1/2}L^{-1}P K_{f*}\right) \\
    & \sim \mathcal{N}\left(K_{*f} \mathbf{v}, K_{**} - Q_{f*}^T Q_{f*}\right)
    
Where :math:`Q_{f*} = D^{-1/2}L^{-1}P K_{*f}^T`.

So to put this into distinct steps to compute the posterior mean we would:

* Evaluate :math:`K_{*f}`
* Compute the mean :math:`K_{*f} \mathbf{v}`

To compute the posterior ``MarginalDistribution`` we would then:

* Compute :math:`Q_{f*} = D^{-1/2}L^{-1}P K_{*f}^T`  
* Evaluate the prior variance :math:`\mbox{diag}(K_{**})`
* Compute the posterior variance :math:`\mbox{diag}(K_{**}) - \mbox{diag}(Q_{f*}^T Q_{f*})`
  
and to compute the posterior ``JointDistribution`` we would instead compute:

* Evaluate the prior covariance :math:`K_{**}`
* Compute the posterior covariance :math:`K_{**} - Q_{f*}^T Q_{f*}`

