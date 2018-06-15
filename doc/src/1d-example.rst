##########
1D Example
##########

.. _1d-example:

--------------
Introduction
--------------

Regression problems typically involve estimating some unknown function :math:`f` from a set of noisy data :math:`y`.  As an example we can attempt to estimate a function :math:`f(x)`,

.. math::

    f(x) = a x + b + \mbox{something_nonlinear}(x)

based off of noisy observations,

.. math::

    y = f(x) + \mathcal{N}(0, \sigma^2).

For this example we'll use the ``sinc`` function for the non linear portion,

.. math::

     f(x) = a x + b + c \frac{\mbox{sin}(x)}{x}

but for illustrative purposes we'll assume we know nothing about the non linear
component other than that we think it's smooth.  To capture this with a Gaussian
process we may want to include a systematic offset and linear component and a component
which adds a soft constraint that the function value for neighboring points will be similar.

More specifically we can define our priors on the parameters,

.. math::

   a \sim \mathcal{N}(0, \sigma_a^2)

   b \sim \mathcal{N}(0, \sigma_b^2)

For the nonlinear portion we'll use a prior which states that two points
:math:`x` and :math:`x'` which are separated by a distance :math:`d = x - x'`
will have a covariance given by,

.. math::

   \mbox{sqr_exp}(d) = \sigma_c^2 \mbox{e}^{-\left(\frac{d}{\ell}\right)^2}.

We can then define the covariance function which captures the constant, linear, nonlinear and measurement noise components,

.. math::

   \mbox{cov}(x, x') = \sigma_a^2 x x' + \sigma_b^2 + \mbox{sqr_exp}(x - x') + \sigma^2 \mathbf{I}(x = x').

Where the first term captures the covariance between :math:`x` and :math:`x'` from the linear component.  The second term captures the covariance from the common offset.  The third term is provides the flexibility for non linear functions and the fourth term captures the measurement noise through the use of the indicator function, :math:`\mathbf{I}(\cdot)`, which takes on a value of :math:`1` if the argument is true.

-------------------------------
Implementation in ``albatross``
-------------------------------

Using ``albatross`` this would look like,

.. code-block:: c

  using Noise = IndependentNoise<double>;
  using SqrExp = SquaredExponential<EuclideanDistance>;

  CovarianceFunction<Constant> mean = {Constant(100.)};
  CovarianceFunction<SlopeTerm> slope = {SlopeTerm(100.)};
  CovarianceFunction<Noise> noise = {Noise(meas_noise)};
  CovarianceFunction<SqrExp> sqrexp = {SqrExp(2., 5.)};

  auto covariance = mean + slope + noise + sqrexp;

which incorporates prior knowledge that the function consists of a mean offset, a linear term, measurement noise and an unknown smooth compontent (which we captures using a squared exponential covariance function).

We can inspect the model and its parameters,

.. code-block:: c

    std::cout << covariance.to_string() << std::endl;

Which shows us,

.. code-block:: bash

    model_name: (((constant+slope_term)+independent_noise)+squared_exponential[scalar_distance])
    model_params:
      length_scale: 2
      sigma_constant: 100
      sigma_independent_noise: 1
      sigma_slope: 100
      sigma_squared_exponential: 5

then condition the model on random observations, which we stored in ``data``,

.. code-block:: c

  auto model = gp_from_covariance(covariance);
  model.fit(data);

and make some gridded predictions,

.. code-block:: c

  const int k = 161;
  const auto grid_xs = uniform_points_on_line(k, low - 2., high + 2.);
  const auto predictions = model.predict(grid_xs);

Here are the resulting predictions when we have only two noisy observations,

---------------
2 Observations
---------------

.. image:: https://raw.githubusercontent.com/swift-nav/albatross/master/examples/sinc_function_2.png
   :align: center

not great, but at least it knows it isn't great.  As we start to add more observations
we can watch the model slowly get more confident,

---------------
5 Observations
---------------

.. image:: https://raw.githubusercontent.com/swift-nav/albatross/master/examples/sinc_function_5.png
   :align: center

---------------
10 Observations
---------------

.. image:: https://raw.githubusercontent.com/swift-nav/albatross/master/examples/sinc_function_10.png
   :align: center

---------------
30 Observations
---------------

.. image:: https://raw.githubusercontent.com/swift-nav/albatross/master/examples/sinc_function_30.png
   :align: center

