##########
1D Example
##########

.. _1d-example:

.. image:: https://raw.githubusercontent.com/swift-nav/albatross/master/examples/images/sinc_function_30.png
   :align: center

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

----------------
Model Definition
----------------

For illustrative purposes we'll assume we know nothing about the non linear
component other than that we think it's smooth.  To capture this with a Gaussian
process we can use the popular squared exponential covariance function which states
that the function value at two points :math:`x` and :math:`x'` will be similar
if the points are close and less similar the further apart they get.  More specifically
the covariance between values separated by a distance :math:`d = |x - x'|`
will be given by,

.. math::

   \mbox{sqr_exp}(d) = \sigma_c^2 \mbox{e}^{-\left(\frac{d}{\ell}\right)^2}.

Our first iteration may then say that the covariance between any two locations can be defined by

.. math::

   \mbox{cov}(x, x') = \mbox{sqr_exp}(|x - x'|) + \mbox{meas_noise}(x, x'),

or in other words we're saying our data comes from a smooth function and has some measurement noise
where the measurement noise is given by,

.. math::

   \mbox{meas_noise}(x, x') = \sigma^2 \mathbf{I}(x == x')

and :math:`\mathbf{I}(b)` evaluates to one if :math:`b` is true and zero otherwise.

We built this model in ``albatross`` (see next section) and used synthetic data to give
us an idea of how the resulting model would perform:

.. image:: https://raw.githubusercontent.com/swift-nav/albatross/master/examples/images/sinc_example_radial_only.png
   :align: center

From this plot we can see that the resulting model does a pretty good job of
capturing the non-linearity of the sinc function in the vicinity of training data,
including reasonable looking uncertainty estimates. However, you wouldn't want to
use this model to extrapolate outside of the training domain since the predictions quickly
return to the prior prediction of ``0``.

To improve the model's performance outside of the training domain we may want to
introduce a systematic term to the model.  For example instead of simply saying "the unknown
function is non-linear but smooth" we may want to say "the unknown function is linear 
plus a non-linear smooth component."  We can do this by introducing a polynomial term
into the covariance function.

More specifically we can define a covariance function which represents :math:`p(x) = a x + b`
by first placing priors on the values :math:`a` and :math:`b`,

.. math::

   a \sim \mathcal{N}(0, \sigma_a^2)

   b \sim \mathcal{N}(0, \sigma_b^2)

leading to the covariance function:

.. math::

   \mbox{linear}(x, x') = \sigma_a^2 x x' + \sigma_b^2.

Where the first term captures the covariance between :math:`x` and :math:`x'` from the linear component and second term captures the covariance from the common offset.

Now we can assemble this into a new covariance function, 

.. math::

   \mbox{cov}(x, x') = \mbox{linear}(x, x') + \mbox{sqr_exp}(|x - x'|) + \mbox{meas_noise}(x, x'),

create a Gaussian process from it and plot the resulting predictions which look like:

.. image:: https://raw.githubusercontent.com/swift-nav/albatross/master/examples/images/sinc_example_radial_with_linear.png
   :align: center

This plot shows that the model's ability to extrapolate has been been significantly improved.

-------------------------------
Implementation in albatross
-------------------------------

One of the primary goals of ``albatross`` is to make iterating on model formulations
like we did in the examples above easy and flexible.  All the components mentioned
earlier are already pre-defined.  The creation of basic Gaussian
process which consists of a squared exponential term plus measurement noise, for example,
can be constructed as follows,

.. code-block:: c

  IndependentNoise<double> independent_noise;
  SquaredExponential<EuclideanDistance> squared_exponential;
  auto model = gp_from_covariance(sqrexp + independent_noise);

Similarly we can build a model which includes the linear (polynomial) term using,

.. code-block:: c

  IndependentNoise<double> independent_noise;
  SquaredExponential<EuclideanDistance> squared_exponential;
  Polynomial<1> linear;
  auto model = gp_from_covariance(linear + sqrexp + independent_noise);

We can inspect the model and its parameters,

.. code-block:: c

    std::cout << covariance.pretty_string() << std::endl;

Which shows us,

.. code-block:: bash

    ((polynomal_1+independent_noise)+squared_exponential[euclidean_distance])
    {
        {"sigma_independent_noise", 1},
        {"sigma_polynomial_0", 100},
        {"sigma_polynomial_1", 100},
        {"sigma_squared_exponential", 5.7},
        {"squared_exponential_length_scale", 3.5},
    };

then condition the model on random observations, which we stored in ``data``,

.. code-block:: c

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

.. image:: https://raw.githubusercontent.com/swift-nav/albatross/master/examples/images/sinc_function_2.png
   :align: center

not great, but at least it knows it isn't great.  As we start to add more observations
we can watch the model slowly get more confident,

---------------
5 Observations
---------------

.. image:: https://raw.githubusercontent.com/swift-nav/albatross/master/examples/images/sinc_function_5.png
   :align: center

---------------
10 Observations
---------------

.. image:: https://raw.githubusercontent.com/swift-nav/albatross/master/examples/images/sinc_function_10.png
   :align: center

---------------
30 Observations
---------------

.. image:: https://raw.githubusercontent.com/swift-nav/albatross/master/examples/images/sinc_function_30.png
   :align: center

