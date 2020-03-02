####################
Gaussian Processess
####################

.. _gp:

Here we describe how to build and work with Gaussian processes in ``albatross``, this assumes a basic understanding of what Gaussian processes are.  The python package `scikit learn`_ has a good practical introduction and for a complete theoretical explanation the book `Gaussian Process Regression`_ is an excellent resource.  It could also be worth going through the :ref:`1d example<1d-example>` and the :ref:`temperature example <temperature-example>` to get a general idea of how Gaussian processes can be applied.  In this section we'll focus on how to build a Gaussian process in ``albatross``, in particular how to create covariance functions, build a GP from them and how to use the model to make predictions.

---------------
Basic Workflow
---------------

TLDR; Here is an example work flow in which we create a GP, tune its parameters to maximize the leave one out cross validated likelihood, then fit the model and use it to make predictions of unobserved data.

.. code-block:: c

  RegressionDataset<double> training_data = make_training_data();
  RegressionDataset<double> evaluation_data = make_evaluation_data();

  const IndependentNoise<double> independent_noise;
  const SquaredExponential<EuclideanDistance> squared_exponential;
  const auto covariance = squared_exponential + independent_noise;
  auto gp = gp_from_covariance(covariance);

  albatross::LeaveOneOutLikelihood<> loo_nll;
  const auto tuned_params = get_tuner(gp, loo_nll, training_data).tune();
  gp.set_params(tuned_params);

  const auto fit_model = gp.fit(training_data);
  const Eigen::VectorXd prediction = fit_model.predict(evaluation_data.features).mean();  
  
  const double rmse = root_mean_square_error(prediction, evaluation_data.targets);

  std::cout << "Evaluation RMSE: " << rmse << std::endl;


.. _`scikit learn`: https://scikit-learn.org/stable/modules/gaussian_process.html
.. _`Gaussian Process Regression`: http://www.gaussianprocess.org/gpml/chapters/RW2.pdf

--------------------
Covariance Functions
--------------------

A Gaussian process (GP) is defined by its covariance function (and occasionally by a mean function).  The covariance function is responsible for describing the relationship between any two variables and is used to build a full description of the training data and subsequently describe how the test data depends on the training data which can be used to make predictions at un-observed locations.  

To start, here is how one would build a very basic but functional covariance function which works with one dimensional data,
 
.. code-block:: c

  IndependentNoise<double> independent_noise;
  SquaredExponential<EuclideanDistance> squared_exponential;
  const auto covariance = squared_exponential + independent_noise;

The resulting variable ``covariance`` will be be derived from the ``albatross::CovarianceFunction<>`` type which contains a large number of helper methods.  In particular you can treat the resulting type as callable object which will return the covariance between any two data points:

.. code-block:: c

  double x = 0.;
  double y = 1.;
  double cov = covariance(x, y);

Similarly you can compute the covariance matrix between all points in a vector,

.. code-block:: c

  std::vector<double> points = {0., 1., 2.};
  // Creates a 2 x 2 matrix
  Eigen::MatrixXd cov = covariance(points);
  
  std::vector<double> other_points = {4., 5.};
  // Creates a 3 x 2 matrix
  Eigen::MatrixXd cross_cov = covariance(points, other_points);

Covariance functions often depend on parameters.  In this case we have the ``independent_noise`` function which will have a parameter representing the magnitude of the measurement noise.  We can inspect it using ``get_params()``:

.. code-block:: c

    std::cout << pretty_params(independent_noise.get_params()) << std::endl;

.. code-block:: bash

    {
        {"sigma_independent_noise", 1},
    };

Notice that the sum of ``independent_noise`` and ``squared_exponential`` will consist of the concatenation of both their params,

.. code-block:: c

    std::cout << pretty_params(covariance.get_params()) << std::endl;

Which would result in,

.. code-block:: bash

    {
        {"sigma_independent_noise", 1},
        {"sigma_squared_exponential", 5.7},
        {"squared_exponential_length_scale", 3.5},
    };

++++++++++++++
Operators
++++++++++++++

We already saw how you can sum covariance functions together to get a new function, but you can also take the product,

.. code-block:: c

  auto sum = foo + bar;
  auto prod = foo * bar;

One situation where you may want to use the product of two covariance functions is when you want to decorrelate
what would otherwise be correlated terms.  For example, when dealing with spatial and temporal data
(such as the :ref:`temperature example <temperature-example>`) you may want a term (``spatial``) which
says "Nearby locations will have a similar temperature" and another term (``temporal``) which says
"Temperature changes over the course of time". Which could be combined into
another covariance function (``spatio_temporal = spatial * temporal``) which says,
"Measurements taken at similar locations and times will be similar."

+++++++++++++++++++++++++++++
Writing Your Own
+++++++++++++++++++++++++++++

The covariance functions in ``albatross`` use the Curiously Recurring Template Pattern (`CRTP`_) which makes defining them slightly different from the standard inheritence pattern in C++.  For example, to write your own simple covariance function you could start with a definition such as,

.. _`CRTP`: https://www.fluentcpp.com/2017/05/12/curiously-recurring-template-pattern/

.. code-block:: c

  class Simple : public CovarianceFunction<Simple> {
   public:
    double _call_impl(const X &x, const X &other) const {
      return 1.;
    }
  }

The resulting covariance function will be callable with any arguments that are of type ``X`` but will otherwise result in a compile time failure:

.. code-block:: c

  Simple simple;
  X x;
  Y y;
  // this is fine:
  double xx = simple(x, x);
  // this would fail to compile: 
  double xy = simple(x, y);

Notice that by defining a ``_call_impl`` method in your covariance function the base class enabled the corresponding call operator(s).  This is the primary reason for the use of CRTP, namely the ``CovarianceFunction<Derived>`` base class is capable of inspecting the ``Derived`` class and enabling methods such as the ``operator()`` depending on which ``_call_impl`` methods have been defined.  This next example is not actually valid C++, but it might help to think of the ``CovarianceFunction`` class as an abstract class with signature.

.. code-block:: c

  class CovarianceFunction {
   public:
    template <typename X, typename Y>
    virtual double _call_impl(const X &, const Y &) const = 0

    template <typename X, typename Y>
    double operator()(const X &x, const Y &y) const {
      return this->_call_impl(x, y);
    }
  }

Covariance functions can be parametrized, there are several ways to accomplish this
but the ``ALBATROSS_DECLARE_PARAMS`` is likely your best bet:

.. code-block:: c

  class Simple : public CovarianceFunction<Simple> {

    ALBATROSS_DECLARE_PARAMS(simple_sigma);

    Simple(const double &sigma) {
      simple_sigma = {sigma, PositivePrior()};
    }

   public:
    double _call_impl(const X &x, const X &other) const {
      return simple_sigma.value * simple_sigma.value;
    }
  }

any parameters you define will then be gettable and settable using the ``get_params()`` and ``set_params()``
methods (as well as a number of other helper methods) in both the covariance function itself
and any compositions including it as well as any Gaussian processes which use it.  Also worth noting
that there are a number of other `priors`_ you can choose from.

If you are writing your own covariance functions you might find it helpful to take a look at some
of the examples and the `predefined covariance functions`_.

CRTP definitely adds to the complexity but it enables some of the most powerful
features in ``albatross``; the ability for covariance functions to work with arbitrary custom types and the composition of covariance functions through ``+`` and ``*`` operators.

.. _`predefined covariance functions`: https://github.com/swift-nav/albatross/tree/master/include/albatross/src/covariance_functions
.. _`priors`: https://github.com/swift-nav/albatross/blob/master/include/albatross/src/core/priors.hpp

+++++++++++++++++++++
Multiple Types
+++++++++++++++++++++

Covariance functions are not restricted to work with a single type, in fact
this is one of the more powerful features in ``albatross``.
For example you could write a ``CovarianceFunction`` like this:

.. code-block:: c

  class Both : public CovarianceFunction<Both> {

    double _call_impl(const X &x, const X &other) const {
      return 3.;
    }

    double _call_impl(const X &x, const Y &y) const {
      return 5.;
    }

    double _call_impl(const Y &y, const Y &other) const {
      return 7.;
    }

  }

Which we can then sum together with ``Simple`` and the behavior changes,

.. code-block:: c

  Simple simple;
  Both both;
  auto sum = simple + both;

  sum(x, x) // 4.
  sum(x, y) // 5.
  sum(y, y) // 7.

Once you've defined a covariance function you can also call it with a ``variant``,

.. code-block:: c

  variant<X, Y> vx = x;
  variant<X, Y> vy = y;

  sum(x, x) == sum(vx, vx);
  sum(x, y) == sum(vx, vy);
  sum(y, y) == sum(vy, vy);

------------------------------
Auto Is Your Friend
------------------------------

One of the drawbacks to CRTP is that the resulting types can be extremely verbose.  Take the example above and note the use of ``auto sum = simple + both``.  The actual type of ``sum`` in this case would be ``SumOfCovarianceFunctions<Simple, Both>``.  Not too bad, but you can see how if you begin building covariance functions with multiple terms you quickly end up with very complicated types.  Thankfully the use of ``auto`` should keep you from ever needing to actually know the underlying type.




