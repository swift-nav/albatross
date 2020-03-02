#############
Custom Models
#############

.. _custom-models:

----------------
Defining a Model
----------------

The ``ModelBase`` class in ``albatross`` uses the Curiously Recurring Template Pattern (`CRTP`_)
which makes defining them slightly different from the standard inheritence pattern in C++.

.. _`CRTP`: https://www.fluentcpp.com/2017/05/12/curiously-recurring-template-pattern/

In general an albatross model requires defining ``_fit_impl``, ``_predict_impl`` and a
struct ``Fit<MyModel>`` which is in charge of storing any coefficients.

----------------
Fit Method
----------------

To get `model.fit(dataset)` to work you need to add a ``_fit_impl`` method to your class,
this fit implementation needs to take a vector of features and corresponding targets
(often measurements) and needs to return a ``Fit<ModelType>`` object holding any
information required to make predictions.

.. code-block:: c

  class ModelType : public ModelBase<ModelType> {

    Fit<ModelType> _fit_impl(const std::vector<FeatureType> &features,
                             const MarginalDistribution &targets) const;

  }


The ``FeatureType`` here can be which ever type your problem requires and
you can have multiple ``_fit_impl`` methods for different types in the same model.
Templated ``_fit_impl`` methods also work.

----------------
Predict Method
----------------

To get `model.predict(features)` to work you need to add a ``_predict_impl`` method to your class,
this predict implementation needs to take a vector of features and a ``Fit<ModelType>`` and
needs to return either an ``Eigen::VectorXd`` (mean only),
``MarginalDistribution`` (mean and variance) or ``JointDistribution`` (mean and covariance).

.. code-block:: c

  class ModelType : public ModelBase<ModelType> {

    JointDistribution _predict_impl(const std::vector<FeatureType> &features,
                                  const Fit<ModelType> &fit,
                                  PredictTypeIdentity<JointDistribution>) const;

  }

In this case above we've implemented predict to return a ``JointDistribution`` which holds
the mean prediction as well as a full covariance.  A ``JointDistribution`` can be converted
into a ``MarginalDistribution`` by taking the diagonal of the covariance matrix and a ``MarginalDistribution``
can be converted into a mean only prediction (``Eigen::VectorXd``) by simply taking the mean of the distribution.
As a result, by implementing predict for a ``JointDistribution`` you will be able to call
all of the following.

.. code-block:: c

  const auto prediction model.fit(dataset).predict(features);
  JointDistribution joint_pred =prediction.joint();
  MarginalDistribution marginal_pred = prediction.marginal();
  Eigen::VectorXd mean_pred = prediction.mean();

If you define ``_predict_impl`` with a ``MarginalDistribution`` instead, then you'll find
that you can call,

.. code-block:: c

  MarginalDistribution marginal_pred = prediction.marginal();
  Eigen::VectorXd mean_pred = prediction.mean();

but calling ``prediction.joint();`` would result in a compile time error.  Similarly if you just define the mean only version 
then asking for anything other than ``prediction.mean()`` will result in a compile time error.

We saw above that you could implement the ``JointDistribution`` version and have access to all the predict types,
but that is often inefficient.  Instead you may want to impelement specialized version for each of
the predict types.  This is what is done in for the Gaussian processes (see ``gp.hpp``).  The
desire to have specialized predict types is what led to the mysterious ``PredictTypeIdentity<>`` argument,
which is required to allow overridable ``_predict_impl`` methods with different return types.

----------------
Fit Type
----------------

The fit type needs to be a specialization of the ``Fit<>`` struct.  The idea is that by forcing the
output of ``_fit_impl`` to be a custom type we can subsequently make model types constant, which
gives us peace of mind that there isn't accidentally some state that get's stored in a model which
would cause two calls to ``fit`` to produce different results.

Once you've defined the ``Fit<>`` you shouldn't ever need to actually inspect that type, that
should be left to the internals of ``albatross``.  Instead you are encouraged to use ``auto``,

.. code-block:: c

  const auto fit_model = model.fit(dataset);

or write everything as one liners.

.. code-block:: c

  const Eigen::VectorXd mean = model.fit(dataset).predict(features).mean();

Here's an illustration of the actual types that would result from a typical model
workflow:

.. code-block:: c

  const ModelType model = make_my_model();
  const FitModel<ModelType, Fit<ModelType>> fit_model = model.fit(dataset);
  const Prediction<ModelType, FeatureType, Fit<ModelType>> prediction = fit_model.predict(features);
  const JointDistribution joint_prediction = prediction.joint();

Again, thanks to ``auto`` type declarations you shouldn't need to actually know these types
but it may be helpful to get a glimpse of what's happening under the hood.  This chain of
types is what allows ``albatross`` to keep track of how exactly you're using a model and
decide (at compile time) the most efficient methods to use.

----------------
Example
----------------

Here's an example of a model which always returns the mean of the training data.

.. code-block:: c

  struct Fit<MeanModel> {
    double mean;
  }

  class MeanModel : public albatross::ModelBase<MeanModel> {
   public:

    using FitType = Fit<MeanModel>;

    std::string get_name() const { return "mean"; }

    template <typename FeatureType>
    FitType _fit_impl(const std::vector<FeatureType> &features,
                      const MarginalDistribution &targets) const {
      FitType model_fit = {targets.mean.mean()};
      return model_fit;
    }

    template <typename FeatureType>
    Eigen::VectorXd _predict_impl(const std::vector<FeatureType> &features,
                                  const FitType &fit,
                                  PredictTypeIdentity<Eigen::VectorXd>) const {
      Eigen::VectorXd output(features.size());
      output.fill(fit.mean);
      return output;
    }
  }


While defining your own model isn't as simple as standard inheritence
, the benefits are large.  Once you've defined a model using the ``ModelBase`` class you can immediately start using all the
tools built around it, things such as :ref:`cross validation <crossvalidation>`, outlier detection using RANSAC,
and tuning :ref:`tuning <tuning>`.


