##########
Tuning
##########

.. _tuning:

So you've built a model, but somewhere in the process it required adding some nuisance parameters.
Sometimes these parameters have realtively well known values, but often that's not the case.  In
such situations it's common to tune (or optimize) the parameters in your model.

Here is an example of how you could tune your model's parameters to maximize the leave one out
cross validated likelihood,

.. code-block:: c
  
  LeaveOneOutLikelihood<MarginalDistribution> loo_nll;
  const auto tuner = get_tuner(model, loo_nll, data);
  auto tuned_params = tuner.tune();
  model.set_params(params);

Tuning in ``albatross`` piggy backs off of `nlopt`_.  There are a lot of ways to customize the
tuning process most of which can be accomplished by accessing the ``tuner.optimizer`` member.

.. _`nlopt`: https://nlopt.readthedocs.io/en/latest/

+++++++++++
Priors
+++++++++++

Model parameters often have priors, you may know for example that one of them is always between
0 and 1.  Others may be strictly positive.  This sort of information can all get stored in an
albatross model.  Here, for example, is how you might define your model to expose the parameters,

.. code-block:: c

  class MyModel : public ModelBase<MyModel> {
    MyModel() {
      params_["foo"] = {0.5, UniformPrior(0., 1.)};
      params_["bar"] = {10, PositivePrior()};
    }
  }

If you've properly defined the parameters in your model (and picked a compatible ``nlopt`` algorithm)
tuning will respect the parameter bounds.  Metrics such as the ``LeaveOneOutLikelihood`` will also
include the likelihood of the parameters.
