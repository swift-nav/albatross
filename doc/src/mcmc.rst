########################
Markov Chain Monte Carlo
########################

.. _mcmc:

Albatross has an implementation of an ensemble Bayesian sampler based off the paper `Ensemble samplers with affine invariance`_
which was inspired by the `emcee`_ package in python.

The idea is that by defining a function which computes the likelihood (or something proportional to the likelihood)
you can then produce samples from the distribution of parameters.  To do so the algorithm iteratively proposes new
parameters and accepts the new proposed parameters based off the ratio of the current parameters likelihood with the
proposed parameters likelihood.

Here is how you could run the ensemble sampler for some arbitrary likelihood function,

.. code-block:: c

  auto proportional_to_ll(const std::vector<double> &params) {
    // compute and return a value proportional to the likelihood of the params
  }

  std::default_random_engine gen(0);
  std::vector<std::vector<double>> initial_params = get_initial_params();
  std::size_t max_iterations = 1000;
  const auto ensemble_samples =
      ensemble_sampler(proportional_to_ll, initial_params, max_iterations, gen);

There are some helper functions when dealing directly with albatross models,

.. code-block:: c

  std::default_random_engine gen(0);
  std::size_t walters = 32;
  std::size_t max_iterations = 1000;
  auto csv_callback = get_csv_writing_callback(model, "path_to_params.csv");
  const auto samples = ensemble_sampler(model, dataset, walkers, max_iterations,
                                        gen, csv_callback);

In this case the initial parameters will be directly extracted fro mthe model, and as the sampler runs the output
will get stored in a csv file which can then be inspected using ``examples/plot_sampler_output.csv``.


.. _`Ensemble samplers with affine invariance`: https://ui.adsabs.harvard.edu/abs/2010CAMCS...5...65G/abstract

.. _`emcee`: https://emcee.readthedocs.io/en/stable/
