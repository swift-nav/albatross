####################
Datasets
####################

.. _datasets:

-------------------
Regression Datasets
-------------------

One of the core data types in ``albatross`` is the ``RegressionDataset``.  These objects are built to store a set of measurements (or predictions) and are used throughout albatross.  There are a number of additional helper methods in the ``RegressionDataset``, but you can think of it as a simple struct:

.. code-block:: c

  template <typename FeatureType>
  struct RegressionDataset {
    std::vector<FeatureType> features;
    MarginalDistribution targets;
  }

In the majority of literature and Gaussian process tutorials online you'll see :math:`x` are points in time or space. As a result they're often refered to as locations. In spatio-temporal processes :math:`x` could be a single floating point value (like you'd see in temproal modelling) or a pair of values (like you'd see in spatial modelling), but albatross was designed with the hope that the same dataset structures could be used to train and test with arbitrary models so we use variable names which are more in line with machine learning community. Consider building a model :math:`m` which takes inputs :math:`x` and produces an output :math:`y`,

.. math::

  y \leftarrow m(x)
  
In albatross the inputs, :math:`x`, are "features" which can be thought of as the characteristics of a measurement. They may contain pre-processed quantities which capture all information required to fit a model. Instead of "measurements" we call :math:`y` the  "target". The target is the output which corresponds to an input feature. During training the targets are usually measurements (and measurement noise), but during prediction they're the output of the model.  In regression problems the targets will be floating point values, but the term target generalizes well to classification as well.

Each dataset consists of a set of features and corresponding targets. Each feature fully describes a target (measurement) and each target consists of the actual value and an optional variance.  These targets are held in a ``MarginalDistribution`` which looks something like this:

.. code-block:: c

  struct MarginalDistribution {
    Eigen::VectorXd mean;
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> variance;
  }


The :ref:`temperature dataset <temperature-example>` is a good example of how data is stored.  That dataset consists of a large number of average daily temperature measurements at stations spread across the continental US.  In this case we need to know things about where each measurement was taken which we store in a ``Station`` struct,

.. code-block:: c

  struct Station {
    double lat;
    double lon;
    double height;
  }

The full dataset is then a ``RegressionDataset<Station>`` and we could do things like print the ``i`` th temperature measurement and where it was observed:

.. code-block:: c

  const RegressionDataset<Station> temperature_dataset = make_dataset();
  std::cout << temperature_dataset.features[i].lat << " ";
  std::cout << temperature_dataset.features[i].lon << " ";
  std::cout << temperature_dataset.targets.mean[i] << std::endl;

There are some useful utility functions for manipulating these datasets.  We can subset the dataset,

.. code-block:: c

  std::vector<std::size_t> first_few_odd_indices = {1, 3, 5, 7};
  const RegressionDataset<Station> first_few_odd =
          subset(temperature_dataset, first_few_odd_indices);

Or concatenate two datasets:

.. code-block:: c

  RegressionDataset<Station> ca = make_california_dataset();
  RegressionDataset<Station> ny = make_new_york_dataset();
  RegressionDataset<Station> both = concatenate_datasets(ca, ny);

You can actually concatenate datasets of different types as well, which will result in a new dataset which uses mapbox ``variant`` to store the combined types.

.. code-block:: c

  RegressionDataset<Station> ca = make_california_dataset();
  RegressionDataset<int> constraints = make_constraint_dataset();
  RegressionDataset<variant<Station, int>> both = concatenate_datasets(ca, constraints);

Finally, if you've written serialization routines for the types involved (see `cereal`_ and ) you can then dump your dataset to csv:

.. code-block:: c

  std::ofstream ofs("example.csv");
  write_to_csv(ofs, temperature_dataset);

and if you've defined ``<<`` stream operators for ``Station`` you'd be able to dump it to ``std::cout`` for debug purposes,

.. code-block:: c

  std::cout << temperature_dataset << std::endl;


.. _`cereal` : https://uscilab.github.io/cereal/

