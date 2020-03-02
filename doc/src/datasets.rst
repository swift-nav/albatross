####################
Datasets
####################

.. _datasets:

-------------------
Regression Datasets
-------------------

One of the core data types in ``albatross`` is the ``RegressionDataset``.  These objects are built to store a set of measurements which we use all over albatross.  There are a number of additional helper methods in the ``RegressionDataset``, but you can think of it as a simple struct:

.. code-block:: c

  template <typename FeatureType>
  struct RegressionDataset {
    std::vector<FeatureType> features;
    MarginalDistribution targets;
  }


Here we see that there is a vector of features.  Each feature is meant to be an object which fully describes a measurement and each measurement consists of the actual value and an optional variance.  These measurements are held in a ``MarginalDistribution`` which looks something like this:

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

.. _`cereal` : https://uscilab.github.io/cereal/

