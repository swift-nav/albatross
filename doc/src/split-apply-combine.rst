##############################
Group By : Split Apply Combine
##############################

.. _split-apply-combine:

The split apply combine workflow is something that is heavily used in the popular
python package `pandas`_ and which we've borrowed in ``albatross``.  The general
idea is that many data manipulation operations can be broken into three steps in
which a dataset is first split apart into groups, a function is then applied to
each of the groups and the result is recombined into a new dataset.

This technique can be used in ``albatross`` using syntax that ends up very similar
to that of ``pandas``

.. code-block:: c

  dataset.group_by(my_criteria).apply(some_operation).combine();

.. _`pandas` : https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html

-----------------
Group By
-----------------

The grouping (and applying) can be done using anything callable which takes
a single ``FeatureType`` as an argument, for example:

.. code-block:: c

  int my_criteria(const double &x) { return round(x); }

  struct MyCritera {
    int operator() (const double &x) {
      return my_criteria(x);
    }
  }

  void examples() {
    RegressionDataset<double> dataset = make_dataset();

    // using a free function
    dataset.group_by(my_criteria);
    // using a lambda
    dataset.group_by([](const auto &x){return my_criteria(x);});
    // using a callable object
    MyCritera criteria;
    dataset.group_by(criteria);
  }

The result of ``group_by(f).groups()`` can be treated as if it were a ``std::map``,
for example to get the third group:

.. code-block:: c

  auto dataset_3 = dataset.group_by(my_criteria).groups()[3];

Not all operations can be efficiently done working with the grouped datasets, for these
cases you may find it helpful to work directly with the indices of each group:

.. code-block:: c

  std::vector<std::size_t> inds_3 = dataset.group_by(my_criteria).indexers()[3];

The ``group_by`` technique can also be used directly on vectors:

.. code-block:: c

    std::vector<double> values = make_values();
    group_by(values, my_criteria);

In this case the ``groups()`` will return a map from ``int`` to ``std::vector<double>``.

-----------------
Apply
-----------------

Similar to ``group_by`` an apply function can be anything callable and should take
either the key value pair or just the value as arguments and can (optionally) return
a new object.  In other words an apply function should have one of the following signatures.

``ApplyType f(KeyType &key, ValueType &value)``
    The result will be a new map-like from ``KeyType`` to ``ApplyType``

.. code-block:: c

  auto sum = [](const int &key, const RegressionDataset<double>& value)
    { return value.targets.mean.sum(); };
  std::map<int, double> sums = dataset.group_by(my_criteria).apply(sum);

``void f(KeyType &key, ValueType &value)``
    The return type will be void.

.. code-block:: c

  auto print_sum = [](const int &key, const RegressionDataset<double>& value)
    { std::cout << key << " : " << value.targets.mean.sum() << std::endl; };
  dataset.group_by(my_criteria).apply(print_sum);

``ApplyType f(ValueType &value)``
    The result will be a new map-like from ``KeyType`` to ``ApplyType``

.. code-block:: c

  auto sum = [](const RegressionDataset<double>& value)
    { return value.targets.mean.sum(); };
  std::map<int, double> sums = dataset.group_by(my_criteria).apply(sum);

``void f(KeyType &key, ValueType &value)``
    The return type will be void.

.. code-block:: c

  auto print_sum = [](const RegressionDataset<double>& value)
    { std::cout << value.targets.mean.sum() << std::endl; };
  dataset.group_by(my_criteria).apply(print_sum);

For example, we could do something like:

.. code-block:: c

  RegressionDataset<Bar> dataset;
  auto get_foo = [](const Bar &bar) { return Foo(bar); };
  dataset.group_by(get_foo).apply(f);

In this situation the ``ValueType = RegressionDataset<Bar>`` and ``KeyType = Foo``.

``auto`` can be used for the argument types in which case a single argument is assumed
to be a ``ValueType``.  For example,

.. code-block:: c

  dataset.group_by(get_foo).apply([](const auto &data) {return f(data);});

-----------------
Combine
-----------------

In the apply step there are very few restrictions on what can be returned from an apply
function.  When it comes to the combine step however, there are a few restrictions.  Namely
combine only supports ``RegressionDataset<>``, ``std::vector<>`` and ``double`` types.

In this example you can see how you could start with a dataset, split it into groups
compute some metric for each group and recombine into a vector of the results:

.. code-block:: c

  auto compute_something = [](const RegressionDataset<Bar> &data) -> double {
    double something = data.features[0].foo;
    return something;
  }

  Eigen::VectorXd results = dataset.group_by(get_group).apply(compute_something).combine();


--------------------
Motivational Example
--------------------

One common pattern when working with data is the need to break a dataset apart and
do something with each of the resulting groups.  For example, in the ``group_by_example``
we built a dataset which contains a bunch of people defined by their age and gender:

.. code-block:: c

  struct Person {
    enum Gender {FEMALE, MALE};

    Gender gender;
    int age;
  };

In ``albatross`` we store data using the ``RegressionDataset<>`` type which consists
of a vector of features and an ``Eigen::VectorXd`` of targets.  You can think of the
features as an object containing all the information you need to describe some measurement
and the ``targets`` as containing the actual measurements.

We might then, for example, want to take our dataset of people and print out the
average salary depending on the gender.  Here's how you might do that manually:

.. code-block:: c

  std::size_t female_count = 0;
  double female_average = 0.;
  std::size_t male_count = 0;
  double male_average = 0.;

  for (std::size_t i = 0; i < dataset.size(); ++i) {
    if (dataset.features[i].gender == Person::Female) {
      female_average += dataset.targets.mean[i];
      ++female_count;
    } else {
      male_average += dataset.targets.mean[i];
      ++male_count;
    }
  }

  female_average /= female_count;
  male_average /= male_count;

  std::cout << "female : " << female_average << std::endl;
  std::cout << "male : " << male_average << std::endl;


There are several issues with this though.  If there are no males (or females) in the
dataset we'll end up dividing by zero.  Also, if you are dealing with more than two
options the details of the for loop (which are already a bit difficult to follow)
could get very complicated. Instead we can use the ``group_by`` and ``apply`` methods
to come up with an alternative approach:

.. code-block:: c

  const RegressionDataset<Person> dataset = make_data();

  auto get_gender = [](const auto &f){return f.gender;};

  auto print_average_salary = [](const auto &gender, const auto &dataset) {
    std::cout << to_string(gender) << "  :  " << dataset.targets.mean.mean() << std::endl;
  };

  dataset.group_by(get_gender).apply(print_average_salary);

Not only will this avoid the pitfall of missing groups, but the split-apply approach forces the use of smaller helper functions which ends up making everything much easier to read.

