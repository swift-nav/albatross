=====================================
albatross
=====================================
|Build Status|

A framework for statistical modelling in C++, with a focus on Gaussian processes.

.. _`albatross`: https://travis-ci.com/swift-nav/albatross

.. |Build Status| image:: https://travis-ci.com/swift-nav/albatross.svg?token=ZCoayM24vorooTuykqeC&branch=master

***********
Features
***********
 * `Gaussian Process Regression`_ which is accomplished using composable covariance functions and templated predictor types.
 * Evaluation utilities, with a focus on cross validation.
 * Written using generics in an attempt to make these core routines applicable to a number of fields.
 * Parameter handling which makes it easy to get and set parameters in a standardized way  as well as compose and (de)serialize models to string.

.. _`Gaussian Process Regression`: http://www.gaussianprocess.org/gpml/chapters/RW2.pdf

**********
Examples
**********

- :ref:`A One dimensional example using a sinc function. <1d-example>`
- :ref:`An example spatial model for temperature estimation. <temperature-example>`

.. image:: https://github.com/swift-nav/albatross/raw/akleeman/temperature-example/examples/temperature_example/mean_temperature.png
   :align: center
   :target: temperature-example.html

**********
Install
**********

`albatross`_ is a header only library so incorporating it in your C++ project should be as simple as adding ``./albatross`` as an include directory.

-----------------------
Install as a submodule
-----------------------
If you're using ``git`` you can run ``git submodule add https://github.com/swift-nav/albatross.git``)

Then make sure you've run ``git submodule update --recursive --init`` to be sure all the third party libraries required by albatross are also up to date.

-----------------------
Tests and Examples
-----------------------

If you want to run the tests you can do so using ``cmake``,

.. code-block:: bash

    mkdir build;
    cd build;
    cmake ../
    make run_albatross_unit_tests

Similarly you can make/run the examples,

.. code-block:: bash

    make sinc_example
    ./examples/sinc_example -input ./examples/sinc_input.csv -output ./examples/sinc_predictions.csv

and plot the results (though this'll require a numerical python environment),

.. code-block:: bash

    python ../examples/plot_example_predictions.py ./examples/sinc_input.csv ./examples/sinc_predictions.csv
    ./examples/sinc_example -input ../examples/sinc_input.csv -output ./examples/sinc_predictions.csv -n 10

.. toctree::
    :maxdepth: 1

    1d-example
    temperature-example


#########
Credit
#########

The ``fit``, ``predict``, ``get_params`` functionality was directly inspired by `scikit-learn`_ and the covariance function composition by `george`_.

Like this project? Want to get paid to help us apply it to our GNSS models? `Join us`_ at `Swift Navigation`_ !

.. image:: https://static.fjcdn.com/gifs/Albatross_408ca5_5434150.gif
   :align: center

.. _`scikit-learn` : https://github.com/scikit-learn/scikit-learn
.. _`george` : https://github.com/dfm/george
.. _`Join us` : https://www.swiftnav.com/join-us
.. _`Swift Navigation` : https://www.swiftnav.com/
   

* :ref:`search`
