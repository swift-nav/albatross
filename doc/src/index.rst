=====================================
albatross
=====================================
|Build Status|

A framework for statistical modelling in C++, with a focus on Gaussian processes.

.. _`albatross`: https://travis-ci.com/swift-nav/albatross

.. |Build Status| image:: https://api.travis-ci.com/swift-nav/albatross.svg

***********
Features
***********
 * `Gaussian Process Regression`_ using composable covariance functions and works with custom data types.
 * Bayesian inferrence using an ensemble :ref:`MCMC <mcmc>` sampler based off `emcee`_.
 * Helpful utilities for working with datasets using the :ref:`split apply combine <split-apply-combine>` approach.
 * An interface around `nlopt`_ to make model :ref:`tuning <tuning>` straight forward.
 * Evaluation utilities, with a focus on :ref:`cross validation <crossvalidation>`.
 * Parameter handling which makes it easy to get and set parameters in a standardized way and (de)serialize models.

.. _`nlopt` : https://nlopt.readthedocs.io/en/latest/
.. _`Gaussian Process Regression`: http://www.gaussianprocess.org/gpml/chapters/RW2.pdf

**********
Examples
**********

- :ref:`A One dimensional example using a sinc function. <1d-example>`
- :ref:`An example spatial model for temperature estimation. <temperature-example>`

.. image:: https://raw.githubusercontent.com/swift-nav/albatross/master/examples/temperature_example/mean_temperature.png
   :align: center
   :target: temperature-example.html

**************
Why Albatross?
**************
First let's start with a bigger question, "Why write anything from scratch when you can use external packages?".  Opinions may vary, but some reasonable answers are:

 * The external packages don't quite do what you want.
 * You think you can do it better.
 * You don't want any dependencies.

In on our situation it's mostly the first; we want to build statistical models in a way that accommodates the research phase of development (rapid model iteration, evaluation, comparison, and tuning) but also runs fast in a production environment.  So while just about everything in albatross could also be done with python packages such as ``pandas``, ``scikit-learn``, ``emcee`` and ``george``, using them directly just wasn't practical.  Instead we started developing ``albatross``, which draws heavily on paradigms we liked from those packages but with an emphasis on compile time safety and speed.  In short you could say albatross is:

    "A package containing some of the stats modeling tools from python that are missing in C++"

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

Similarly you can make and run the examples,

.. code-block:: bash

    make sinc_example
    ./examples/sinc_example -input ./examples/sinc_input.csv -output ./examples/sinc_predictions.csv

and plot the results (though this'll require a numerical python environment),

.. code-block:: bash

    python ../examples/plot_example_predictions.py ./examples/sinc_input.csv ./examples/sinc_predictions.csv

.. toctree::
    :maxdepth: 1

    1d-example
    temperature-example
    datasets
    split-apply-combine
    gp
    custom-models
    crossvalidation
    tuning
    mcmc
    sparse-gp-details


#########
Credit
#########

The ``fit``, ``predict``, ``get_params`` functionality was directly inspired by `scikit-learn`_ and the covariance function composition by `george`_.

The ensemble sampler was inspired by `emcee`_ which itself was inspired by the paper `Ensemble samplers with affine invariance`_.

Like this project? Want to get paid to help us apply it to our GNSS models? `Join us`_ at `Swift Navigation`_ !


.. image:: https://bigmemes.funnyjunk.com/gifs/Albatross_408ca5_5434150.gif
   :align: center

.. _`cross validation` : https://web.stanford.edu/~hastie/ElemStatLearn/
.. _`scikit-learn` : https://github.com/scikit-learn/scikit-learn
.. _`george` : https://github.com/dfm/george
.. _`Join us` : https://www.swiftnav.com/join-us
.. _`Swift Navigation` : https://www.swiftnav.com/
.. _`emcee` : https://emcee.readthedocs.io/en/stable/
.. _`Ensemble samplers with affine invariance` : https://msp.org/camcos/2010/5-1/p04.xhtml
   

* :ref:`search`
