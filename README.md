# albatross [![Build Status](https://travis-ci.com/swift-nav/albatross.svg?token=ZCoayM24vorooTuykqeC&branch=master)](https://travis-ci.com/swift-nav/albatross)
A framework for statistical modelling in C++, with a focus on Gaussian processes.

## Features
 * [Gaussian process regression](http://www.gaussianprocess.org/gpml/chapters/RW2.pdf) which is accomplished using composable covariance functions and templated predictor types.
 * Evaluation utilities, with a focus on cross validation.
 * Written using generics in an attempt to make these core routines applicable to a number of fields.
 * Parameter handling which makes it easy to get and set parameters in a standardized way  as well as compose and (de)serialize models to string.

For more details [See the full documentation](https://swiftnav-albatross.readthedocs.io/en/latest/)
