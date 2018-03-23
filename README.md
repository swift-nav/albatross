# albatross [![Build Status](https://travis-ci.com/swift-nav/albatross.svg?token=ZCoayM24vorooTuykqeC&branch=master)](https://travis-ci.com/swift-nav/albatross)
A framework for statistical modelling in C++, with a focus on Gaussian processes.

## Features
 * [Gaussian process regression](http://www.gaussianprocess.org/gpml/chapters/RW2.pdf) which is accomplished using composable covariance functions and templated predictor types.
 * Evaluation utilities, with a focus on cross validation.
 * Written using generics in an attempt to make these core routines applicable to a number of fields.
 * Parameter handling which makes it easy to get and set parameters in a standardized way  as well as compose and (de)serialize models to string.

## Install

`albatross` is a header only library so incorporating it in your C++ project should be as simple as adding `./albatross` as an include directory.

Make sure you've run `git submodule update --recursive` to be sure all the third party libraries required by albatross are up to date.

If you want to run the tests you can do so using `cmake`,

```
mkdir build;
cd build;
cmake ../
make run_albatross_unit_tests
```
Similarly you can make/run the examples,
```
make sinc_example
./examples/sinc_example -input ./examples/sinc_input.csv -output ./examples/sinc_predictions.csv
```
and plot the results (though this'll require a numerical python environment),
```
python ../examples/plot_example_predictions.py ./examples/sinc_input.csv ./examples/sinc_predictions.csv
./examples/sinc_example -input ../examples/sinc_input.csv -output ./examples/sinc_predictions.csv -n 10
```

## Examples

As an example we can build a model which estimates a function `f(x)`,
```
f(x) = a x + b + c sin(x) / x
```
by using noisy observations of the function `y = f(x) + N(0, s^2)`.  To do so we can build up our model,
```
  using Noise = IndependentNoise<double>;
  using SqrExp = SquaredExponential<ScalarDistance>;

  CovarianceFunction<Constant> mean = {Constant(100.)};
  CovarianceFunction<SlopeTerm> slope = {SlopeTerm(100.)};
  CovarianceFunction<Noise> noise = {Noise(meas_noise)};
  CovarianceFunction<SqrExp> sqrexp = {SqrExp(2., 5.)};

  auto linear_model = mean + slope + noise + sqrexp;
```
which incorporates prior knowledge that the function consists of a mean offset, a linear term, measurement noise and an unknown smooth compontent (which we captures using a squared exponential covariance function).

We can inspect the model and its parameters,
```
>> std::cout << linear_model.to_string() << std::endl;
model_name: (((constant+slope_term)+independent_noise)+squared_exponential[scalar_distance])
model_params:
  length_scale: 2
  sigma_constant: 100
  sigma_independent_noise: 1
  sigma_slope: 100
  sigma_squared_exponential: 5
```
then condition the model on random observations, which we stored in `data`,
```
  auto model = gp_from_covariance(linear_model);
  model.fit(data);
```
and make some gridded predictions,
```
  const int k = 161;
  const auto grid_xs = uniform_points_on_line(k, low - 2., high + 2.);
  const auto predictions = model.predict(grid_xs);
```
Here are the resulting predictions when we have only two noisy observations,

### 2 Observations
![2](https://github.com/swift-nav/albatross/blob/master/examples/sinc_function_2.png)

not great, but at least it knows it isn't great.  As we start to add more observations
we can watch the model slowly get more confident,

### 5 Observations
![5](https://github.com/swift-nav/albatross/blob/master/examples/sinc_function_5.png)
### 10 Observations
![10](https://github.com/swift-nav/albatross/blob/master/examples/sinc_function_10.png)
### 30 Observations
![30](https://github.com/swift-nav/albatross/blob/master/examples/sinc_function_30.png)


## Credit
The `fit`, `predict`, `get_params` functionality was inspired by [scikit-learn](https://github.com/scikit-learn/scikit-learn) and the covariance function composition by [george](https://github.com/dfm/george).

Like this project? Want to get paid to help us apply it to our GNSS models? [Join us](https://www.swiftnav.com/join-us) at [Swift Navigation](https://www.swiftnav.com/)!

![albatross](https://static.fjcdn.com/gifs/Albatross_408ca5_5434150.gif)

