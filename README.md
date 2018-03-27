# albatross
A framework for statistical modelling in C++, with a focus on Gaussian processes.

## Features
 * Parameter handling which makes it easy to get and set parameters in a standardized way  as well as compose and (de)serialize models to string.
 * (Gaussian process regression)[http://www.gaussianprocess.org/gpml/chapters/RW2.pdf] which is accomplished using composable covariance functions and templated predictor types.
 * Evaluation utilities for performing cross validation.

## Examples

As an example we can build a model which estimates a function `f(x)`,
```
f(x) = a x + b + c sin(x) / x
```
by using noisy observations of the function `y = f(x) + N(0, s^2)`.  To do so we can build up our model,
```
  using Mean = albatross::ConstantMean<double>;
  using Slope = albatross::SlopeTerm;
  using Noise = albatross::IndependentNoise<double>;
  using SqrExp = albatross::SquaredExponential<double, albatross::ScalarDistance>;

  albatross::CovarianceFunction<Mean, double> mean = {Mean(10.)};
  albatross::CovarianceFunction<Slope, double> slope = {Slope(10.)};
  albatross::CovarianceFunction<Noise, double> noise = {Noise(meas_noise)};
  albatross::CovarianceFunction<SqrExp, double> sqrexp = {SqrExp(2., 5.)};

  auto linear_model = mean + slope + noise + sqrexp;
```
which incorporates prior knowledge that the function consists of a mean offset, a linear term, measurement noise and an unknown smooth compontent (which we captures using a squared exponential covariance function).

We can then condition the model on random observations, `y`, and compare the predictive distributions as a function of the number of observations,

![2](https://github.com/swift-nav/albatross/blob/master/examples/sinc_function_2.png)
![5](https://github.com/swift-nav/albatross/blob/master/examples/sinc_function_5.png)
![10](https://github.com/swift-nav/albatross/blob/master/examples/sinc_function_10.png)
![30](https://github.com/swift-nav/albatross/blob/master/examples/sinc_function_30.png)

 ## Credit
 A lot of the functionality was inspired by [scikit-learn](https://github.com/scikit-learn/scikit-learn)) and [george](https://github.com/dfm/george).

![albatross](https://static.fjcdn.com/gifs/Albatross_408ca5_5434150.gif)

