# albatross
A framework for statistical modelling in C++, with a focus on Gaussian processes.

## Features
 * Parameter handling which makes it easy to get and set parameters in a standardized way  as well as compose and (de)serialize models to string.
 * [Gaussian process regression](http://www.gaussianprocess.org/gpml/chapters/RW2.pdf) which is accomplished using composable covariance functions and templated predictor types.
 * Evaluation utilities for performing cross validation.

## Examples

As an example we can build a model which estimates a function `f(x)`,
```
f(x) = a x + b + c sin(x) / x
```
by using noisy observations of the function `y = f(x) + N(0, s^2)`.  To do so we can build up our model,
```
  using Mean = ConstantMean<double>;
  using Slope = SlopeTerm;
  using Noise = IndependentNoise<double>;
  using SqrExp = SquaredExponential<double, ScalarDistance>;

  CovarianceFunction<Mean, double> mean = {Mean(10.)};
  CovarianceFunction<Slope, double> slope = {Slope(10.)};
  CovarianceFunction<Noise, double> noise = {Noise(meas_noise)};
  CovarianceFunction<SqrExp, double> sqrexp = {SqrExp(2., 5.)};

  auto linear_model = mean + slope + noise + sqrexp;
```
which incorporates prior knowledge that the function consists of a mean offset, a linear term, measurement noise and an unknown smooth compontent (which we captures using a squared exponential covariance function).

We can inspect the model
```
>> std::cout << linear_model.to_string() << std::endl;
model_name: (((constant_mean+slope_term)+independent_noise)+squared_exponential[scalar_distance])
model_params:
  length_scale: 2
  sigma_constant_mean: 10
  sigma_independent_noise: 1
  sigma_slope: 10
  sigma_squared_exponential: 5
```

then condition the model on random observations, `data`,

```
  auto model = gp_from_covariance(linear_model);
  model.fit(data);
  const int k = 161;
  const auto grid_xs = uniform_points_on_line(k, low - 2., high + 2.);
  const auto predictions = model.predict(grid_xs);
```

and compare the predictive distributions as a function of the number of observations,

![2](https://github.com/swift-nav/albatross/blob/master/examples/sinc_function_2.png)
![5](https://github.com/swift-nav/albatross/blob/master/examples/sinc_function_5.png)
![10](https://github.com/swift-nav/albatross/blob/master/examples/sinc_function_10.png)
![30](https://github.com/swift-nav/albatross/blob/master/examples/sinc_function_30.png)

 ## Credit
 A lot of the functionality was inspired by [scikit-learn](https://github.com/scikit-learn/scikit-learn) and [george](https://github.com/dfm/george).

![albatross](https://static.fjcdn.com/gifs/Albatross_408ca5_5434150.gif)

