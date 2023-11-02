#################################
Quick Start Tutorial
#################################

.. _quickstart:

Here is an example work flow in which we create a GP, tune its parameters to maximize the leave one out cross validated likelihood, then fit the model and use it to make predictions of unobserved data.

.. code-block:: c

  RegressionDataset<double> training_data = make_training_data();
  RegressionDataset<double> evaluation_data = make_evaluation_data();

  const IndependentNoise<double> independent_noise;
  const SquaredExponential<EuclideanDistance> squared_exponential;
  const auto covariance = squared_exponential + independent_noise;
  auto gp = gp_from_covariance(covariance);

  albatross::LeaveOneOutLikelihood<> loo_nll;
  const auto tuned_params = get_tuner(gp, loo_nll, training_data).tune();
  gp.set_params(tuned_params);

  const auto fit_model = gp.fit(training_data);
  const Eigen::VectorXd prediction = fit_model.predict(evaluation_data.features).mean();  
  
  const double rmse = root_mean_square_error(prediction, evaluation_data.targets);

  std::cout << "Evaluation RMSE: " << rmse << std::endl;


