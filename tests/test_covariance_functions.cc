
#include <gtest/gtest.h>
#include <Eigen/Core>
#include <iostream>
#include <vector>
#include "covariance_functions/covariance_functions.h"

namespace albatross {

std::vector<Eigen::Vector3d> points_on_a_line(const int n) {
  std::vector<Eigen::Vector3d> xs;
  for (int i = 0; i < n; i++) {
    Eigen::Vector3d x;
    for (int j = 0; j < 3; j++) x[static_cast<std::size_t>(j)] = 1000*i + j;
    xs.push_back(x);
  }
  return xs;
}

TEST(test_gp_covariance_functions, test_build_covariance) {
  using Predictor = Eigen::Vector3d;
  using Mean = ConstantMean<Predictor>;
  using Noise = IndependentNoise<Predictor>;
  using SqExp = SquaredExponential<Predictor, EuclideanDistance<Predictor>>;
  using RadialSqExp = SquaredExponential<Predictor, RadialDistance<Predictor>>;

  CovarianceFunction<SqExp, Predictor> sqexp = {SqExp()};
  CovarianceFunction<Mean, Predictor> mean = {Mean()};
  CovarianceFunction<Noise, Predictor> noise = {Noise()};
  CovarianceFunction<RadialSqExp, Predictor> radial_sqexp = {RadialSqExp()};

  // Add and multiply covariance functions together and make sure they are
  // still capable of producing a covariance matrix.
  auto product = sqexp * radial_sqexp;
  auto covariance_function = mean + product + noise;

  auto xs = points_on_a_line(5);
  Eigen::MatrixXd C = symmetric_covariance(covariance_function, xs);
  std::cout << C << std::endl;
  std::cout << covariance_function.to_string() << std::endl;
}

TEST(test_gp_covariance_functions, test_build_covariance) {
  using Predictor = Eigen::Vector3d;
  using Mean = ConstantMean<Predictor>;
  using Noise = IndependentNoise<Predictor>;
  using SqExp = SquaredExponential<Predictor, EuclideanDistance<Predictor>>;
  using RadialSqExp = SquaredExponential<Predictor, RadialDistance<Predictor>>;

  CovarianceFunction<SqExp, Predictor> sqexp = {SqExp()};
  CovarianceFunction<Mean, Predictor> mean = {Mean()};
  CovarianceFunction<Noise, Predictor> noise = {Noise()};
  CovarianceFunction<RadialSqExp, Predictor> radial_sqexp = {RadialSqExp()};

  // Add and multiply covariance functions together and make sure they are
  // still capable of producing a covariance matrix.
  auto product = sqexp * radial_sqexp;
  auto covariance_function = mean + product + noise;

  auto xs = points_on_a_line(5);
  Eigen::MatrixXd C = symmetric_covariance(covariance_function, xs);
  std::cout << C << std::endl;
  std::cout << covariance_function.to_string() << std::endl;
}

}
