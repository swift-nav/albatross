/*
 * Copyright (C) 2019 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef ALBATROSS_EVALUATION_PREDICTION_METRICS_H_
#define ALBATROSS_EVALUATION_PREDICTION_METRICS_H_

namespace albatross {

/*
 * An PredictionMetric is basically just a wrapper around a
 * function which enforces a signature used in evaluation.  One
 * alternative would be to use std::function.  Ie,
 *
 *   template <typename RequiredPredictType>
 *   using PredictionMetric =
 *         std::function<double(const RequiredPredictType &,
 *                              const MarginalDistribution &);
 *
 * the problem with this is that when you pass the std::function
 * object as an argument the RequiredPredictType cannot be
 * inferred.
 */

template <typename RequiredPredictType>
using PredictionMetricFunction = double (*)(const RequiredPredictType &,
                                            const MarginalDistribution &);

template <typename RequiredPredictType> struct PredictionMetric {

  PredictionMetricFunction<RequiredPredictType> eval_;

  PredictionMetric(PredictionMetricFunction<RequiredPredictType> eval)
      : eval_(eval) {}

  double operator()(const RequiredPredictType &prediction,
                    const MarginalDistribution &truth) const {
    return eval_(prediction, truth);
  }

  template <typename ModelType, typename FeatureType, typename FitType>
  double
  operator()(const Prediction<ModelType, FeatureType, FitType> &prediction,
             const MarginalDistribution &truth) const {
    return (*this)(prediction.template get<RequiredPredictType>(), truth);
  }
};

static inline double root_mean_square_error(const Eigen::VectorXd &prediction,
                                            const Eigen::VectorXd &truth) {
  const Eigen::VectorXd error = prediction - truth;
  double mse = error.dot(error) / static_cast<double>(error.size());
  return sqrt(mse);
}

static inline double root_mean_square_error(const Eigen::VectorXd &prediction,
                                            const MarginalDistribution &truth) {
  return root_mean_square_error(prediction, truth.mean);
}

struct RootMeanSquareError : public PredictionMetric<Eigen::VectorXd> {
  RootMeanSquareError()
      : PredictionMetric<Eigen::VectorXd>(root_mean_square_error) {}
};

static inline double standard_deviation(const Eigen::VectorXd &x) {
  if (x.size() == 0) {
    return NAN;
  }

  if (x.size() == 1) {
    return 0.;
  }

  double output = 0.;
  double mean = x.mean();
  const auto n_elements = static_cast<double>(x.size());

  for (Eigen::Index i = 0; i < x.size(); ++i) {
    output += pow(x[i] - mean, 2.) / (n_elements - 1);
  }

  return std::sqrt(output);
}

static inline double standard_deviation(const Eigen::VectorXd &prediction,
                                        const Eigen::VectorXd &truth) {
  return standard_deviation(prediction - truth);
}

static inline double standard_deviation(const Eigen::VectorXd &prediction,
                                        const MarginalDistribution &truth) {
  return standard_deviation(prediction, truth.mean);
}

struct StandardDeviation : public PredictionMetric<Eigen::VectorXd> {
  StandardDeviation() : PredictionMetric<Eigen::VectorXd>(standard_deviation) {}
};

/*
 * Computes the negative log likelihood under the assumption that the
 * predictive
 * distribution is multivariate normal.
 */
static inline double
negative_log_likelihood(const JointDistribution &prediction,
                        const MarginalDistribution &truth) {
  const Eigen::VectorXd mean = prediction.mean - truth.mean;
  Eigen::MatrixXd covariance(prediction.covariance);
  covariance += truth.covariance;
  return albatross::negative_log_likelihood(mean, covariance);
}

static inline double
negative_log_likelihood(const MarginalDistribution &prediction,
                        const MarginalDistribution &truth) {
  const Eigen::VectorXd mean = prediction.mean - truth.mean;
  Eigen::VectorXd variance(prediction.covariance.diagonal());
  variance += truth.covariance.diagonal();
  return albatross::negative_log_likelihood(mean, variance.asDiagonal());
}

template <typename PredictType = JointDistribution>
struct NegativeLogLikelihood : public PredictionMetric<PredictType> {
  NegativeLogLikelihood()
      : PredictionMetric<PredictType>(negative_log_likelihood) {}
};

inline double chi_squared_cdf(const JointDistribution &prediction,
                              const MarginalDistribution &truth) {
  Eigen::MatrixXd covariance(prediction.covariance);
  covariance += truth.covariance;
  return chi_squared_cdf(prediction.mean - truth.mean, covariance);
}

struct ChiSquaredCdf : public PredictionMetric<JointDistribution> {
  ChiSquaredCdf() : PredictionMetric<JointDistribution>(chi_squared_cdf){};
};

} // namespace albatross

#endif /* ALBATROSS_EVALUATION_PREDICTION_METRICS_H_ */
