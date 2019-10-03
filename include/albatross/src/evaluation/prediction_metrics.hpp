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

#include "stats.hpp"

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

static inline double standard_deviation(const Eigen::VectorXd &prediction,
                                        const Eigen::VectorXd &truth) {
  Eigen::VectorXd error = prediction - truth;
  const auto n_elements = static_cast<double>(error.size());
  const double mean_error = error.sum() / n_elements;
  error.array() -= mean_error;
  return std::sqrt(error.dot(error) / (n_elements - 1));
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
  if (truth.has_covariance()) {
    covariance += truth.covariance;
  }
  return albatross::negative_log_likelihood(mean, covariance);
}

static inline double
negative_log_likelihood(const MarginalDistribution &prediction,
                        const MarginalDistribution &truth) {
  const Eigen::VectorXd mean = prediction.mean - truth.mean;
  Eigen::VectorXd variance(prediction.covariance.diagonal());
  if (truth.has_covariance()) {
    variance += truth.covariance.diagonal();
  }
  return albatross::negative_log_likelihood(mean, variance.asDiagonal());
}

template <typename PredictType = JointDistribution>
struct NegativeLogLikelihood : public PredictionMetric<PredictType> {
  NegativeLogLikelihood()
      : PredictionMetric<PredictType>(negative_log_likelihood) {}
};

/*
 * Mahalanobis Chi-Squared Statistic
 *
 * Tests the normality of a multivariate distribution by computing
 * the squared Mahalanobis distance which should follow a chi-squared
 * distribution
 */

inline double
mahalanobis_chi_squared_statistic(const Eigen::VectorXd &deviation,
                                  const Eigen::MatrixXd &covariance) {
  double distance_squared = covariance.llt().matrixL().solve(deviation).squaredNorm();
  Eigen::Index n = deviation.size();
  return 1. - stats::pchisq(n, distance_squared);
}

inline double
mahalanobis_chi_squared_statistic(const JointDistribution &prediction,
                                  const MarginalDistribution &truth) {
  return mahalanobis_chi_squared_statistic(prediction.mean - truth.mean, prediction.covariance);
}

struct MahalanobisChiSquaredCdf : public PredictionMetric<JointDistribution> {
  MahalanobisChiSquaredCdf()
      : PredictionMetric<JointDistribution>(mahalanobis_chi_squared_statistic) {}
};

} // namespace albatross

#endif /* ALBATROSS_EVALUATION_PREDICTION_METRICS_H_ */
