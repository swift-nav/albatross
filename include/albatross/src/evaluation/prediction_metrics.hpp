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
  double mse = error.dot(error) / cast::to_double(error.size());
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
  const auto n_elements = cast::to_double(x.size());

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
  ChiSquaredCdf() : PredictionMetric<JointDistribution>(chi_squared_cdf) {}
};

namespace distance {

namespace detail {

inline Eigen::MatrixXd principal_sqrt(const Eigen::MatrixXd &input) {
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigs(input);
  Eigen::VectorXd eigenvalues{eigs.eigenvalues()};
  if (!(eigs.eigenvalues().array() >= 0.).all()) {
    // In the unfortunate case of ill-conditioned arguments, which can
    // easily happen when comparing two nearby distributions, we clamp
    // numerically negative but morally OK eigenvalues to 0.
    //
    // The factor of 10 here is a heuristic to allow small negative
    // eigenvalues (due to numerical errors) but not clamp
    // meaningfully large ones that indicate a serious problem in
    // calculation.
    const double min_positive =
        (eigenvalues.array() > 0)
            .select(eigenvalues.array(),
                    std::numeric_limits<double>::infinity())
            .minCoeff();
    eigenvalues =
        (eigenvalues.array() < 0 && eigenvalues.array() > -10 * min_positive)
            .select(0, eigenvalues.array());
  }

  return eigs.eigenvectors() *
         eigenvalues.array().sqrt().matrix().asDiagonal() *
         eigs.eigenvectors().transpose();
}

} // namespace detail

inline double wasserstein_2(const JointDistribution &a,
                            const JointDistribution &b) {
  auto b_sqrt{detail::principal_sqrt(b.covariance)};
  return (a.mean - b.mean).squaredNorm() +
         (a.covariance + b.covariance -
          2 * detail::principal_sqrt(b_sqrt * a.covariance * b_sqrt))
             .trace();
}

} // namespace distance

namespace score {

enum class VariogramScoreOrder {
  cVariogram,
  cMadogram,
};

namespace detail {

template <typename Generator>
Eigen::MatrixXd draw_mvn(const Eigen::LDLT<Eigen::MatrixXd> &decomp,
                         const Eigen::VectorXd &mean, Eigen::Index n_draws,
                         Generator &&gen) {
  ALBATROSS_ASSERT(decomp.info() == Eigen::Success &&
                   "Please pass a successful covariance decomposition!");
  ALBATROSS_ASSERT((decomp.vectorD().array() > 0.).all() &&
                   "Please pass a positive definite covariance!");
  std::normal_distribution<double> standard_normal{0.0, 1.0};
  return (decomp.transpositionsP() *
          (decomp.matrixL() *
           (decomp.vectorD().array().sqrt().matrix().asDiagonal() *
            Eigen::MatrixXd::NullaryExpr(
                decomp.rows(), n_draws,
                [&standard_normal, &gen] { return standard_normal(gen); }))))
             .colwise() +
         mean;
}

// First energy score term: mean among samples of the 2-norm ||x_i -
// y|| for each sample x_i
inline double mean_err_norms(const Eigen::MatrixXd &samples,
                             const Eigen::VectorXd &truth,
                             const Eigen::VectorXd &weights) {
  ALBATROSS_ASSERT(samples.rows() == truth.size() &&
                   "invalid sampling distribution");
  ALBATROSS_ASSERT(samples.rows() == weights.size() &&
                   "invalid sampling weights");
  return (((samples.colwise() - truth).array().square().colwise() *
           weights.array())
              .colwise()
              .sum())
      .sqrt()
      .mean();
}

// Second energy score term: mean of all the pairwise distances
// between samples x_i, x_j
//
// Rather than compute the triangle of cross-terms, which is quadratic
// in either time or space, we simply demand two matched IID sets of
// sample vectors and compute between corresponding pairs.  This gives
// up statistical power for a considerable speed improvement, so just
// sample more if you are worried.
inline double pairwise_errors_paired(const Eigen::MatrixXd &samples_a,
                                     const Eigen::MatrixXd &samples_b,
                                     const Eigen::VectorXd &weights) {
  ALBATROSS_ASSERT(samples_a.rows() == samples_b.rows() &&
                   "Sample matrices must be the same size!");
  ALBATROSS_ASSERT(samples_a.cols() == samples_b.cols() &&
                   "Sample matrices must be the same size!");
  return ((samples_a - samples_b).array().colwise() * weights.array())
      .matrix()
      .colwise()
      .norm()
      .mean();
}

template <typename Generator>
Eigen::MatrixXd antithetic_sample(
    Eigen::Index num_samples, const Eigen::VectorXd &prediction_mean,
    const Eigen::LDLT<Eigen::MatrixXd> &cov_decomp, Generator &&generator) {
  // Antithetic sampling: use the same variations on both sides of the
  // mean.  This lets us do less sampling work for the same variance
  // without introducing any bias.
  Eigen::Index k_generate = (num_samples / 2 + 1);
  Eigen::MatrixXd samples(cov_decomp.rows(), k_generate * 2);
  samples.leftCols(k_generate) =
      detail::draw_mvn(cov_decomp, prediction_mean, k_generate,
                       std::forward<Generator>(generator));
  // already sampled (mu + v)
  // get (mu - v) without knowing v:
  // mu - v = mu - ((mu + v) - mu)
  //        = 2 * mu - (mu + v)
  samples.rightCols(k_generate) = -samples.leftCols(k_generate);
  samples.rightCols(k_generate).colwise() += 2 * prediction_mean;
  return samples;
}

// The expected value of the absolute value of a normal random
// variable.
//
// Equation 17 of
//
// "Moments and absolute moments of the normal distribution"
// Andreas Winkelbauer
// https://arxiv.org/pdf/1209.4340
inline double expected_abs_normal_1(double mu, double sigma) {
  if (!std::isfinite(mu) || !std::isfinite(sigma)) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  if (sigma <= 0.0) {
    // Degenerate case: expected absolute value of point mass at mu
    return std::abs(mu);
  }

  const double normalized = std::fabs(mu) / std::max(1.0e-16, sigma);
  return sigma * std::sqrt(2. / M_PI) *
             std::exp(-0.5 * normalized * normalized) +
         std::fabs(mu) * std::erf(normalized / std::sqrt(2.));
}

// The expected value of the 2-norm of a normal random variable.
inline double expected_abs_normal_2(double mu, double sigma) {
  return mu * mu + sigma * sigma;
}

// Compute the `p`-th expected absolute moment of the given normal
// distribution.
inline double expected_abs_normal(double mu, double sigma,
                                  VariogramScoreOrder order) {
  switch (order) {
  case VariogramScoreOrder::cVariogram:
    return expected_abs_normal_2(mu, sigma);
  case VariogramScoreOrder::cMadogram:
    return expected_abs_normal_1(mu, sigma);
  default:
    ALBATROSS_ASSERT(false && "Invalid variogram score order!");
    return INFINITY;
  }
}

inline double variogram_score_order_p(VariogramScoreOrder order) {
  switch (order) {
  case VariogramScoreOrder::cVariogram:
    return 2.;
  case VariogramScoreOrder::cMadogram:
    return 1.;
  default:
    ALBATROSS_ASSERT(false && "Invalid variogram score order!");
    return INFINITY;
  }
}

} // namespace detail

namespace constant {
// Default number of MC samples to compute the energy score
static constexpr const Eigen::Index cEnergyScoreDefaultSampleCount{1000};
// Default seed value for MC sampling
static constexpr const unsigned cEnergyScoreDefaultSeed{22U};
// Default variogram score order `p`
static constexpr const VariogramScoreOrder cDefaultVariogramScoreOrder{
    VariogramScoreOrder::cMadogram};
} // namespace constant

// The continuous-ranked probability score (CRPS) for univariate
// normal distributions.
inline double crps_normal(double mu, double sigma, double y) {
  if (!std::isfinite(mu) || !std::isfinite(sigma) || !std::isfinite(y)) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  if (sigma <= 0.0) {
    // Degenerate case: CRPS -> absolute error
    return std::abs(y - mu);
  }

  const double z = (y - mu) / sigma;
  const double erfz = std::erf(z / std::sqrt(2.0));
  const double phi = std::exp(-0.5 * z * z) / std::sqrt(2. * M_PI);

  return sigma * (z * erfz + 2.0 * phi - 1.0 / std::sqrt(M_PI));
}

// Compute the energy score of the `truth` data under the `prediction`
// distribution.  This is the multivariate generalisation of the CRPS:
//
//   ES(F, y) = E||X - y|| - 0.5 * E||X - X'||
//
// where
//
//   X is drawn from F ~ MVN(\mu, \Sigma)
//
// The references compute this score from ensemble forecasts.  We are
// working with closed-form multivariate Gaussians, but I haven't
// found any nice analytic way to compute this for MVNs, so we also
// compute it via Monte Carlo sampling.
//
// You may supply `weights` to control the relative scale or
// importance of different dimensions in your distribution.
//
// "Assessing probabilistic forecasts of multivariate quantities, with
// an application to ensemble predictions of surface winds"
// Gneiting, Stanberry et al.
// https://stat.uw.edu/sites/default/files/files/reports/2008/tr537.pdf
inline double energy_score(
    const JointDistribution &prediction, const Eigen::VectorXd &truth,
    const Eigen::VectorXd *weights = nullptr,
    unsigned seed = constant::cEnergyScoreDefaultSeed,
    Eigen::Index num_samples = constant::cEnergyScoreDefaultSampleCount) {
  ALBATROSS_ASSERT(num_samples > 1 &&
                   "Cannot form an MC approximation with 1 or fewer samples");
  ALBATROSS_ASSERT(prediction.mean.size() == truth.size() &&
                   "Predictive distribution and truth have different sizes!");

  Eigen::VectorXd weight_vector;
  if (weights != nullptr) {
    weight_vector = *weights;
  } else {
    weight_vector = Eigen::VectorXd::Ones(truth.size());
  }
  ALBATROSS_ASSERT(weight_vector.size() == truth.size() &&
                   "Variogram score weights must be a square matrix matched to "
                   "the size of the problem!");

  Eigen::LDLT<Eigen::MatrixXd> cov_decomp(prediction.covariance);

  std::default_random_engine rng(seed);
  Eigen::MatrixXd samples =
      detail::antithetic_sample(num_samples, prediction.mean, cov_decomp, rng);
  Eigen::MatrixXd samples_b =
      detail::antithetic_sample(num_samples, prediction.mean, cov_decomp, rng);

  const double es =
      0.5 * (detail::mean_err_norms(samples, truth, weight_vector) +
             detail::mean_err_norms(samples_b, truth, weight_vector)) -
      0.5 * detail::pairwise_errors_paired(samples, samples_b, weight_vector);
  ALBATROSS_ASSERT(es >= -1.e-12 &&
                   "Energy score should never be significantly negative!");
  return std::max(0.0, es);
}

// The energy score of a truth distribution measured with some
// uncertainty.
inline double energy_score(
    const JointDistribution &prediction, const MarginalDistribution &truth,
    const Eigen::VectorXd *weights = nullptr,
    unsigned seed = constant::cEnergyScoreDefaultSeed,
    Eigen::Index num_samples = constant::cEnergyScoreDefaultSampleCount) {
  JointDistribution combined_prediction(prediction);
  combined_prediction.covariance += truth.covariance;
  return energy_score(combined_prediction, truth.mean, weights, seed,
                      num_samples);
}

// Compute the variogram score of order `p` of the `truth` data under
// the `prediction` distribution.  The variogram score is a proper
// (but not _strictly_ proper) multivariate scoring rule that places
// greater emphasis on correct cross-covariance terms of the
// predictive distribution than the energy score does.
//
//   VS_p(F, y) = \Sum_{i, j}^d w_{ij} (|y_i - y_j|^p - E_F |X_i - X_j|^p)^2
//
// where
//
//   X is drawn from F ~ MVN(\mu, \Sigma)
//
// The reference defines this score for ensemble forecasts.  We are
// working with closed-form multivariate Gaussians, so we can define
// the same score by replacing the second (expectation) term by a
// closed-form calculation for the special cases of `p` = 1 (the
// "madogram") and `p` = 2 (the "variogram").
//
// You may supply `weights` to control the relative scale or
// importance of different dimensions in your distribution.  The
// reference suggests emphasizing correlations at short distances by
// weighting in proportion to the inverse distances between components
// of the distribution.
//
// Variogram-based proper scoring rules for probabilistic forecasts of
// multivariate quantities
// Michael Scheuerer and Thomas Hamill
// https://journals.ametsoc.org/view/journals/mwre/143/4/mwr-d-14-00269.1.xml
inline double variogram_score(
    const JointDistribution &prediction, const Eigen::VectorXd &truth,
    const Eigen::MatrixXd *weights = nullptr,
    VariogramScoreOrder order = constant::cDefaultVariogramScoreOrder) {
  ALBATROSS_ASSERT(prediction.mean.size() == truth.size() &&
                   "Predictive distribution and truth have different sizes!");
  Eigen::MatrixXd weight_matrix;
  if (weights != nullptr) {
    weight_matrix = *weights;
  } else {
    weight_matrix = Eigen::MatrixXd::Constant(truth.size(), truth.size(), 1.);
  }
  ALBATROSS_ASSERT(weight_matrix.rows() == weight_matrix.cols() &&
                   weight_matrix.rows() == truth.size() &&
                   "Variogram score weights must be a square matrix matched to "
                   "the size of the problem!");

  double sum{0.};
  const double p = detail::variogram_score_order_p(order);
  for (Eigen::Index row = 0; row < truth.size() - 1; ++row) {
    const Eigen::Index rest = truth.size() - row - 1;
    ALBATROSS_ASSERT(rest > 0 && "Internal error");
    ALBATROSS_ASSERT(rest < truth.size() && "Internal error");
    Eigen::VectorXd mean_err =
        prediction.mean.tail(rest).array() - prediction.mean(row);
    Eigen::VectorXd sigma =
        (prediction.get_diagonal(row) +
         prediction.covariance.diagonal().tail(rest).array() -
         2. * prediction.covariance.row(row).tail(rest).array())
            .sqrt();
    Eigen::VectorXd expectation =
        mean_err.binaryExpr(sigma, [order](double mean, double std) {
          return detail::expected_abs_normal(mean, std, order);
        });
    Eigen::VectorXd truth_err =
        (truth(row) - truth.tail(rest).array()).abs().pow(p);
    Eigen::VectorXd diff = truth_err - expectation;
    sum += (weight_matrix.row(row).tail(rest).array() * diff.array() *
            diff.array())
               .sum();
  }

  ALBATROSS_ASSERT(sum >= 0.0 && "Invalid (negative) variogram score!");
  return sum;
}

// The variogram score of a truth distribution measured with some
// uncertainty.
inline double variogram_score(
    const JointDistribution &prediction, const MarginalDistribution &truth,
    const Eigen::MatrixXd *weights = nullptr,
    VariogramScoreOrder order = constant::cDefaultVariogramScoreOrder) {
  JointDistribution combined_prediction(prediction);
  combined_prediction.covariance += truth.covariance;
  return variogram_score(combined_prediction, truth.mean, weights, order);
}

} // namespace score

} // namespace albatross

#endif /* ALBATROSS_EVALUATION_PREDICTION_METRICS_H_ */
