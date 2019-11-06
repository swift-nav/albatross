/*
 * Copyright (C) 2018 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef ALBATROSS_CORE_DECLARATIONS_H
#define ALBATROSS_CORE_DECLARATIONS_H

namespace mapbox {
namespace util {
template <typename... Ts> class variant;
}
} // namespace mapbox

using mapbox::util::variant;

namespace albatross {

/*
 * Model
 */
template <typename ModelType> class ModelBase;

template <typename FeatureType> struct RegressionDataset;

template <typename T> struct PredictTypeIdentity;

template <typename ModelType, typename FeatureType, typename FitType>
class Prediction;

template <typename ModelType, typename FitType> class FitModel;

template <typename Derived> class Fit {};

template <typename X> struct Measurement;

/*
 * Group By
 */

using GroupIndices = std::vector<std::size_t>;

template <typename GroupKey, typename ValueType, typename Enable = void>
class Grouped;

template <typename GroupKey>
using GroupIndexer = Grouped<GroupKey, GroupIndices>;

struct LeaveOneOutGrouper;

template <typename ValueType, typename GrouperFunction> class GroupBy;

/*
 * Parameter Handling
 */
struct Parameter;
class PriorContainer;

using ParameterKey = std::string;
// If you change the way these are stored, be sure there's
// a corresponding cereal type included or you'll get some
// really impressive compilation errors.
using ParameterPrior = PriorContainer;
using ParameterValue = double;

using ParameterStore = std::map<ParameterKey, Parameter>;

/*
 * Distributions
 */
template <typename CovarianceType> struct Distribution;

using JointDistribution = Distribution<Eigen::MatrixXd>;
using DiagonalMatrixXd = Eigen::DiagonalMatrix<double, Eigen::Dynamic>;
using MarginalDistribution = Distribution<DiagonalMatrixXd>;

/*
 * Models
 */
template <typename CovarianceFunc, typename ImplType> class GaussianProcessBase;

template <typename CovarianceFunc> class GaussianProcessRegression;

struct NullLeastSquaresImpl {};

template <typename ImplType = NullLeastSquaresImpl> class LeastSquares;

/*
 * Cross Validation
 */
template <typename FeatureType> struct RegressionFold;

template <typename GroupKey, typename FeatureType>
using RegressionFolds = Grouped<GroupKey, RegressionFold<FeatureType>>;

template <typename FeatureType>
using GroupFunction = std::string (*)(const FeatureType &);

template <typename ModelType> class CrossValidation;

template <typename MetricType> class ModelMetric;

template <typename RequiredPredictType> struct PredictionMetric;

/*
 * RANSAC
 */

template <typename GroupKey> struct RansacOutput;

struct RansacConfig;

template <typename ModelType, typename StrategyType> class Ransac;

template <typename ModelType, typename StrategyType, typename FeatureType,
          typename GroupKey>
struct RansacFit;

template <typename InlierMetric, typename ConsensusMetric,
          typename GrouperFunction>
struct GenericRansacStrategy;

struct AlwaysAcceptCandidateMetric;

template <typename InlierMetric, typename ConsensusMetric,
          typename GrouperFunction,
          typename IsValidCandidateMetric = AlwaysAcceptCandidateMetric>
struct GaussianProcessRansacStrategy;

/*
 * Traits
 */

template <typename First, typename Second> struct TypePair {
  using first_type = First;
  using second_type = Second;
};

} // namespace albatross

#endif
