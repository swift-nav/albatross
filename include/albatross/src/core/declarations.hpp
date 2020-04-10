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
 * We frequently inspect for definitions of functions which
 * must be defined for const references to objects
 * (so that repeated evaluations return the same thing
 *  and so the computations are not repeatedly copying.)
 * This type conversion utility will turn a type `T` into `const T&`
 */
template <class T> struct const_ref {
  typedef std::add_lvalue_reference_t<std::add_const_t<T>> type;
};

template <typename T> using const_ref_t = typename const_ref<T>::type;

/*
 * Model
 */
template <typename ModelType> class ModelBase;

template <typename FeatureType> struct RegressionDataset;

template <typename T> struct PredictTypeIdentity;

template <typename ModelType, typename FeatureType, typename FitType>
class Prediction;

template <typename ModelType, typename FeatureType, typename FitType>
using PredictionReference =
    Prediction<const_ref_t<ModelType>, FeatureType, const_ref_t<FitType>>;

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
template <typename Derived> struct DistributionBase;

using DiagonalMatrixXd = Eigen::DiagonalMatrix<double, Eigen::Dynamic>;
struct JointDistribution;
struct MarginalDistribution;

/*
 * Covariance Functions
 */
template <typename X, typename Y> class SumOfCovarianceFunctions;

template <typename X, typename Y> class ProductOfCovarianceFunctions;

template <typename Derived> class CallTrace;

struct ZeroMean;

template <typename X, typename Y> class SumOfMeanFunctions;

template <typename X, typename Y> class ProductOfMeanFunctions;

template <typename X> struct LinearCombination;

/*
 * Models
 */
template <typename CovarianceFunc, typename MeanFunction, typename ImplType>
class GaussianProcessBase;

template <typename CovarianceFunc, typename MeanFunction = ZeroMean>
class GaussianProcessRegression;

struct NullLeastSquaresImpl {};

template <typename ImplType = NullLeastSquaresImpl> class LeastSquares;

template <typename FeatureType> struct LinearCombination;

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
          typename IsValidCandidateMetric, typename GrouperFunction>
struct GaussianProcessRansacStrategy;

/*
 * Samplers
 */

struct SamplerState;

struct NullCallback;

using EnsembleSamplerState = std::vector<SamplerState>;

/*
 * Traits
 */

template <typename First, typename Second> struct TypePair {
  using first_type = First;
  using second_type = Second;
};

template <typename T> struct is_measurement;

} // namespace albatross

#endif
