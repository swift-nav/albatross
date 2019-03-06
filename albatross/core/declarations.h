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

namespace Eigen {

template <typename _Scalar, int SizeAtCompileTime>
class SerializableDiagonalMatrix;
}

namespace albatross {

/*
 * Model
 */
template <typename ModelType> class ModelBase;

template <typename FeatureType> struct RegressionDataset;
// template <typename FeatureType> struct RegressionFold;

template <typename T> struct PredictTypeIdentity;

template <typename ModelType, typename FeatureType, typename FitType> class Prediction;

template <typename ModelType, typename FitType> class FitModel;

template <typename ModelType, typename FeatureType=void> class Fit {};

//template <typename ModelType, typename FeatureType, typename Fit>
//Prediction<ModelType, FeatureType, Fit>
//predict(const FitModel<ModelType, Fit> &,
//        const std::vector<FeatureType> &);

/*
 * Parameter Handling
 */
class Prior;
struct Parameter;

using ParameterKey = std::string;
// If you change the way these are stored, be sure there's
// a corresponding cereal type included or you'll get some
// really impressive compilation errors.
using ParameterPrior = std::shared_ptr<Prior>;
using ParameterValue = double;

using ParameterStore = std::map<ParameterKey, Parameter>;

/*
 * Distributions
 */
template <typename CovarianceType> struct Distribution;

using JointDistribution = Distribution<Eigen::MatrixXd>;
using DiagonalMatrixXd =
    Eigen::SerializableDiagonalMatrix<double, Eigen::Dynamic>;
using MarginalDistribution = Distribution<DiagonalMatrixXd>;

/*
 * Models
 */
template <typename FeatureType, typename CovarianceFunc, typename ImplType>
class GaussianProcessBase;

template <typename FeatureType, typename CovarianceFunc>
class GaussianProcessRegression;

struct NullLeastSquaresImpl {};

template <typename ImplType = NullLeastSquaresImpl>
class LeastSquares;



/*
 * Cross Validation
 */
 using FoldIndices = std::vector<std::size_t>;
 using FoldName = std::string;
 using FoldIndexer = std::map<FoldName, FoldIndices>;

 template <typename FeatureType>
 using IndexerFunction =
    std::function<FoldIndexer(const RegressionDataset<FeatureType> &)>;

 template <typename ModelType>
 class CrossValidation;

/*
 * RANSAC
 */
// template <typename ModelType, typename FeatureType> class GenericRansac;
// template <typename FeatureType, typename ModelType>
// std::unique_ptr<GenericRansac<ModelType, FeatureType>>
// make_generic_ransac_model(ModelType *model, double inlier_threshold,
//                          std::size_t min_inliers,
//                          std::size_t random_sample_size,
//                          std::size_t max_iterations,
//                          const IndexerFunction<FeatureType>
//                          &indexer_function);
}

#endif
