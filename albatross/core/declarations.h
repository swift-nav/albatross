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

template <typename ModelType, typename FeatureType> class Prediction;

template <typename ModelType> class Fit {};

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

struct NullGPImpl {};

template <typename FeatureType, typename CovarianceFunc, typename ImplType = NullGPImpl>
class GaussianProcessRegression;

struct NullLeastSquaresImpl {};

template <typename ImplType = NullLeastSquaresImpl>
class LeastSquares;



/*
 * Cross Validation
 */
// using FoldIndices = std::vector<std::size_t>;
// using FoldName = std::string;
// using FoldIndexer = std::map<FoldName, FoldIndices>;
// template <typename FeatureType>
// using IndexerFunction =
//    std::function<FoldIndexer(const RegressionDataset<FeatureType> &)>;

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
