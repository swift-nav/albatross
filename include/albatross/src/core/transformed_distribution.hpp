/*
 * Copyright (C) 2022 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef ALBATROSS_CORE_TRANSFORMED_DISTRIBUTION_H
#define ALBATROSS_CORE_TRANSFORMED_DISTRIBUTION_H

// Make it possible to premultiply matrices by distributions. There are a few
// things a user might want as a result, on one hand an arbitrary matrix by a
// distribution (marginal or joint) should result in a joint distribution.  In
// otherwords even if you start with independent variables (marginal)
// multiplying by a matrix will likely introduce correlations (joint).
// Considering only this case it would be tempting to write a multiplication
// operator which looks like:
//
//   JointDistribution operator*(const MatrixType &lhs,
//                               const MarginalDistrbution &rhs);
//
// However, if the user knows the result will stay independent, or if they
// don't care about the resulting correlations, they may only want the
// marginal distribution.  Here we add a TransformedDistribution helper
// class which makes the multiplication lazy and leaves it up to the
// user to decide what they want, enabling operations like:
//
//   MarginalDistribution x = (matrix * dist).marginal();
//   JointDistribution x = (matrix * dist).joint();
namespace albatross {

namespace details {

template <typename X> inline auto diagonal_wrapper(X &x) {
  return Eigen::DiagonalWrapper<X>(x);
}

template <typename MatrixType, typename DiagType>
inline Eigen::MatrixXd
product_sqrt(const Eigen::MatrixBase<MatrixType> &matrix,
             const Eigen::DiagonalBase<DiagType> &diag_matrix) {

  return matrix.derived() *
         diagonal_wrapper(diag_matrix.diagonal().array().sqrt().eval());
}

template <typename Lhs, typename Rhs>
inline Eigen::MatrixXd product_sqrt(const Eigen::MatrixBase<Lhs> &lhs,
                                    const Eigen::MatrixBase<Rhs> &rhs) {
  return lhs.derived() *
         Eigen::SerializableLDLT(rhs).sqrt_transpose().transpose();
}

template <typename MatrixType, typename DiagType>
inline Eigen::MatrixXd
product_sqrt(const Eigen::SparseMatrixBase<MatrixType> &matrix,
             const Eigen::DiagonalBase<DiagType> &diag_matrix) {
  return matrix.derived() *
         diagonal_wrapper(diag_matrix.diagonal().array().sqrt().eval());
}

template <typename SparseType, typename MatrixType>
inline Eigen::MatrixXd
product_sqrt(const Eigen::SparseMatrixBase<SparseType> &lhs,
             const Eigen::MatrixBase<MatrixType> &rhs) {
  return lhs.derived() *
         Eigen::SerializableLDLT(rhs).sqrt_transpose().transpose();
}

template <typename MatrixType, typename DistributionType>
struct TransformedDistribution {

  TransformedDistribution(
      const Eigen::MatrixBase<MatrixType> &matrix,
      const albatross::DistributionBase<DistributionType> &distribution)
      : mean(matrix.derived() * distribution.mean),
        covariance_product(matrix.derived(),
                           distribution.derived().covariance) {}

  TransformedDistribution(
      const Eigen::SparseMatrixBase<MatrixType> &matrix,
      const albatross::DistributionBase<DistributionType> &distribution)
      : mean(matrix.derived() * distribution.mean),
        covariance_product(matrix.derived(),
                           distribution.derived().covariance) {}

  MarginalDistribution marginal() const {
    const auto partial_variance =
        product_sqrt(covariance_product.lhs(), covariance_product.rhs());
    const Eigen::VectorXd variance =
        partial_variance.array().square().rowwise().sum();
    return MarginalDistribution(mean, variance);
  }

  JointDistribution joint() const {
    const auto partial_variance =
        product_sqrt(covariance_product.lhs(), covariance_product.rhs());
    return JointDistribution(mean,
                             partial_variance * partial_variance.transpose());
  }

  operator MarginalDistribution() const { return marginal(); }

  operator JointDistribution() const { return joint(); }

  Eigen::VectorXd mean;
  Eigen::Product<MatrixType, typename DistributionType::CovarianceType>
      covariance_product;
};

} // namespace details

template <typename MatrixType, typename DistributionType>
inline auto
operator*(const Eigen::MatrixBase<MatrixType> &matrix,
          const albatross::DistributionBase<DistributionType> &distribution) {
  return albatross::details::TransformedDistribution<MatrixType,
                                                     DistributionType>(
      matrix, distribution);
}

template <typename MatrixType, typename DistributionType>
inline auto
operator*(const Eigen::SparseMatrixBase<MatrixType> &matrix,
          const albatross::DistributionBase<DistributionType> &distribution) {
  return albatross::details::TransformedDistribution<MatrixType,
                                                     DistributionType>(
      matrix, distribution);
}

} // namespace albatross

#endif
