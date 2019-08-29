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

#ifndef ALBATROSS_COVARIANCE_FUNCTION_REPRESENTATIONS_HPP_
#define ALBATROSS_COVARIANCE_FUNCTION_REPRESENTATIONS_HPP_

/*
 * These "Representations" relate to matrices build from covariance
 * functions, namely Positive Semi Definite (PSD) matrices.
 *
 * The most popular representation for such matrices is the
 * Cholesky, or related LDLT decomposition:
 *
 *   A = L D L^T
 *
 * But sometimes there are more efficient, or numerically
 * stable approaches which depend on the application.
 *
 * A covariance Representation is effectively just an
 * approach for efficiently storing and inverting the matrix
 * in question.
 */
namespace albatross {

/*
 * The ExplainedCovariance represents a matrix of the form,
 *
 *     S = A B^{-1} A
 *
 * Where A is referred to as the outer and B as the inner matrix.
 * The solve (inverse) can then be performed by taking,
 *
 *     x = A^-1 B A^-1 rhs
 *
 * One example of where this sort of form shows up is in posterior
 * distributions of Gaussian processes.  Posterior covariances
 * often take the form:
 *
 *     post = prior - cross^T dependent^-1 cross
 *
 * Ie, the posterior covariance would be the prior covariance minus
 * the explained ( cross^T dependent^-1 cross ) covariance.
 * Here the explained covariance consists of the cross term which
 * may represent the relationship between the variable in
 * question and some dependent variable, and the dependent^-1 term
 * represents the inverse of the prior on the dependent variable.
 *
 * While (typically) the prior covariances are positive definite
 * it's much more likely that the explained covariance would be
 * positive SEMI definite which would happen in the case of an
 * underdetermined set of observations.  This particular
 * CovarianceRepresentation therefore avoids the inversion of B
 * if possible, under the assumption that it may be singular.
 */
struct ExplainedCovariance {

  ExplainedCovariance(){};

  ExplainedCovariance(const Eigen::MatrixXd &outer,
                      const Eigen::MatrixXd &inner_) {
    outer_ldlt = outer.ldlt();
    inner = inner_;
  }

  /*
   * Returns S^-1 rhs by using S^-1 = A^-1 B A^-1
   */
  Eigen::MatrixXd solve(const Eigen::MatrixXd &rhs) const {
    return outer_ldlt.solve(inner * outer_ldlt.solve(rhs));
  }

  bool operator==(const ExplainedCovariance &rhs) const {
    return (outer_ldlt == rhs.outer_ldlt && inner == rhs.inner);
  }

  Eigen::Index rows() const { return inner.rows(); }

  Eigen::Index cols() const { return inner.cols(); }

  Eigen::SerializableLDLT outer_ldlt;
  Eigen::MatrixXd inner;
};

/*
 * Simply stores a pre-computed inverse.
 */
struct DirectInverse {
  DirectInverse(){};

  DirectInverse(const Eigen::MatrixXd &inverse) : inverse_(inverse){};

  Eigen::MatrixXd solve(const Eigen::MatrixXd &rhs) const {
    return inverse_ * rhs;
  }

  bool operator==(const DirectInverse &rhs) const {
    return (inverse_ == rhs.inverse_);
  }

  Eigen::Index rows() const { return inverse_.rows(); }

  Eigen::Index cols() const { return inverse_.cols(); }

  Eigen::MatrixXd inverse_;
};

} // namespace albatross

#endif /* ALBATROSS_COVARIANCE_FUNCTION_REPRESENTATIONS_HPP_ */
