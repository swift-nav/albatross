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

#ifndef ALBATROSS_UTILS_CONDITION_HPP_
#define ALBATROSS_UTILS_CONDITION_HPP_

// Unless otherwise stated, the functions in this file are derived
// from
//
//   A Survey of Condition Number Estimation for Triangular Matrices
//   N J Higham, 1987
//
//   http://eprints.ma.man.ac.uk/695/1/high87t.pdf
//
// All references to equation and algorithm numbers point to that
// document.
//
// Condition number estimates based on triangular matrices accept
// `Eigen::TriangularView`s for safety.  The matrix type being viewed
// is templated mainly to avoid dealing with the const-ness of the
// nested expression(s).

namespace albatross {

// Estimate the condition number of A based on its upper-triangular
// factor R.
//
// Empirically, this method typically underestimates the true
// condition number (as measured by the ratio of extremal singular
// values computed by the full SVD).
//
// https://en.wikipedia.org/wiki/Condition_number
//
// > If || . || is the matrix norm induced by the L^\inf (vector) norm
// > and A is lower triangular non-singular (i.e. a i i /= 0 for all
// > i), then
// > 
// >    \kappa(A) >= max_i(|a_{ii}|) / min_i(|a_{ii}|) 
// > 
// > recalling that the eigenvalues of any triangular matrix are
// > simply the diagonal entries.
// > 
// > The condition number computed with this norm is generally larger
// > than the condition number computed relative to the Euclidean
// > norm, but it can be evaluated more easily (and this is often the
// > only practicably computable condition number, when the problem to
// > solve involves a non-linear algebra[clarification needed], for
// > example when approximating irrational and transcendental
// > functions or numbers with numerical methods).
//
// https://eigen.tuxfamily.org/dox/group__TutorialReductionsVisitorsBroadcasting.html#TutorialReductionsVisitorsBroadcastingReductionsNorm
template <typename MatrixType>
inline double estimate_condition_number(
     const Eigen::TriangularView<MatrixType, Eigen::Upper> &R) {
  return R.nestedExpression().diagonal().array().abs().maxCoeff() /
    R.nestedExpression().diagonal().array().abs().minCoeff();
}

// Algorthm 2.1: Upper-bound |R^{-1}|_\inf using the comparison matrix
// M.
//
// Cost: n^2 / 2 flops
template <typename MatrixType>
double upper_bound_R_inv_inf(const Eigen::TriangularView<MatrixType, Eigen::Upper> &R) {
  Eigen::VectorXd z(R.rows());
  const Eigen::Index R_last = R.rows() - 1;
  z.coeffRef(R_last) = 1 / std::abs(R.coeff(R_last, R_last));
  for (Eigen::Index i = R_last - 1; i >= 0; --i) {
    const Eigen::Index jsize = R_last - i;
    assert(jsize > 0 && jsize < R.rows());
    const double s =
        1 + Eigen::MatrixXd(R).row(i).tail(jsize).array().abs().matrix().dot(
                z.tail(jsize));
    z(i) = s / std::abs(R.coeff(i, i));
  }
  return z.maxCoeff();
}

// Algorithm 2.1: Upper-bound |R^{-1}|_\inf using the comparison matrix
// M, modified to compute the 1-norm.  Page 579:
//
// > Algorithm 2.2 evaluates the \inf-norm of W(T)^{-1}, and the
// > 1-norm can be evaluated by applying a "lower triangular" version
// > of the algorithm to T^T (since |A|_1 = |A^T|_\inf).
//
// Cost: n^2 / 2 flops
template <typename MatrixType>
double upper_bound_R_inv_1(const Eigen::TriangularView<MatrixType, Eigen::Lower> &R) {
  Eigen::VectorXd z(R.rows());
  z.coeffRef(0) = 1 / std::abs(R.coeff(0, 0));
  for (Eigen::Index i = 1; i < R.rows(); ++i) {
    const double s =
        1 +
        Eigen::MatrixXd(R).row(i).tail(i).array().abs().matrix().dot(z.head(i));
    z(i) = s / std::abs(R.coeff(i, i));
  }
  return z.maxCoeff();
}

// The operator 2-norm requires computing the singular values, but
// we can upper-bound the operator 2-norm using Equation 1.6:
//
// |R|_2 <= sqrt(|R|_1 |R|_\inf)
template <typename MatrixType>
double upper_bound_R_2(const Eigen::TriangularView<MatrixType, Eigen::Upper> &R) {
  const auto Rdense = Eigen::MatrixXd(R);
  return std::sqrt(Rdense.cwiseAbs().rowwise().sum().maxCoeff() *
                   Rdense.cwiseAbs().colwise().sum().maxCoeff());
}
  
// Equation 2.4, upper-bounding the 2-norm of the inverse of
// upper-triangular matrix using \inf-norm and 1-norm
template <typename MatrixType>
double upper_bound_R_inv_2(const Eigen::TriangularView<MatrixType, Eigen::Upper> &R) {
  return std::sqrt(upper_bound_R_inv_inf(R) * upper_bound_R_inv_1(R.transpose()));
}

// Computes an upper bound on the condition number of A given its
// upper-triangular factor R.
//
// Empirically, this upper bound is quite loose and may approach the
// square of the true condition number (as measured by the ratio of
// extremal singular values computed by the full SVD).
template <typename MatrixType>
double upper_bound_condition_number(
       const Eigen::TriangularView<MatrixType, Eigen::Upper> &R) {
  // Definition of condition number
  return upper_bound_R_2(R) * upper_bound_R_inv_2(R);
}

// Compute \kappa_2, the 2-norm condition number of A, given its SVD.
template <typename MatrixType>
double condition_number(const Eigen::JacobiSVD<MatrixType> &svd) {
  return svd.singularValues()(0) /
    svd.singularValues()(svd.singularValues().size() - 1);
}

} // namespace albatross

#endif /* ALBATROSS_UTILS_CONDITION_HPP_ */
