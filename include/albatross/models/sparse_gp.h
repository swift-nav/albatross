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

#ifndef INCLUDE_ALBATROSS_MODELS_SPARSE_GP_H_
#define INCLUDE_ALBATROSS_MODELS_SPARSE_GP_H_

namespace albatross {

template <typename CovFunc, typename InducingPointStrategy,
          typename IndexingFunction>
class SparseGaussianProcessRegression;

std::vector<double> inline linspace(double a, double b, std::size_t n) {
  double h = (b - a) / static_cast<double>(n - 1);
  std::vector<double> xs(n);
  typename std::vector<double>::iterator x;
  double val;
  for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h)
    *x = val;
  return xs;
}

struct UniformlySpacedInducingPoints {

  UniformlySpacedInducingPoints(std::size_t num_points_ = 10)
      : num_points(num_points_) {}

  std::vector<double> operator()(const std::vector<double> &features) const {
    double min = *std::min_element(features.begin(), features.end());
    double max = *std::max_element(features.begin(), features.end());

    return linspace(min, max, num_points);
  }

  std::size_t num_points;
};

/*
 *  This class implements an approximation technique for Gaussian processes
 * which relies on an assumption that all observations are independent (or
 * groups of observations are independent) conditional on a set of inducing
 * points.  The method is based off:
 *
 *     [1] Sparse Gaussian Processes using Pseudo-inputs
 *     Edward Snelson, Zoubin Ghahramani
 *     http://www.gatsby.ucl.ac.uk/~snelson/SPGP_up.pdf
 *
 *  Though the code uses notation closer to that used in this (excellent)
 * overview of these methods:
 *
 *     [2] A Unifying View of Sparse Approximate Gaussian Process Regression
 *     Joaquin Quinonero-Candela, Carl Edward Rasmussen
 *     http://www.jmlr.org/papers/volume6/quinonero-candela05a/quinonero-candela05a.pdf
 *
 *  Very broadly speaking this method starts with a prior over the observations,
 *
 *     [f] ~ N(0, K_ff)
 *
 *  where K_ff(i, j) = covariance_function(features[i], features[j]) and f
 * represents the function value.
 *
 *  It then uses a set of inducing points, u, and makes some assumptions about
 * the conditional distribution:
 *
 *     [f|u] ~ N(K_fu K_uu^-1 u, K_ff - Q_ff)
 *
 *  Where Q_ff = K_fu K_uu^-1 K_uf represents the variance in f that is
 * explained by u.
 *
 *  For FITC (Fully Independent Training Contitional) the assumption is that
 * K_ff - Qff is diagonal, for PITC (Partially Independent Training Conditional)
 * that it is block diagonal.  These assumptions lead to an efficient way of
 * inferring the posterior distribution for some new location f*,
 *
 *     [f*|f=y] ~ N(K_*u S K_uf A^-1 y, K_** − Q_** + K_*u S K_u*)
 *
 *  Where S = (K_uu + K_uf A^-1 K_fu)^-1 and A = diag(K_ff - Q_ff) and "diag"
 * may mean diagonal or block diagonal.  Regardless we end up with O(m^2n)
 * complexity instead of O(n^3) of direct Gaussian processes.  (Note that in [2]
 * S is called sigma and A is lambda.)
 *
 *  Of course, the implementation details end up somewhat more complex in order
 * to improve numerical stability.  A few great resources were heavily used to
 * get those deails straight:
 *
 *     - https://bwengals.github.io/pymc3-fitcvfe-implementation-notes.html
 *     - https://github.com/SheffieldML/GPy see fitc.py
 */
template <typename CovFunc, typename InducingPointStrategy,
          typename IndexingFunction>
class SparseGaussianProcessRegression
    : public GaussianProcessBase<
          CovFunc, SparseGaussianProcessRegression<
                       CovFunc, InducingPointStrategy, IndexingFunction>> {

public:
  using Base = GaussianProcessBase<
      CovFunc, SparseGaussianProcessRegression<CovFunc, InducingPointStrategy,
                                               IndexingFunction>>;

  SparseGaussianProcessRegression() : Base(){};
  SparseGaussianProcessRegression(CovFunc &covariance_function)
      : Base(covariance_function){};
  SparseGaussianProcessRegression(
      CovFunc &covariance_function,
      InducingPointStrategy &inducing_point_strategy_,
      IndexingFunction &independent_group_indexing_function_,
      const std::string &model_name)
      : Base(covariance_function, model_name),
        inducing_point_strategy(inducing_point_strategy_),
        independent_group_indexing_function(
            independent_group_indexing_function_){};
  SparseGaussianProcessRegression(CovFunc &covariance_function,
                                  const std::string &model_name)
      : Base(covariance_function, model_name){};

  template <typename FeatureType>
  auto _fit_impl(const std::vector<FeatureType> &features,
                 const MarginalDistribution &targets) const {
    static_assert(std::is_same<IndexingFunction, LeaveOneOut>::value,
                  "Only FITC is currently implemented, but by allowing "
                  "arbitrary groups this could turn into PITC");

    // Determine the set of inducing points, u.
    auto u = inducing_point_strategy(features);

    Eigen::Index n = targets.mean.size();
    Eigen::Index m = static_cast<Eigen::Index>(u.size());

    // Create the covariance matrices we need.
    Eigen::VectorXd K_ff_diag = this->covariance_function_.diagonal(features);
    if (targets.has_covariance()) {
      K_ff_diag += targets.covariance.diagonal();
    }
    Eigen::MatrixXd K_fu = this->covariance_function_(features, u);
    Eigen::MatrixXd K_uu = this->covariance_function_(u);

    auto K_uu_llt = K_uu.llt();
    // P is such that:
    //     Q_ff = K_fu K_uu^-1 K_uf
    //          = K_fu L^-T L^-1 K_uf
    //          = P^T P
    Eigen::MatrixXd P = K_uu_llt.matrixL().solve(K_fu.transpose());

    // Efficiently compute the diagonal diag[Q_ff].
    Eigen::VectorXd Q_ff_diag = P.colwise().squaredNorm();

    Eigen::VectorXd A = K_ff_diag - Q_ff_diag;

    if (A.minCoeff() < 1e-6) {
      // It's possible that the inducing points will perfectly describe
      // some of the data, in which case we need to add a bit of extra
      // noise to make sure lambda is invertible.
      A += 1e-6 * Eigen::VectorXd::Ones(n);
    }
    /*
     *
     * The end goal here is to produce a vector, v, and matrix, C, such that
     * for a prediction, f*, we can do,
     *
     *     [f*|f=y] ~ N(K_*u * v , K_** - K_*u * C^-1 * K_u*)
     *
     *  and it would match the desired prediction described above,
     *
     *     [f*|f=y] ~ N(K_*u S K_uf^-1 A^-1 y, K_** − Q_** + K_*u S K_u*)
     *
     *  we can find v easily,
     *
     *     v = S K_uf^-1 A^-1 y
     *
     *  and to get C we need to do some algebra,
     *
     *     K_** - K_*u * C^-1 * K_u* = K_** - Q_** + K_*u S K_u*
     *                               = K_** - K_*u (K_uu^-1 - S) K_u*
     *  which leads to:
     *     C^-1 = K_uu^-1 - S
     *                                                  (Expansion of S)
     *          = K_uu^-1 - (K_uu + K_uf A^-1 K_fu)^-1
     *                                        (Woodbury Matrix Identity)
     *          = (K_uu^-1 K_uf (A + K_fu K_uu^-1 K_uf)^-1 K_fu K_uu^-1)
     *                                   (LL^T = K_uu and P = L^-1 K_uf)
     *          = L^-T P (A + P^T P)^-1 P^T L^-1
     *                                        (Searle Set of Identities)
     *          = L^-T P A^-1 P^T (I + P A^-1 P^T)^-1 L^-1
     *                         (B = (I + P A^-1 P^T) and R = A^-1/2 P^T)
     *          = L^-T R^T R B^-1 L^-1
     *
     *  taking the inverse of that then gives us:
     *
     *      C   = L B (R^T R)^-1 L^T
     *
     *  reusing some of the precomputed values there leads to:
     *
     *     v = L^-T B^-1 P * A^-1 y
     */
    Eigen::VectorXd A_sqrt = A.array().sqrt();
    Eigen::MatrixXd RtR = A_sqrt.asDiagonal().inverse() * P.transpose();
    RtR = RtR.transpose() * RtR;
    Eigen::MatrixXd B = Eigen::MatrixXd::Identity(m, m) + RtR;

    auto B_ldlt = B.ldlt();

    Eigen::VectorXd v = P * (A.asDiagonal().inverse() * targets.mean);
    v = B_ldlt.solve(v);
    v = K_uu_llt.matrixL().transpose().solve(v);

    Eigen::MatrixXd L_uu_inv =
        K_uu_llt.matrixL().solve(Eigen::MatrixXd::Identity(m, m));
    Eigen::MatrixXd RtRBiLi = RtR * B_ldlt.solve(L_uu_inv);
    Eigen::MatrixXd LT = K_uu_llt.matrixL().transpose();
    Eigen::MatrixXd C = K_uu_llt.matrixL() * B * RtR.ldlt().solve(LT);

    return typename Base::template GPFitType<FeatureType>(u, C.ldlt(), v);
  }

  using Base::_predict_impl;

  InducingPointStrategy inducing_point_strategy;
  IndexingFunction independent_group_indexing_function;
};

template <typename CovFunc, typename InducingPointStrategy,
          typename IndexingFunction>
auto sparse_gp_from_covariance(CovFunc covariance_function,
                               InducingPointStrategy &strategy,
                               IndexingFunction &index_function,
                               const std::string &model_name) {
  return SparseGaussianProcessRegression<CovFunc, InducingPointStrategy,
                                         IndexingFunction>(
      covariance_function, strategy, index_function, model_name);
};
} // namespace albatross

#endif /* INCLUDE_ALBATROSS_MODELS_SPARSE_GP_H_ */
