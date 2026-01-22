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

#ifndef ALBATROSS_COVARIANCE_FUNCTIONS_SCALING_FUNCTION_H
#define ALBATROSS_COVARIANCE_FUNCTIONS_SCALING_FUNCTION_H

namespace albatross {

class ScalingFunction : public ParameterHandlingMixin {
public:
  virtual std::string get_name() const = 0;

  // A scaling function should also implement calls
  // for whichever types it is intended to scale using
  // the signature:
  //   double _call_impl(const X &x) const;
};

/*
 * A scaling term is not actually a covariance function
 * in the rigorous sense.  It doesn't describe the uncertainty
 * between variables, but instead operates deterministically
 * on other uncertain variables.  For instance, you may have
 * some random variable,
 *     y ~ N(0, S)  with  S_ij = cov(y_i, y_j)
 * And you may then make observations of that random variable
 * but through a known transformation,
 *     z = f(y) * y
 * where f is a determinstic function of y that returns a scalar.
 * You might then ask what the covariance between two elements in
 * z is which would be given by,
 *     cov(z_i, z_j) = f(y_i) * cov(y_i, y_j) * f(y_j)
 * but you might also be interested in the covariance between
 * some y_i and an observation z_j,
 *     cov(y_i, z_j) = cov(y_i, y_j) * f(y_j)
 * Here we see that for a typical covariance term, the covariance
 * is only defined for two pairs of the same type, in this case
 *     operator()(Y &y, Y &y)
 * but by multiplying by a ScalingTerm we end up with definitions
 * for,
 *     operator()(Y &y, Z &z)
 * which provides us with a way of computing the covariance between
 * some hidden representation of a variable (y) and the actual
 * observations (z) using a single determinstic mapping (f).
 *
 * This might be better explained by example which can be found
 * in the tests (test_scaling_function).
 */
template <typename ScalingFunction>
class ScalingTerm : public CovarianceFunction<ScalingTerm<ScalingFunction>> {
public:
  ScalingTerm() : scaling_function_() {}

  ScalingTerm(const ScalingFunction &func) : scaling_function_(func) {}

  std::string get_name() const { return scaling_function_.get_name(); }

  ParameterStore get_params() const override {
    return scaling_function_.get_params();
  }

  void set_param(const ParameterKey &name, const Parameter &param) override {
    scaling_function_.set_param(name, param);
  }

  /*
   * If both Scaling and Covariance have a valid call method for the types X
   * and Y this will return the product of the two.
   */
  template <typename X, typename Y,
            typename std::enable_if<
                (has_valid_call_impl<ScalingFunction, X &>::value &&
                 has_valid_call_impl<ScalingFunction, Y &>::value),
                int>::type = 0>
  double _call_impl(const X &x, const Y &y) const {
    return this->scaling_function_._call_impl(x) *
           this->scaling_function_._call_impl(y);
  }

  /*
   * If only one of the types has a scaling function we ignore the other.
   */
  template <typename X, typename Y,
            typename std::enable_if<
                (!has_valid_call_impl<ScalingFunction, X &>::value &&
                 has_valid_call_impl<ScalingFunction, Y &>::value),
                int>::type = 0>
  double _call_impl(const X &, const Y &y) const {
    return this->scaling_function_._call_impl(y);
  }

  template <typename X, typename Y,
            typename std::enable_if<
                (has_valid_call_impl<ScalingFunction, X &>::value &&
                 !has_valid_call_impl<ScalingFunction, Y &>::value),
                int>::type = 0>
  double _call_impl(const X &x, const Y &) const {
    return this->scaling_function_._call_impl(x);
  }

  /*
   * Batch covariance methods for ScalingTerm.
   * These compute the scaling factors as vectors and form the result matrix.
   */

  // Helper to compute scaling factors for a vector of features
  template <typename X>
  Eigen::VectorXd compute_scale_vector(const std::vector<X> &xs) const {
    const Eigen::Index n = cast::to_index(xs.size());
    Eigen::VectorXd scales(n);
    for (Eigen::Index i = 0; i < n; ++i) {
      scales[i] = this->scaling_function_._call_impl(xs[cast::to_size(i)]);
    }
    return scales;
  }

  // Both X and Y have scaling: outer product of scale vectors
  template <typename X, typename Y,
            typename std::enable_if<
                (has_valid_call_impl<ScalingFunction, X &>::value &&
                 has_valid_call_impl<ScalingFunction, Y &>::value),
                int>::type = 0>
  Eigen::MatrixXd _call_impl_vector(const std::vector<X> &xs,
                                    const std::vector<Y> &ys,
                                    ThreadPool * /*pool*/ = nullptr) const {
    const Eigen::VectorXd scale_x = compute_scale_vector(xs);
    const Eigen::VectorXd scale_y = compute_scale_vector(ys);
    return scale_x * scale_y.transpose();
  }

  // Only X has scaling: replicate scale_x across columns
  template <typename X, typename Y,
            typename std::enable_if<
                (has_valid_call_impl<ScalingFunction, X &>::value &&
                 !has_valid_call_impl<ScalingFunction, Y &>::value),
                int>::type = 0>
  Eigen::MatrixXd _call_impl_vector(const std::vector<X> &xs,
                                    const std::vector<Y> &ys,
                                    ThreadPool * /*pool*/ = nullptr) const {
    const Eigen::VectorXd scale_x = compute_scale_vector(xs);
    const Eigen::Index m = cast::to_index(ys.size());
    // Each row i is filled with scale_x[i]
    return scale_x.replicate(1, m);
  }

  // Only Y has scaling: replicate scale_y across rows
  template <typename X, typename Y,
            typename std::enable_if<
                (!has_valid_call_impl<ScalingFunction, X &>::value &&
                 has_valid_call_impl<ScalingFunction, Y &>::value),
                int>::type = 0>
  Eigen::MatrixXd _call_impl_vector(const std::vector<X> &xs,
                                    const std::vector<Y> &ys,
                                    ThreadPool * /*pool*/ = nullptr) const {
    const Eigen::VectorXd scale_y = compute_scale_vector(ys);
    const Eigen::Index n = cast::to_index(xs.size());
    // Each column j is filled with scale_y[j]
    return scale_y.transpose().replicate(n, 1);
  }

  // Symmetric batch: outer product of scale vector with itself
  template <typename X,
            typename std::enable_if<
                has_valid_call_impl<ScalingFunction, X &>::value, int>::type = 0>
  Eigen::MatrixXd _call_impl_vector(const std::vector<X> &xs,
                                    ThreadPool * /*pool*/ = nullptr) const {
    const Eigen::VectorXd scale_x = compute_scale_vector(xs);
    return scale_x * scale_x.transpose();
  }

  // Diagonal batch: squared scaling factors
  template <typename X,
            typename std::enable_if<
                has_valid_call_impl<ScalingFunction, X &>::value, int>::type = 0>
  Eigen::VectorXd _call_impl_vector_diagonal(const std::vector<X> &xs,
                                             ThreadPool * /*pool*/ = nullptr) const {
    const Eigen::VectorXd scale_x = compute_scale_vector(xs);
    return scale_x.array().square().matrix();
  }

private:
  ScalingFunction scaling_function_;
};

/*
 * Product in the form:  scaling_term * other
 */
template <class ScalingFunction, class RHS>
class ProductOfCovarianceFunctions<ScalingTerm<ScalingFunction>, RHS>
    : public CovarianceFunction<
          ProductOfCovarianceFunctions<ScalingTerm<ScalingFunction>, RHS>> {
public:
  using LHS = ScalingTerm<ScalingFunction>;

  ProductOfCovarianceFunctions() : lhs_(), rhs_() {}
  ProductOfCovarianceFunctions(const LHS &lhs, const RHS &rhs)
      : lhs_(lhs), rhs_(rhs) {}

  std::string name() const {
    return "(" + lhs_.get_name() + "*" + rhs_.get_name() + ")";
  }

  ParameterStore get_params() const override {
    return map_join(lhs_.get_params(), rhs_.get_params());
  }

  void set_param(const ParameterKey &name, const Parameter &param) override {
    const bool success = set_param_if_exists_in_any(name, param, &lhs_, &rhs_);
    ALBATROSS_ASSERT(success);
  }

  /*
   * If both LHS and RHS have a valid call method for the types X and Y
   * this will return the product of the two.
   */
  template <typename X, typename Y,
            typename std::enable_if<(has_equivalent_caller<LHS, X, Y>::value &&
                                     has_equivalent_caller<RHS, X, Y>::value),
                                    int>::type = 0>
  double _call_impl(const X &x, const Y &y) const {
    double output = this->lhs_(x, y);
    if (output != 0.) {
      output *= this->rhs_(x, y);
    }
    return output;
  }

  /*
   * If only RHS has a valid call method we ignore L.
   */
  template <typename X, typename Y,
            typename std::enable_if<(!has_equivalent_caller<LHS, X, Y>::value &&
                                     has_equivalent_caller<RHS, X, Y>::value),
                                    int>::type = 0>
  double _call_impl(const X &x, const Y &y) const {
    return this->rhs_(x, y);
  }

  /*
   * Batch covariance methods for Product(ScalingTerm, RHS).
   * These delegate to the RHS batch method and then apply scaling element-wise.
   */

  // Cross-covariance batch: both LHS and RHS apply
  template <typename X, typename Y,
            typename std::enable_if<(has_equivalent_caller<LHS, X, Y>::value &&
                                     has_equivalent_caller<RHS, X, Y>::value),
                                    int>::type = 0>
  Eigen::MatrixXd _call_impl_vector(const std::vector<X> &xs,
                                    const std::vector<Y> &ys,
                                    ThreadPool *pool = nullptr) const {
    // Get the scaling matrix from LHS
    const Eigen::MatrixXd scale_mat = this->lhs_(xs, ys, pool);
    // Get the covariance matrix from RHS
    const Eigen::MatrixXd cov_mat = this->rhs_(xs, ys, pool);
    // Element-wise product
    return scale_mat.cwiseProduct(cov_mat);
  }

  // Cross-covariance batch: only RHS applies (LHS doesn't apply to these types)
  template <typename X, typename Y,
            typename std::enable_if<(!has_equivalent_caller<LHS, X, Y>::value &&
                                     has_equivalent_caller<RHS, X, Y>::value),
                                    int>::type = 0>
  Eigen::MatrixXd _call_impl_vector(const std::vector<X> &xs,
                                    const std::vector<Y> &ys,
                                    ThreadPool *pool = nullptr) const {
    return this->rhs_(xs, ys, pool);
  }

  // Symmetric batch: both LHS and RHS apply
  template <typename X,
            typename std::enable_if<(has_equivalent_caller<LHS, X, X>::value &&
                                     has_equivalent_caller<RHS, X, X>::value),
                                    int>::type = 0>
  Eigen::MatrixXd _call_impl_vector(const std::vector<X> &xs,
                                    ThreadPool *pool = nullptr) const {
    const Eigen::MatrixXd scale_mat = this->lhs_(xs, pool);
    const Eigen::MatrixXd cov_mat = this->rhs_(xs, pool);
    return scale_mat.cwiseProduct(cov_mat);
  }

  // Diagonal batch: both LHS and RHS apply
  template <typename X,
            typename std::enable_if<(has_equivalent_caller<LHS, X, X>::value &&
                                     has_equivalent_caller<RHS, X, X>::value),
                                    int>::type = 0>
  Eigen::VectorXd _call_impl_vector_diagonal(const std::vector<X> &xs,
                                             ThreadPool *pool = nullptr) const {
    const Eigen::VectorXd scale_diag = this->lhs_.diagonal(xs, pool);
    const Eigen::VectorXd cov_diag = this->rhs_.diagonal(xs, pool);
    return scale_diag.cwiseProduct(cov_diag);
  }

protected:
  LHS lhs_;
  RHS rhs_;
  friend class CallTrace<ProductOfCovarianceFunctions<LHS, RHS>>;
};

/*
 * Product in the form:  other * scaling_term
 */
template <class LHS, class ScalingFunction>
class ProductOfCovarianceFunctions<LHS, ScalingTerm<ScalingFunction>>
    : public CovarianceFunction<
          ProductOfCovarianceFunctions<LHS, ScalingTerm<ScalingFunction>>> {
public:
  using RHS = ScalingTerm<ScalingFunction>;

  ProductOfCovarianceFunctions() : lhs_(), rhs_() {}
  ProductOfCovarianceFunctions(const LHS &lhs, const RHS &rhs)
      : lhs_(lhs), rhs_(rhs) {}

  std::string name() const {
    return "(" + lhs_.get_name() + "*" + rhs_.get_name() + ")";
  }

  ParameterStore get_params() const override {
    return map_join(lhs_.get_params(), rhs_.get_params());
  }

  void set_param(const ParameterKey &name, const Parameter &param) override {
    const bool success = set_param_if_exists_in_any(name, param, &lhs_, &rhs_);
    ALBATROSS_ASSERT(success);
  }

  /*
   * If both LHS and RHS have a valid call method for the types X and Y
   * this will return the product of the two.
   */
  template <typename X, typename Y,
            typename std::enable_if<(has_equivalent_caller<LHS, X, Y>::value &&
                                     has_equivalent_caller<RHS, X, Y>::value),
                                    int>::type = 0>
  double _call_impl(const X &x, const Y &y) const {
    double output = this->lhs_(x, y);
    if (output != 0.) {
      output *= this->rhs_(x, y);
    }
    return output;
  }

  /*
   * If only LHS has a valid call method we ignore R.
   */
  template <typename X, typename Y,
            typename std::enable_if<(has_equivalent_caller<LHS, X, Y>::value &&
                                     !has_equivalent_caller<RHS, X, Y>::value),
                                    int>::type = 0>
  double _call_impl(const X &x, const Y &y) const {
    return this->lhs_(x, y);
  }

  /*
   * Batch covariance methods for Product(LHS, ScalingTerm).
   * These delegate to the LHS batch method and then apply scaling element-wise.
   */

  // Cross-covariance batch: both LHS and RHS apply
  template <typename X, typename Y,
            typename std::enable_if<(has_equivalent_caller<LHS, X, Y>::value &&
                                     has_equivalent_caller<RHS, X, Y>::value),
                                    int>::type = 0>
  Eigen::MatrixXd _call_impl_vector(const std::vector<X> &xs,
                                    const std::vector<Y> &ys,
                                    ThreadPool *pool = nullptr) const {
    // Get the covariance matrix from LHS
    const Eigen::MatrixXd cov_mat = this->lhs_(xs, ys, pool);
    // Get the scaling matrix from RHS
    const Eigen::MatrixXd scale_mat = this->rhs_(xs, ys, pool);
    // Element-wise product
    return cov_mat.cwiseProduct(scale_mat);
  }

  // Cross-covariance batch: only LHS applies (RHS doesn't apply to these types)
  template <typename X, typename Y,
            typename std::enable_if<(has_equivalent_caller<LHS, X, Y>::value &&
                                     !has_equivalent_caller<RHS, X, Y>::value),
                                    int>::type = 0>
  Eigen::MatrixXd _call_impl_vector(const std::vector<X> &xs,
                                    const std::vector<Y> &ys,
                                    ThreadPool *pool = nullptr) const {
    return this->lhs_(xs, ys, pool);
  }

  // Symmetric batch: both LHS and RHS apply
  template <typename X,
            typename std::enable_if<(has_equivalent_caller<LHS, X, X>::value &&
                                     has_equivalent_caller<RHS, X, X>::value),
                                    int>::type = 0>
  Eigen::MatrixXd _call_impl_vector(const std::vector<X> &xs,
                                    ThreadPool *pool = nullptr) const {
    const Eigen::MatrixXd cov_mat = this->lhs_(xs, pool);
    const Eigen::MatrixXd scale_mat = this->rhs_(xs, pool);
    return cov_mat.cwiseProduct(scale_mat);
  }

  // Diagonal batch: both LHS and RHS apply
  template <typename X,
            typename std::enable_if<(has_equivalent_caller<LHS, X, X>::value &&
                                     has_equivalent_caller<RHS, X, X>::value),
                                    int>::type = 0>
  Eigen::VectorXd _call_impl_vector_diagonal(const std::vector<X> &xs,
                                             ThreadPool *pool = nullptr) const {
    const Eigen::VectorXd cov_diag = this->lhs_.diagonal(xs, pool);
    const Eigen::VectorXd scale_diag = this->rhs_.diagonal(xs, pool);
    return cov_diag.cwiseProduct(scale_diag);
  }

protected:
  LHS lhs_;
  RHS rhs_;
  friend class CallTrace<ProductOfCovarianceFunctions<LHS, RHS>>;
};

} // namespace albatross
#endif
