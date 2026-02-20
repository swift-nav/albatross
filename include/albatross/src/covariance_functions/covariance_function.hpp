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

#ifndef ALBATROSS_COVARIANCE_FUNCTIONS_COVARIANCE_FUNCTION_H
#define ALBATROSS_COVARIANCE_FUNCTIONS_COVARIANCE_FUNCTION_H

namespace albatross {

/*
 * CovarianceFunction is a CRTP base class which can be used in a
 * way similar to a polymorphic abstract class.  For example if
 * you want to implement a new covariance function you'll need to
 * do something like:
 *
 *   class MyCovFunc : public CovarianceFunction<MyCovFunc> {
 *
 *     std::string get_name() const {return "my_cov_func";}
 *
 *     double _call_impl(const X &x, const X &y) const {
 *       return covariance_between_x_and_y(x, y);
 *     }
 *
 *   }
 *
 * so in a way you can think of CovarianceFunction as an abstract
 * base class with API,
 *
 *   class CovarianceFunction {
 *
 *     virtual std::string get_name() const = 0;
 *
 *     template <typename X, typename Y=X>
 *     double _call_impl(const X &x, const Y &y) const = 0;
 *
 *     template <typename X, typename Y=X>
 *     double operator()(const X &x, const Y &y) const {
 *       return this->_call_impl(x, y);
 *     }
 *
 *   }
 *
 * But note that you can't actually have an abstract type with virtual methods
 * which are templated, so the abstract class approach in that example would
 * never compile.
 *
 * This is where CRTP comes in handy.  To write a new CovarianceFunction you
 * need to provide `_call_impl(` functions for any pair of types you'd like to
 * have defined.  The CovarianceFunction CRTP base class is then capable of
 * detecting which methods are required at compile time and creating the
 * corresponding operator() methods.  This also makes it possible to compose
 * CovarianceFunctions (see operator* and operator+).
 *
 */
template <typename Derived>
class CovarianceFunction : public ParameterHandlingMixin {
private:
  // Declaring these private makes it impossible to accidentally do things like:
  //     class A : public CovarianceFunction<B> {}
  // or
  //     using A = CovarianceFunction<B>;
  //
  // which if unchecked can lead to some very strange behavior.
  CovarianceFunction() : ParameterHandlingMixin() {}
  friend Derived;

  template <typename X, typename Y> double call(const X &x, const Y &y) const {
    return DefaultCaller::call(derived(), x, y);
  }

public:
  static_assert(!is_complete<Derived>::value,
                "\n\nPassing a complete type in as template parameter "
                "implies you aren't using CRTP.  Implementations "
                "of a CovarianceFunction should look something like:\n"
                "\n\tclass Foo : public CovarianceFunction<Foo> {"
                "\n\t\tdouble _call_impl(const X &x, const Y &y) const;"
                "\n\t\t..."
                "\n\t}\n");

  template <typename DummyType = Derived,
            typename std::enable_if<!has_name<DummyType>::value, int>::type = 0>
  std::string get_name() const {
    static_assert(std::is_same<DummyType, Derived>::value,
                  "never do covariance_function.get_name<T>()");
    return typeid(Derived).name();
  }

  template <typename DummyType = Derived,
            typename std::enable_if<has_name<DummyType>::value, int>::type = 0>
  std::string get_name() const {
    static_assert(std::is_same<DummyType, Derived>::value,
                  "never do covariance_function.get_name<T>()");
    return derived().name();
  }

  std::string pretty_string() const {
    std::ostringstream ss;
    ss << get_name() << std::endl;
    ss << ParameterHandlingMixin::pretty_string();
    return ss.str();
  }

  /*
   * Scalar covariance evaluation.
   *
   * When _call_impl_vector exists, we ALWAYS use it (synthesizing scalar
   * from batch). This ensures consistency between scalar and matrix calls.
   * Only when batch is not available do we fall back to pointwise _call_impl.
   */

  // Primary: use batch when available (synthesize scalar from batch)
  template <
      typename X, typename Y,
      typename std::enable_if<has_valid_call_impl_vector<Derived, X, Y>::value,
                              int>::type = 0>
  double operator()(const X &x, const Y &y) const {
    const std::vector<X> xs{x};
    const std::vector<Y> ys{y};
    return derived()._call_impl_vector(xs, ys, nullptr)(0, 0);
  }

  // Fallback: use pointwise when batch is not available
  template <typename X, typename Y,
            typename std::enable_if<
                (!has_valid_call_impl_vector<Derived, X, Y>::value &&
                 has_valid_caller<Derived, X, Y>::value),
                int>::type = 0>
  auto operator()(const X &x, const Y &y) const {
    return call(x, y);
  }

  /*
   * Covariance between each element and every other in a vector.
   * Enabled if EITHER pointwise OR symmetric batch is defined.
   */
  template <typename X,
            typename std::enable_if<
                has_valid_caller_or_batch_symmetric<Derived, X>::value,
                int>::type = 0>
  Eigen::MatrixXd operator()(const std::vector<X> &xs,
                             ThreadPool *pool = nullptr) const {
    return DefaultCaller::call_vector(derived(), xs, pool);
  }

  /*
   * Cross covariance between two vectors of (possibly) different types.
   * Enabled if EITHER pointwise (_call_impl) OR batch (_call_impl_vector) is
   * defined.
   */
  template <typename X, typename Y,
            typename std::enable_if<
                has_valid_caller_or_batch<Derived, X, Y>::value, int>::type = 0>
  Eigen::MatrixXd operator()(const std::vector<X> &xs, const std::vector<Y> &ys,
                             ThreadPool *pool = nullptr) const {
    return DefaultCaller::call_vector(derived(), xs, ys, pool);
  }

  /*
   * Diagonal of the covariance matrix.
   * Enabled if EITHER pointwise OR diagonal batch is defined.
   */
  template <
      typename X,
      typename std::enable_if<
          has_valid_caller_or_batch_diagonal<Derived, X>::value, int>::type = 0>
  Eigen::VectorXd diagonal(const std::vector<X> &xs,
                           ThreadPool *pool = nullptr) const {
    return DefaultCaller::call_vector_diagonal(derived(), xs, pool);
  }

  template <typename X,
            typename std::enable_if<has_valid_ssr_impl<Derived, X>::value,
                                    int>::type = 0>
  auto state_space_representation(const std::vector<X> &xs) const {
    return derived()._ssr_impl(xs);
  }

  /*
   * Stubs to catch the case where a covariance function was called
   * with arguments that aren't supported.
   */
  template <typename X, typename std::enable_if<
                            (!has_valid_caller<Derived, X, X>::value &&
                             !has_possible_call_impl<Derived, X, X>::value),
                            int>::type = 0>
  void operator()(const X &x ALBATROSS_UNUSED) const ALBATROSS_FAIL(
      X, "No public method with signature 'double "
         "Derived::_call_impl(const X&, const X&) const'")

      template <typename X, typename std::enable_if<
                                (!has_valid_caller<Derived, X, X>::value &&
                                 has_invalid_call_impl<Derived, X, X>::value),
                                int>::type = 0>
      void operator()(const X &x ALBATROSS_UNUSED) const
      ALBATROSS_FAIL(X, "Incorrectly defined method 'double "
                        "Derived::_call_impl(const X&, const X&) const'")

          template <typename X,
                    typename std::enable_if<
                        !has_valid_caller_or_batch_diagonal<Derived, X>::value,
                        int>::type = 0>
          void diagonal(const std::vector<X> &xs ALBATROSS_UNUSED) const
      ALBATROSS_FAIL(
          X, "No public method with signature 'double "
             "Derived::_call_impl(const X&, const X&) const' or "
             "'Eigen::VectorXd Derived::_call_impl_vector_diagonal(const "
             "std::vector<X>&, ThreadPool*) const'")

          CallTrace<Derived> call_trace() const;

  template <typename Other>
  const SumOfCovarianceFunctions<Derived, Other>
  operator+(const CovarianceFunction<Other> &other) const;

  template <typename Other>
  const ProductOfCovarianceFunctions<Derived, Other>
  operator*(const CovarianceFunction<Other> &other) const;

  Derived &derived() { return *static_cast<Derived *>(this); }

  const Derived &derived() const { return *static_cast<const Derived *>(this); }
};

/*
 * SUM
 */
template <class LHS, class RHS>
class SumOfCovarianceFunctions
    : public CovarianceFunction<SumOfCovarianceFunctions<LHS, RHS>> {
public:
  SumOfCovarianceFunctions() : lhs_(), rhs_() {}

  SumOfCovarianceFunctions(const LHS &lhs, const RHS &rhs)
      : lhs_(lhs), rhs_(rhs) {}

  std::string name() const {
    return "(" + lhs_.get_name() + "+" + rhs_.get_name() + ")";
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
   * this will return the sum of the two.
   *
   * Note that we don't use has_valid_caller which is going to look for any
   * of a large number of complicated composite types, we only care about
   * directly equivalent calls (such as symmetric definitions, or the use
   * of the Measurement<> wrapper, see traits.hpp). This is to avoid
   * polution of the instantiated methods.  If, for example, we used
   * has_valid_caller here summing two functions would result in the
   * instantiation of methods such as:
   *
   *   double _call_impl(const LinearCombination<X> &x,
   *                     const variant<X, Y> &y) const;
   *
   * which doesn't neccesarily breaking anything, but those sorts of
   * equivalencies are meant to be managed by the CovarianceFunction::call
   * method above. Instead we only want summing (and product etc ...)
   * to instantiate the minimal number of _call_impl methods and we
   * then consolidate any of the composite type management to the
   * highest level calls.
   */
  template <typename X, typename Y,
            typename std::enable_if<(has_equivalent_caller<LHS, X, Y>::value &&
                                     has_equivalent_caller<RHS, X, Y>::value),
                                    int>::type = 0>
  double _call_impl(const X &x, const Y &y) const {
    return this->lhs_(x, y) + this->rhs_(x, y);
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
   * If only RHS has a valid call method we ignore L.
   */
  template <typename X, typename Y,
            typename std::enable_if<(!has_equivalent_caller<LHS, X, Y>::value &&
                                     has_equivalent_caller<RHS, X, Y>::value),
                                    int>::type = 0>
  double _call_impl(const X &x, const Y &y) const {
    return this->rhs_(x, y);
  }

  template <typename X,
            typename std::enable_if<has_valid_ssr_impl<LHS, X>::value &&
                                        has_valid_ssr_impl<RHS, X>::value,
                                    int>::type = 0>
  auto _ssr_impl(const std::vector<X> &xs) const {
    return concatenate(this->lhs_.state_space_representation(xs),
                       this->rhs_.state_space_representation(xs));
  }

  template <typename X,
            typename std::enable_if<has_valid_ssr_impl<LHS, X>::value &&
                                        !has_valid_ssr_impl<RHS, X>::value,
                                    int>::type = 0>
  auto _ssr_impl(const std::vector<X> &xs) const {
    return this->lhs_.state_space_representation(xs);
  }

  template <typename X,
            typename std::enable_if<!has_valid_ssr_impl<LHS, X>::value &&
                                        has_valid_ssr_impl<RHS, X>::value,
                                    int>::type = 0>
  auto _ssr_impl(const std::vector<X> &xs) const {
    return this->rhs_.state_space_representation(xs);
  }

  /*
   * Batch covariance implementations
   *
   * These use has_valid_batch_or_measurement_batch to check whether batch
   * calls can be made, either directly or through Measurement unwrapping.
   * We then call through DefaultCaller::call_vector to leverage the
   * caller hierarchy's transformation capabilities (especially
   * MeasurementForwarder).
   */

  // Case 1: Both LHS and RHS have batch support (direct or via Measurement)
  template <typename X, typename Y,
            typename std::enable_if<
                (has_valid_batch_or_measurement_batch<LHS, X, Y>::value &&
                 has_valid_batch_or_measurement_batch<RHS, X, Y>::value),
                int>::type = 0>
  Eigen::MatrixXd _call_impl_vector(const std::vector<X> &xs,
                                    const std::vector<Y> &ys,
                                    ThreadPool *pool = nullptr) const {
    Eigen::MatrixXd result = DefaultCaller::call_vector(lhs_, xs, ys, pool);
    result += DefaultCaller::call_vector(rhs_, xs, ys, pool);
    return result;
  }

  // Case 2: Only LHS has batch support (direct or via Measurement)
  template <typename X, typename Y,
            typename std::enable_if<
                (has_valid_batch_or_measurement_batch<LHS, X, Y>::value &&
                 !has_valid_batch_or_measurement_batch<RHS, X, Y>::value &&
                 has_equivalent_caller<RHS, X, Y>::value),
                int>::type = 0>
  Eigen::MatrixXd _call_impl_vector(const std::vector<X> &xs,
                                    const std::vector<Y> &ys,
                                    ThreadPool *pool = nullptr) const {
    Eigen::MatrixXd result = DefaultCaller::call_vector(lhs_, xs, ys, pool);
    result += compute_covariance_matrix(rhs_, xs, ys, pool);
    return result;
  }

  // Case 3: Only RHS has batch support (direct or via Measurement)
  template <typename X, typename Y,
            typename std::enable_if<
                (!has_valid_batch_or_measurement_batch<LHS, X, Y>::value &&
                 has_equivalent_caller<LHS, X, Y>::value &&
                 has_valid_batch_or_measurement_batch<RHS, X, Y>::value),
                int>::type = 0>
  Eigen::MatrixXd _call_impl_vector(const std::vector<X> &xs,
                                    const std::vector<Y> &ys,
                                    ThreadPool *pool = nullptr) const {
    Eigen::MatrixXd result = DefaultCaller::call_vector(rhs_, xs, ys, pool);
    result += compute_covariance_matrix(lhs_, xs, ys, pool);
    return result;
  }

  /*
   * Diagonal batch implementations
   */

  // Both have diagonal batch
  template <typename X,
            typename std::enable_if<
                (has_valid_call_impl_vector_diagonal<LHS, X>::value &&
                 has_valid_call_impl_vector_diagonal<RHS, X>::value),
                int>::type = 0>
  Eigen::VectorXd _call_impl_vector_diagonal(const std::vector<X> &xs,
                                             ThreadPool *pool = nullptr) const {
    return this->lhs_._call_impl_vector_diagonal(xs, pool) +
           this->rhs_._call_impl_vector_diagonal(xs, pool);
  }

  // Only LHS has diagonal batch
  template <typename X,
            typename std::enable_if<
                (has_valid_call_impl_vector_diagonal<LHS, X>::value &&
                 !has_valid_call_impl_vector_diagonal<RHS, X>::value &&
                 has_equivalent_caller<RHS, X, X>::value),
                int>::type = 0>
  Eigen::VectorXd _call_impl_vector_diagonal(const std::vector<X> &xs,
                                             ThreadPool *pool = nullptr) const {
    Eigen::VectorXd result = this->lhs_._call_impl_vector_diagonal(xs, pool);
    const Eigen::Index n = cast::to_index(xs.size());
    for (Eigen::Index i = 0; i < n; ++i) {
      const auto si = cast::to_size(i);
      result[i] += this->rhs_(xs[si], xs[si]);
    }
    return result;
  }

  // Only RHS has diagonal batch
  template <typename X,
            typename std::enable_if<
                (!has_valid_call_impl_vector_diagonal<LHS, X>::value &&
                 has_equivalent_caller<LHS, X, X>::value &&
                 has_valid_call_impl_vector_diagonal<RHS, X>::value),
                int>::type = 0>
  Eigen::VectorXd _call_impl_vector_diagonal(const std::vector<X> &xs,
                                             ThreadPool *pool = nullptr) const {
    Eigen::VectorXd result = this->rhs_._call_impl_vector_diagonal(xs, pool);
    const Eigen::Index n = cast::to_index(xs.size());
    for (Eigen::Index i = 0; i < n; ++i) {
      const auto si = cast::to_size(i);
      result[i] += this->lhs_(xs[si], xs[si]);
    }
    return result;
  }

  // Single-arg symmetric batch: at least one child has single-arg support.
  // DefaultCaller::call_vector(child, xs, pool) dispatches to the best
  // available path for each child independently (single-arg if available,
  // two-arg fallback otherwise).
  //
  // Note: each child's result is mirrored by BatchCaller, so the sum is
  // already a full symmetric matrix.  BatchCaller then mirrors *this*
  // result again — a redundant O(n^2/2) write that we accept for
  // simplicity (avoiding it would require a parallel non-mirroring
  // dispatch path through the entire caller chain).
  template <
      typename X,
      typename std::enable_if<
          (has_valid_batch_or_measurement_batch_single_arg<LHS, X>::value ||
           has_valid_batch_or_measurement_batch_single_arg<RHS, X>::value),
          int>::type = 0>
  Eigen::MatrixXd _call_impl_vector(const std::vector<X> &xs,
                                    ThreadPool *pool = nullptr) const {
    Eigen::MatrixXd result;
    result.triangularView<Eigen::Lower>() =
        DefaultCaller::call_vector(lhs_, xs, pool);
    result.triangularView<Eigen::Lower>() +=
        DefaultCaller::call_vector(rhs_, xs, pool);
    return result;
  }

protected:
  LHS lhs_;
  RHS rhs_;
  friend class CallTrace<SumOfCovarianceFunctions<LHS, RHS>>;
};

/*
 * PRODUCT
 */
template <class LHS, class RHS>
class ProductOfCovarianceFunctions
    : public CovarianceFunction<ProductOfCovarianceFunctions<LHS, RHS>> {
public:
  ProductOfCovarianceFunctions() : lhs_(), rhs_() {}
  ProductOfCovarianceFunctions(const LHS &lhs, const RHS &rhs)
      : lhs_(lhs), rhs_(rhs) {
    ProductOfCovarianceFunctions();
  }

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
   * If only RHS has a valid call method we ignore L.
   */
  template <typename X, typename Y,
            typename std::enable_if<(!has_equivalent_caller<LHS, X, Y>::value &&
                                     has_equivalent_caller<RHS, X, Y>::value),
                                    int>::type = 0>
  double _call_impl(const X &x, const Y &y) const {
    return this->rhs_(x, y);
  }

  template <typename X,
            typename std::enable_if<has_valid_ssr_impl<LHS, X>::value &&
                                        has_valid_ssr_impl<RHS, X>::value,
                                    int>::type = 0>
  auto _ssr_impl(const std::vector<X> &xs) const {
    return concatenate(this->lhs_.state_space_representation(xs),
                       this->rhs_.state_space_representation(xs));
  }

  template <typename X,
            typename std::enable_if<has_valid_ssr_impl<LHS, X>::value &&
                                        !has_valid_ssr_impl<RHS, X>::value,
                                    int>::type = 0>
  auto _ssr_impl(const std::vector<X> &xs) const {
    return this->lhs_.state_space_representation(xs);
  }

  template <typename X,
            typename std::enable_if<!has_valid_ssr_impl<LHS, X>::value &&
                                        has_valid_ssr_impl<RHS, X>::value,
                                    int>::type = 0>
  auto _ssr_impl(const std::vector<X> &xs) const {
    return this->rhs_.state_space_representation(xs);
  }

  /*
   * Batch covariance implementations
   *
   * These use has_valid_batch_or_measurement_batch to check whether batch
   * calls can be made, either directly or through Measurement unwrapping.
   * We then call through DefaultCaller::call_vector to leverage the
   * caller hierarchy's transformation capabilities (especially
   * MeasurementForwarder).
   */

  // Case 1: Both LHS and RHS have batch support (direct or via Measurement)
  template <typename X, typename Y,
            typename std::enable_if<
                (has_valid_batch_or_measurement_batch<LHS, X, Y>::value &&
                 has_valid_batch_or_measurement_batch<RHS, X, Y>::value),
                int>::type = 0>
  Eigen::MatrixXd _call_impl_vector(const std::vector<X> &xs,
                                    const std::vector<Y> &ys,
                                    ThreadPool *pool = nullptr) const {
    // Element-wise multiplication using Eigen array operations
    Eigen::MatrixXd result = DefaultCaller::call_vector(lhs_, xs, ys, pool);
    result.array() *= DefaultCaller::call_vector(rhs_, xs, ys, pool).array();
    return result;
  }

  // Case 2: Only LHS has batch support (direct or via Measurement)
  template <typename X, typename Y,
            typename std::enable_if<
                (has_valid_batch_or_measurement_batch<LHS, X, Y>::value &&
                 !has_valid_batch_or_measurement_batch<RHS, X, Y>::value &&
                 has_equivalent_caller<RHS, X, Y>::value),
                int>::type = 0>
  Eigen::MatrixXd _call_impl_vector(const std::vector<X> &xs,
                                    const std::vector<Y> &ys,
                                    ThreadPool *pool = nullptr) const {
    Eigen::MatrixXd result = DefaultCaller::call_vector(lhs_, xs, ys, pool);
    result.array() *= compute_covariance_matrix(rhs_, xs, ys, pool).array();
    return result;
  }

  // Case 3: Only RHS has batch support (direct or via Measurement)
  template <typename X, typename Y,
            typename std::enable_if<
                (!has_valid_batch_or_measurement_batch<LHS, X, Y>::value &&
                 has_equivalent_caller<LHS, X, Y>::value &&
                 has_valid_batch_or_measurement_batch<RHS, X, Y>::value),
                int>::type = 0>
  Eigen::MatrixXd _call_impl_vector(const std::vector<X> &xs,
                                    const std::vector<Y> &ys,
                                    ThreadPool *pool = nullptr) const {
    Eigen::MatrixXd result = DefaultCaller::call_vector(rhs_, xs, ys, pool);
    result.array() *= compute_covariance_matrix(lhs_, xs, ys, pool).array();
    return result;
  }

  /*
   * Diagonal batch implementations
   */

  // Both have diagonal batch
  template <typename X,
            typename std::enable_if<
                (has_valid_call_impl_vector_diagonal<LHS, X>::value &&
                 has_valid_call_impl_vector_diagonal<RHS, X>::value),
                int>::type = 0>
  Eigen::VectorXd _call_impl_vector_diagonal(const std::vector<X> &xs,
                                             ThreadPool *pool = nullptr) const {
    return this->lhs_._call_impl_vector_diagonal(xs, pool).array() *
           this->rhs_._call_impl_vector_diagonal(xs, pool).array();
  }

  // Only LHS has diagonal batch
  template <typename X,
            typename std::enable_if<
                (has_valid_call_impl_vector_diagonal<LHS, X>::value &&
                 !has_valid_call_impl_vector_diagonal<RHS, X>::value &&
                 has_equivalent_caller<RHS, X, X>::value),
                int>::type = 0>
  Eigen::VectorXd _call_impl_vector_diagonal(const std::vector<X> &xs,
                                             ThreadPool *pool = nullptr) const {
    Eigen::VectorXd result = this->lhs_._call_impl_vector_diagonal(xs, pool);
    const Eigen::Index n = cast::to_index(xs.size());
    for (Eigen::Index i = 0; i < n; ++i) {
      const auto si = cast::to_size(i);
      if (result[i] != 0.) {
        result[i] *= this->rhs_(xs[si], xs[si]);
      }
    }
    return result;
  }

  // Only RHS has diagonal batch
  template <typename X,
            typename std::enable_if<
                (!has_valid_call_impl_vector_diagonal<LHS, X>::value &&
                 has_equivalent_caller<LHS, X, X>::value &&
                 has_valid_call_impl_vector_diagonal<RHS, X>::value),
                int>::type = 0>
  Eigen::VectorXd _call_impl_vector_diagonal(const std::vector<X> &xs,
                                             ThreadPool *pool = nullptr) const {
    Eigen::VectorXd result = this->rhs_._call_impl_vector_diagonal(xs, pool);
    const Eigen::Index n = cast::to_index(xs.size());
    for (Eigen::Index i = 0; i < n; ++i) {
      const auto si = cast::to_size(i);
      if (result[i] != 0.) {
        result[i] *= this->lhs_(xs[si], xs[si]);
      }
    }
    return result;
  }

  // Single-arg symmetric batch: at least one child has single-arg support.
  // DefaultCaller::call_vector(child, xs, pool) dispatches to the best
  // available path for each child independently (single-arg if available,
  // two-arg fallback otherwise).
  //
  // Note: each child's result is mirrored by BatchCaller, so the product
  // is already a full symmetric matrix.  BatchCaller then mirrors *this*
  // result again — a redundant O(n^2/2) write that we accept for
  // simplicity (avoiding it would require a parallel non-mirroring
  // dispatch path through the entire caller chain).
  template <
      typename X,
      typename std::enable_if<
          (has_valid_batch_or_measurement_batch_single_arg<LHS, X>::value ||
           has_valid_batch_or_measurement_batch_single_arg<RHS, X>::value),
          int>::type = 0>
  Eigen::MatrixXd _call_impl_vector(const std::vector<X> &xs,
                                    ThreadPool *pool = nullptr) const {
    Eigen::MatrixXd ret;
    ret.triangularView<Eigen::Lower>() =
        DefaultCaller::call_vector(lhs_, xs, pool);
    ret.triangularView<Eigen::Lower>() =
        (ret.array() * DefaultCaller::call_vector(rhs_, xs, pool).array())
            .matrix();
    return ret;
    // return DefaultCaller::call_vector(lhs_, xs, pool).array() *
    // DefaultCaller::call_vector(rhs_, xs, pool).array();
  }

protected:
  LHS lhs_;
  RHS rhs_;
  friend class CallTrace<ProductOfCovarianceFunctions<LHS, RHS>>;
};

template <typename Derived>
template <typename Other>
inline const SumOfCovarianceFunctions<Derived, Other>
CovarianceFunction<Derived>::operator+(
    const CovarianceFunction<Other> &other) const {
  return SumOfCovarianceFunctions<Derived, Other>(derived(), other.derived());
}

template <typename Derived>
template <typename Other>
inline const ProductOfCovarianceFunctions<Derived, Other>
CovarianceFunction<Derived>::operator*(
    const CovarianceFunction<Other> &other) const {
  return ProductOfCovarianceFunctions<Derived, Other>(derived(),
                                                      other.derived());
}

} // namespace albatross

#endif
