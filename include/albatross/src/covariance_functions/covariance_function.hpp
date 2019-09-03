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

template <typename X, typename Y> class SumOfCovarianceFunctions;

template <typename X, typename Y> class ProductOfCovarianceFunctions;

template <typename Derived> class CallTrace;

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
  CovarianceFunction() : ParameterHandlingMixin(){};
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
   * We only allow the covariance function for argument types X and Y
   * to be defined if the corresponding term is defined for these
   * types.  This allows us to distinguish between covariance functions which
   * are 0. and ones that are just not defined (read: possibly bugs).
   */
  template <typename X, typename Y,
            typename std::enable_if<has_valid_caller<Derived, X, Y>::value,
                                    int>::type = 0>
  auto operator()(const X &x, const Y &y) const {
    return call(x, y);
  }

  /*
   * Covariance between each element and every other in a vector.
   */
  template <typename X,
            typename std::enable_if<has_valid_caller<Derived, X, X>::value,
                                    int>::type = 0>
  Eigen::MatrixXd operator()(const std::vector<X> &xs) const {
    int n = static_cast<int>(xs.size());
    Eigen::MatrixXd C(n, n);

    int i, j;
    std::size_t si, sj;
    for (i = 0; i < n; i++) {
      si = static_cast<std::size_t>(i);
      for (j = 0; j <= i; j++) {
        sj = static_cast<std::size_t>(j);
        C(i, j) = call(xs[si], xs[sj]);
        C(j, i) = C(i, j);
      }
    }
    return C;
  }

  /*
   * Cross covariance between two vectors of (possibly) different types.
   */
  template <typename X, typename Y,
            typename std::enable_if<has_valid_caller<Derived, X, Y>::value,
                                    int>::type = 0>
  Eigen::MatrixXd operator()(const std::vector<X> &xs,
                             const std::vector<Y> &ys) const {
    int m = static_cast<int>(xs.size());
    int n = static_cast<int>(ys.size());
    Eigen::MatrixXd C(m, n);

    int i, j;
    std::size_t si, sj;
    for (i = 0; i < m; i++) {
      si = static_cast<std::size_t>(i);
      for (j = 0; j < n; j++) {
        sj = static_cast<std::size_t>(j);
        C(i, j) = call(xs[si], ys[sj]);
      }
    }
    return C;
  }

  /*
   * Diagonal of the covariance matrix.
   */
  template <typename X,
            typename std::enable_if<has_valid_caller<Derived, X, X>::value,
                                    int>::type = 0>
  Eigen::VectorXd diagonal(const std::vector<X> &xs) const {
    int n = static_cast<int>(xs.size());
    Eigen::VectorXd diag(n);

    int i;
    std::size_t si;
    for (i = 0; i < n; i++) {
      si = static_cast<std::size_t>(i);
      diag[i] = call(xs[si], xs[si]);
    }
    return diag;
  }

  /*
   * Cross covariance between two vectors of (possibly) different types.
   */
  template <
      typename FeatureType,
      typename std::enable_if<
          has_valid_ssr_features<Derived, FeatureType>::value, int>::type = 0>
  auto get_ssr_features(const std::vector<FeatureType> &features) const {
    return derived()._ssr_features(features);
  }

  /*
   * Stubs to catch the case where a covariance function was called
   * with arguments that aren't supported.
   */
  template <typename X, typename std::enable_if<
                            (!has_valid_caller<Derived, X, X>::value &&
                             !has_possible_call_impl<Derived, X, X>::value),
                            int>::type = 0>
  void operator()(const X &x) const
      ALBATROSS_FAIL(X, "No public method with signature 'double "
                        "Derived::_call_impl(const X&, const X&) const'");

  template <typename X, typename std::enable_if<
                            (!has_valid_caller<Derived, X, X>::value &&
                             has_invalid_call_impl<Derived, X, X>::value),
                            int>::type = 0>
  void operator()(const X &x) const
      ALBATROSS_FAIL(X, "Incorrectly defined method 'double "
                        "Derived::_call_impl(const X&, const X&) const'");

  template <typename X,
            typename std::enable_if<!has_valid_caller<Derived, X, X>::value,
                                    int>::type = 0>
  void diagonal(const std::vector<X> &xs) const
      ALBATROSS_FAIL(X, "No public method with signature 'double "
                        "Derived::_call_impl(const X&, const X&) const'");

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
  SumOfCovarianceFunctions() : lhs_(), rhs_(){};

  SumOfCovarianceFunctions(const LHS &lhs, const RHS &rhs)
      : lhs_(lhs), rhs_(rhs){};

  std::string name() const {
    return "(" + lhs_.get_name() + "+" + rhs_.get_name() + ")";
  }

  ParameterStore get_params() const {
    return map_join(lhs_.get_params(), rhs_.get_params());
  }

  void unchecked_set_param(const ParameterKey &name, const Parameter &param) {
    if (map_contains(lhs_.get_params(), name)) {
      lhs_.set_param(name, param);
    } else {
      rhs_.set_param(name, param);
    }
  }

  /*
   * If both LHS and RHS have a valid call method for the types X and Y
   * this will return the sum of the two.
   */
  template <typename X, typename Y,
            typename std::enable_if<(has_valid_caller<LHS, X, Y>::value &&
                                     has_valid_caller<RHS, X, Y>::value),
                                    int>::type = 0>
  double _call_impl(const X &x, const Y &y) const {
    return this->lhs_(x, y) + this->rhs_(x, y);
  }

  /*
   * If only LHS has a valid call method we ignore R.
   */
  template <typename X, typename Y,
            typename std::enable_if<(has_valid_caller<LHS, X, Y>::value &&
                                     !has_valid_caller<RHS, X, Y>::value),
                                    int>::type = 0>
  double _call_impl(const X &x, const Y &y) const {
    return this->lhs_(x, y);
  }

  /*
   * If only RHS has a valid call method we ignore L.
   */
  template <typename X, typename Y,
            typename std::enable_if<(!has_valid_caller<LHS, X, Y>::value &&
                                     has_valid_caller<RHS, X, Y>::value),
                                    int>::type = 0>
  double _call_impl(const X &x, const Y &y) const {
    return this->rhs_(x, y);
  }

  /*
   * Manage the concatenation of ssr features.
   */
  template <typename X,
            typename std::enable_if<has_valid_ssr_features<LHS, X>::value &&
                                        !has_valid_ssr_features<RHS, X>::value,
                                    int>::type = 0>
  auto _ssr_features(const std::vector<X> &features) const {
    return this->lhs_._ssr_features(features);
  }

  template <typename X,
            typename std::enable_if<!has_valid_ssr_features<LHS, X>::value &&
                                        has_valid_ssr_features<RHS, X>::value,
                                    int>::type = 0>
  auto _ssr_features(const std::vector<X> &features) const {
    return this->rhs_._ssr_features(features);
  }

  template <typename X,
            typename std::enable_if<has_valid_ssr_features<LHS, X>::value &&
                                        has_valid_ssr_features<RHS, X>::value,
                                    int>::type = 0>
  auto _ssr_features(const std::vector<X> &features) const {
    const auto lhs = this->lhs_._ssr_features(features);
    const auto rhs = this->rhs_._ssr_features(features);
    return concatenate(lhs, rhs);
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
  ProductOfCovarianceFunctions() : lhs_(), rhs_(){};
  ProductOfCovarianceFunctions(const LHS &lhs, const RHS &rhs)
      : lhs_(lhs), rhs_(rhs) {
    ProductOfCovarianceFunctions();
  };

  std::string name() const {
    return "(" + lhs_.get_name() + "*" + rhs_.get_name() + ")";
  }

  ParameterStore get_params() const {
    return map_join(lhs_.get_params(), rhs_.get_params());
  }

  void unchecked_set_param(const ParameterKey &name, const Parameter &param) {
    if (map_contains(lhs_.get_params(), name)) {
      lhs_.set_param(name, param);
    } else {
      rhs_.set_param(name, param);
    }
  }

  /*
   * If both LHS and RHS have a valid call method for the types X and Y
   * this will return the product of the two.
   */
  template <typename X, typename Y,
            typename std::enable_if<(has_valid_caller<LHS, X, Y>::value &&
                                     has_valid_caller<RHS, X, Y>::value),
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
            typename std::enable_if<(has_valid_caller<LHS, X, Y>::value &&
                                     !has_valid_caller<RHS, X, Y>::value),
                                    int>::type = 0>
  double _call_impl(const X &x, const Y &y) const {
    return this->lhs_(x, y);
  }

  /*
   * If only RHS has a valid call method we ignore L.
   */
  template <typename X, typename Y,
            typename std::enable_if<(!has_valid_caller<LHS, X, Y>::value &&
                                     has_valid_caller<RHS, X, Y>::value),
                                    int>::type = 0>
  double _call_impl(const X &x, const Y &y) const {
    return this->rhs_(x, y);
  }

  /*
   * Manage the concatenation of ssr features.
   */
  template <typename X,
            typename std::enable_if<has_valid_ssr_features<LHS, X>::value &&
                                        !has_valid_ssr_features<RHS, X>::value,
                                    int>::type = 0>
  auto _ssr_features(const std::vector<X> &features) const {
    return this->lhs_._ssr_features(features);
  }

  template <typename X,
            typename std::enable_if<!has_valid_ssr_features<LHS, X>::value &&
                                        has_valid_ssr_features<RHS, X>::value,
                                    int>::type = 0>
  auto _ssr_features(const std::vector<X> &features) const {
    return this->rhs_._ssr_features(features);
  }

  template <typename X,
            typename std::enable_if<has_valid_ssr_features<LHS, X>::value &&
                                        has_valid_ssr_features<RHS, X>::value,
                                    int>::type = 0>
  auto _ssr_features(const std::vector<X> &features) const {
    const auto lhs = this->lhs_._ssr_features(features);
    const auto rhs = this->rhs_._ssr_features(features);
    return concatenate(lhs, rhs);
  }

protected:
  LHS lhs_;
  RHS rhs_;
  friend class CallTrace<ProductOfCovarianceFunctions<LHS, RHS>>;
};

template <typename Derived>
template <typename Other>
inline const SumOfCovarianceFunctions<Derived, Other>
CovarianceFunction<Derived>::
operator+(const CovarianceFunction<Other> &other) const {
  return SumOfCovarianceFunctions<Derived, Other>(derived(), other.derived());
};

template <typename Derived>
template <typename Other>
inline const ProductOfCovarianceFunctions<Derived, Other>
    CovarianceFunction<Derived>::
    operator*(const CovarianceFunction<Other> &other) const {
  return ProductOfCovarianceFunctions<Derived, Other>(derived(),
                                                      other.derived());
};

} // namespace albatross

#endif
