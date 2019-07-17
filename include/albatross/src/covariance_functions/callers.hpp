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

#ifndef ALBATROSS_COVARIANCE_FUNCTIONS_CALLERS_HPP_
#define ALBATROSS_COVARIANCE_FUNCTIONS_CALLERS_HPP_

namespace albatross {

/*
 * Implementing a CovarianceFunction requires defining a method with
 * signature:
 *
 *     double _call_impl(const X &x, const Y &y) const
 *
 * There are often different sets of arguments for which the covariance
 * function should be equivalent.  For example, we know that covariance
 * funcitons are symmetric, so a call to cov(x, y) will be equivalent
 * to cov(y, x).
 *
 * This and some additional desired behavior, the ability to distinguish
 * between a measurement and measurement noise free feature for example,
 * led to an intermediary step which use Callers.  These callers
 * can be strung together to avoid repeated trait inspection.
 */

namespace internal {

/*
 * This Caller just directly call the underlying CovFunc.
 */
struct DirectCaller {

  template <typename CovFunc, typename X, typename Y,
            typename std::enable_if<has_valid_call_impl<CovFunc, X, Y>::value,
                                    int>::type = 0>
  static double call(const CovFunc &cov_func, const X &x, const Y &y) {
    return cov_func._call_impl(x, y);
  }
};

/*
 * This Caller turns any CovFunc defined for argument types X, Y into
 * one valid for Y, X as well.
 */
template <typename SubCaller> struct SymmetricCaller {

  /*
   * CovFunc has a direct call implementation for X and Y
   */
  template <
      typename CovFunc, typename X, typename Y,
      typename std::enable_if<
          has_valid_cov_caller<CovFunc, SubCaller, X, Y>::value, int>::type = 0>
  static double call(const CovFunc &cov_func, const X &x, const Y &y) {
    return SubCaller::call(cov_func, x, y);
  }

  /*
   * CovFunc has a call for Y and X but not X and Y
   */
  template <typename CovFunc, typename X, typename Y,
            typename std::enable_if<
                (has_valid_cov_caller<CovFunc, SubCaller, Y, X>::value &&
                 !has_valid_cov_caller<CovFunc, SubCaller, X, Y>::value),
                int>::type = 0>
  static double call(const CovFunc &cov_func, const X &x, const Y &y) {
    return SubCaller::call(cov_func, y, x);
  }
};

/*
 * This Caller maps any call with type Measurement<X> to one which
 * just uses the underlying type (X) UNLESS a call is actually defined
 * for Measurement<X> in which case that is used.  This makes it possible
 * to define a covariance function which behaves differently when presented
 * with training data (measurements) versus test data where you may
 * actually be interested in the underlying process.
 */
template <typename SubCaller> struct MeasurementForwarder {

  template <
      typename CovFunc, typename X, typename Y,
      typename std::enable_if<
          has_valid_cov_caller<CovFunc, SubCaller, X, Y>::value, int>::type = 0>
  static double call(const CovFunc &cov_func, const X &x, const Y &y) {
    return SubCaller::call(cov_func, x, y);
  }

  template <typename CovFunc, typename X, typename Y,
            typename std::enable_if<
                (has_valid_cov_caller<CovFunc, SubCaller, X, Y>::value &&
                 !has_valid_cov_caller<CovFunc, SubCaller, Measurement<X>,
                                       Measurement<Y>>::value),
                int>::type = 0>
  static double call(const CovFunc &cov_func, const Measurement<X> &x,
                     const Measurement<Y> &y) {
    return SubCaller::call(cov_func, x.value, y.value);
  }

  template <
      typename CovFunc, typename X, typename Y,
      typename std::enable_if<
          (has_valid_cov_caller<CovFunc, SubCaller, X, Y>::value &&
           !has_valid_cov_caller<CovFunc, SubCaller, Measurement<X>, Y>::value),
          int>::type = 0>
  static double call(const CovFunc &cov_func, const Measurement<X> &x,
                     const Y &y) {
    return SubCaller::call(cov_func, x.value, y);
  }

  template <
      typename CovFunc, typename X, typename Y,
      typename std::enable_if<
          (has_valid_cov_caller<CovFunc, SubCaller, X, Y>::value &&
           !has_valid_cov_caller<CovFunc, SubCaller, X, Measurement<Y>>::value),
          int>::type = 0>
  static double call(const CovFunc &cov_func, const X &x,
                     const Measurement<Y> &y) {
    return SubCaller::call(cov_func, x, y.value);
  }
};

/*
 * VariantForwarder
 *
 * The variant forwarder allows covariance functions to be called with
 * variant types, without explicitly handling them.  For example, a
 * covariance function which has methods:
 *
 *   double _call_impl(const X&, const X&) const;
 *
 *   double _call_impl(const X&, const Y&) const;
 *
 *   double _call_impl(const Y&, const Y&) const;
 *
 * would then also support the following pairs of arguments:
 *
 *   variant<X, Y>, X
 *   X, variant<X, Y>
 *   variant<X, Y>, Y
 *   Y, variant<X, Y>
 *
 */
template <typename SubCaller> struct VariantForwarder {

  // directly forward on the case where both types aren't variants
  template <typename CovFunc, typename X, typename Y,
            typename std::enable_if<
                !is_variant<X>::value && !is_variant<Y>::value &&
                    has_valid_cov_caller<CovFunc, SubCaller, X, Y>::value,
                int>::type = 0>
  static double call(const CovFunc &cov_func, const X &x, const Y &y) {
    return SubCaller::call(cov_func, x, y);
  }

  /*
   * This visitor helps deal with enabling and disabling the call operator
   * depending on whether pairs of types in variants are defined.
   */
  template <typename CovFunc, typename X> struct CallVisitor {

    CallVisitor(const CovFunc &cov_func, const X &x)
        : cov_func_(cov_func), x_(x){};

    template <
        typename Y,
        typename std::enable_if<
            has_valid_cov_caller<CovFunc, SubCaller, X, Y>::value, int>::type = 0>
    double operator()(const Y &y) const {
      return SubCaller::call(cov_func_, x_, y);
    };

    template <
        typename Y,
        typename std::enable_if<
            !has_valid_cov_caller<CovFunc, SubCaller, X, Y>::value, int>::type = 0>
    double operator()(const Y &y) const {
      return 0.;
    };

    const CovFunc &cov_func_;
    const X &x_;
  };

  // Deals with a type, A, and an arbitrary variant.
  template <typename CovFunc, typename A, typename... Ts,
            typename std::enable_if<
                !is_variant<A>::value &&
                    has_valid_variant_cov_caller<CovFunc, SubCaller, A,
                                                 variant<Ts...>>::value,
                int>::type = 0>
  static double call(const CovFunc &cov_func, const A &x,
                     const variant<Ts...> &y) {
    CallVisitor<CovFunc, A> visitor(cov_func, x);
    return apply_visitor(visitor, y);
  }

  // Deals with a type, A, and an arbitrary variant.
  template <typename CovFunc, typename A, typename... Ts,
            typename std::enable_if<
                !is_variant<A>::value &&
                    has_valid_variant_cov_caller<CovFunc, SubCaller, A,
                                                 variant<Ts...>>::value,
                int>::type = 0>
  static double call(const CovFunc &cov_func,
                     const variant<Ts...> &x,
                     const A &y) {
    CallVisitor<CovFunc, A> visitor(cov_func, y);
    return apply_visitor(visitor, x);
  }

  // Deals with two identical variants.
  template <typename CovFunc, typename... Ts,
            typename std::enable_if<
                has_valid_cov_caller<CovFunc, SubCaller, variant<Ts...>,
                                     variant<Ts...>>::value,
                int>::type = 0>
  static double call(const CovFunc &cov_func, const variant<Ts...> &x,
                     const variant<Ts...> &y) {
    return x.match(
        [&y, &cov_func](const auto &xx) { return call(cov_func, xx, y); });
  }
};

} // namespace internal

/*
 * This defines the order of operations of the covariance function Callers.
 */
using DefaultCaller = internal::VariantForwarder<internal::MeasurementForwarder<internal::SymmetricCaller<internal::DirectCaller>>>;

template <typename Caller, typename CovFunc, typename... Args>
class caller_has_valid_call
    : public has_call_with_return_type<Caller, double,
                                       typename const_ref<CovFunc>::type,
                                       typename const_ref<Args>::type...> {};

template <typename CovFunc, typename... Args>
class has_valid_caller : public caller_has_valid_call<DefaultCaller, CovFunc, Args...> {};


} // namespace albatross

#endif /* ALBATROSS_COVARIANCE_FUNCTIONS_CALLERS_HPP_ */
