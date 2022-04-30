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

#ifndef ALBATROSS_COVARIANCE_FUNCTIONS_MEAN_FUNCTION_H
#define ALBATROSS_COVARIANCE_FUNCTIONS_MEAN_FUNCTION_H

namespace albatross {

template <typename Derived> class MeanFunction : public ParameterHandlingMixin {

private:
  // Declaring these private makes it impossible to accidentally do things like:
  //     class A : public MeanFunction<B> {}
  // or
  //     using A = MeanFunction<B>;
  //
  // which if unchecked can lead to some very strange behavior.
  MeanFunction() : ParameterHandlingMixin(){};
  friend Derived;

public:
  template <typename X> double call(const X &x) const {
    return DefaultCaller::call(derived(), x);
  }

  static_assert(!is_complete<Derived>::value,
                "\n\nPassing a complete type in as template parameter "
                "implies you aren't using CRTP.  Implementations "
                "of a MeanFunction should look something like:\n"
                "\n\tclass Foo : public MeanFunction<Foo> {"
                "\n\t\tdouble _call_impl(const X &x, const Y &y) const;"
                "\n\t\t..."
                "\n\t}\n");

  template <typename DummyType = Derived,
            typename std::enable_if<!has_name<DummyType>::value, int>::type = 0>
  std::string get_name() const {
    static_assert(std::is_same<DummyType, Derived>::value,
                  "never do mean_function.get_name<T>()");
    return typeid(Derived).name();
  }

  template <typename DummyType = Derived,
            typename std::enable_if<has_name<DummyType>::value, int>::type = 0>
  std::string get_name() const {
    static_assert(std::is_same<DummyType, Derived>::value,
                  "never do mean_function.get_name<T>()");
    return derived().name();
  }

  std::string pretty_string() const {
    std::ostringstream ss;
    ss << get_name() << std::endl;
    ss << ParameterHandlingMixin::pretty_string();
    return ss.str();
  }

  template <typename X, typename std::enable_if<
                            has_valid_caller<Derived, X>::value, int>::type = 0>
  auto operator()(const X &x) const {
    return call(x);
  }

  // Computes the mean vector
  template <typename X, typename std::enable_if<
                            has_valid_caller<Derived, X>::value, int>::type = 0>
  Eigen::VectorXd operator()(const std::vector<X> &xs) const {

    if (std::is_same<Derived, ZeroMean>::value) {
      Eigen::Index n = static_cast<Eigen::Index>(xs.size());
      return Eigen::VectorXd::Zero(n);
    }

    auto caller = [&](const auto &x) { return this->call(x); };
    return compute_mean_vector(caller, xs);
  }

  // Adds the mean to a vector
  template <typename FeatureType>
  void add_to(const std::vector<FeatureType> &features,
              Eigen::VectorXd *target) const {
    if (std::is_same<Derived, ZeroMean>::value) {
      return;
    }
    const Eigen::VectorXd mean = this->operator()(features);
    assert(mean.size() == target->size());
    *target += mean;
  }

  // Removes the mean from a vector.
  template <typename FeatureType>
  void remove_from(const std::vector<FeatureType> &features,
                   Eigen::VectorXd *target) const {
    if (std::is_same<Derived, ZeroMean>::value) {
      return;
    }
    const Eigen::VectorXd mean = this->operator()(features);
    assert(mean.size() == target->size());
    *target -= mean;
  }

  /*
   * Stubs to catch the case where a mean function was called
   * with arguments that aren't supported.
   */
  template <typename X, typename std::enable_if<
                            (!has_valid_caller<Derived, X>::value &&
                             !has_possible_call_impl<Derived, X>::value),
                            int>::type = 0>
  void operator()(const X &x) const
      ALBATROSS_FAIL(X, "No public method with signature 'double "
                        "Derived::_call_impl(const X&) const'");

  template <typename X,
            typename std::enable_if<(!has_valid_caller<Derived, X>::value &&
                                     has_invalid_call_impl<Derived, X>::value),
                                    int>::type = 0>
  void operator()(const X &x) const
      ALBATROSS_FAIL(X, "Incorrectly defined method 'double "
                        "Derived::_call_impl(const X&) const'");

  template <typename Other>
  const SumOfMeanFunctions<Derived, Other>
  operator+(const MeanFunction<Other> &other) const;

  template <typename Other>
  const ProductOfMeanFunctions<Derived, Other>
  operator*(const MeanFunction<Other> &other) const;

  Derived &derived() { return *static_cast<Derived *>(this); }

  const Derived &derived() const { return *static_cast<const Derived *>(this); }
};

/*
 * SUM
 */
template <class LHS, class RHS>
class SumOfMeanFunctions : public MeanFunction<SumOfMeanFunctions<LHS, RHS>> {
public:
  SumOfMeanFunctions() : lhs_(), rhs_(){};

  SumOfMeanFunctions(const LHS &lhs, const RHS &rhs) : lhs_(lhs), rhs_(rhs){};

  std::string name() const {
    return "(" + lhs_.get_name() + "+" + rhs_.get_name() + ")";
  }

  ParameterStore get_params() const override {
    return map_join(lhs_.get_params(), rhs_.get_params());
  }

  void set_param(const ParameterKey &name, const Parameter &param) override {
    assert(set_param_if_exists_in_any(name, param, &lhs_, &rhs_));
  }

  /*
   * If both LHS and RHS have a valid call method for the types X and Y
   * this will return the sum of the two.
   */
  template <typename X,
            typename std::enable_if<(has_valid_caller<LHS, X>::value &&
                                     has_valid_caller<RHS, X>::value),
                                    int>::type = 0>
  double _call_impl(const X &x) const {
    return this->lhs_(x) + this->rhs_(x);
  }

  /*
   * If only LHS has a valid call method we ignore R.
   */
  template <typename X,
            typename std::enable_if<(has_valid_caller<LHS, X>::value &&
                                     !has_valid_caller<RHS, X>::value),
                                    int>::type = 0>
  double _call_impl(const X &x) const {
    return this->lhs_(x);
  }

  /*
   * If only RHS has a valid call method we ignore L.
   */
  template <typename X,
            typename std::enable_if<(!has_valid_caller<LHS, X>::value &&
                                     has_valid_caller<RHS, X>::value),
                                    int>::type = 0>
  double _call_impl(const X &x) const {
    return this->rhs_(x);
  }

protected:
  LHS lhs_;
  RHS rhs_;
};

/*
 * PRODUCT
 */
template <class LHS, class RHS>
class ProductOfMeanFunctions
    : public MeanFunction<ProductOfMeanFunctions<LHS, RHS>> {
public:
  ProductOfMeanFunctions() : lhs_(), rhs_(){};
  ProductOfMeanFunctions(const LHS &lhs, const RHS &rhs)
      : lhs_(lhs), rhs_(rhs) {
    ProductOfMeanFunctions();
  };

  std::string name() const {
    return "(" + lhs_.get_name() + "*" + rhs_.get_name() + ")";
  }

  ParameterStore get_params() const override {
    return map_join(lhs_.get_params(), rhs_.get_params());
  }

  void set_param(const ParameterKey &name, const Parameter &param) override {
    assert(set_param_if_exists_in_any(name, param, &lhs_, &rhs_));
  }

  /*
   * If both LHS and RHS have a valid call method for the types X and Y
   * this will return the product of the two.
   */
  template <typename X,
            typename std::enable_if<(has_valid_caller<LHS, X>::value &&
                                     has_valid_caller<RHS, X>::value),
                                    int>::type = 0>
  double _call_impl(const X &x) const {
    double output = this->lhs_(x);
    if (output != 0.) {
      output *= this->rhs_(x);
    }
    return output;
  }

  /*
   * If only LHS has a valid call method we ignore R.
   */
  template <typename X,
            typename std::enable_if<(has_valid_caller<LHS, X>::value &&
                                     !has_valid_caller<RHS, X>::value),
                                    int>::type = 0>
  double _call_impl(const X &x) const {
    return this->lhs_(x);
  }

  /*
   * If only RHS has a valid call method we ignore L.
   */
  template <typename X,
            typename std::enable_if<(!has_valid_caller<LHS, X>::value &&
                                     has_valid_caller<RHS, X>::value),
                                    int>::type = 0>
  double _call_impl(const X &x) const {
    return this->rhs_(x);
  }

protected:
  LHS lhs_;
  RHS rhs_;
  friend class CallTrace<ProductOfMeanFunctions<LHS, RHS>>;
};

struct ZeroMean : public MeanFunction<ZeroMean> {

  template <typename X> double _call_impl(const X &) const { return 0.; }
};

template <typename Derived>
template <typename Other>
inline const SumOfMeanFunctions<Derived, Other> MeanFunction<Derived>::
operator+(const MeanFunction<Other> &other) const {
  return SumOfMeanFunctions<Derived, Other>(derived(), other.derived());
};

template <typename Derived>
template <typename Other>
inline const ProductOfMeanFunctions<Derived, Other> MeanFunction<Derived>::
operator*(const MeanFunction<Other> &other) const {
  return ProductOfMeanFunctions<Derived, Other>(derived(), other.derived());
};

} // namespace albatross

#endif // ALBATROSS_COVARIANCE_FUNCTIONS_MEAN_FUNCTION_H
