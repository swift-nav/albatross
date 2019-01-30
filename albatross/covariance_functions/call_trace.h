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

#ifndef ALBATROSS_COVARIANCE_FUNCTIONS_CALL_TRACE_H
#define ALBATROSS_COVARIANCE_FUNCTIONS_CALL_TRACE_H

namespace albatross {

#define LEVEL_DELIMITER " |  "

struct CallAndValue {
  std::string call_name;
  std::string value;

  void add_operator(const char c) {
    assert(call_name.size() >= 3);
    assert(value.size() >= 3);
    call_name[2] = c;
    value[2] = c;
  }

  void add_delimiter() {
    call_name = LEVEL_DELIMITER + call_name;
    value = LEVEL_DELIMITER + value;
  };
};

/*
 * Takes two call traces and appends one to the other after
 * having applied a level offset delimiter.
 */
inline void append_child_call_trace(std::vector<CallAndValue> &parent,
                                    std::vector<CallAndValue> &child,
                                    const char &operator_character) {

  auto add_delimeter = [](CallAndValue &cv) { cv.add_delimiter(); };
  // Add an inset delimiter to each child.
  std::for_each(child.begin(), child.end(), add_delimeter);
  child.front().add_operator(operator_character);
  parent.insert(parent.end(), child.begin(), child.end());
}

inline void print_call_trace(const std::vector<CallAndValue> &call_trace) {
  std::size_t max_value_length = 0;

  for (const auto &cv : call_trace) {
    if (cv.value.size() > max_value_length) {
      max_value_length = cv.value.size();
    }
  }

  std::cout << std::setw(max_value_length + 2) << std::left << "VALUE"
            << std::setw(100) << "FUNCTION" << std::endl;
  std::cout << std::setw(max_value_length + 2) << std::left << "-----"
            << std::setw(100) << "--------" << std::endl;
  for (const auto &cv : call_trace) {
    std::cout << std::setw(max_value_length + 2) << std::left << cv.value
              << std::setw(100) << cv.call_name << std::endl;
  }
}

template <typename Derived> class CallTraceBase {
public:
  template <typename X, typename Y> void print(const X &x, const Y &y) {
    print_call_trace(derived().get_trace(x, y));
  }

protected:
  Derived &derived() { return *static_cast<Derived *>(this); }

  const Derived &derived() const { return *static_cast<const Derived *>(this); }
};

/*
 * These methods can be useful when debugging a covariance function.
 *
 * The static composition (ie, products and sums) of CovarianceFunction
 * types can lead to a series of operations that are difficult to follow.
 *
 * By doing something along the lines of:
 *
 *     SomeCovFunc cov_func;
 *     FeatureX x;
 *     FeatureY y;
 *     cov_func.call_trace().print(x, y);
 *
 * You'll see a trace of all the calls that go into the final computation
 * of cov_func(x, y).
 */

template <typename CovFunc>
class CallTrace : public CallTraceBase<CallTrace<CovFunc>> {
public:
  CallTrace(const CovFunc &cov_func) : cov_func_(cov_func){};

  template <typename X, typename Y,
            typename std::enable_if<
                has_defined_call_impl<CovFunc, X &, Y &>::value, int>::type = 0>
  std::vector<CallAndValue> get_trace(const X &x, const Y &y) const {
    return {{cov_func_.get_name(), std::to_string(cov_func_(x, y))}};
  }

  template <typename X, typename Y,
            typename std::enable_if<
                (has_defined_call_impl<CovFunc, Y &, X &>::value &&
                 !has_defined_call_impl<CovFunc, X &, Y &>::value),
                int>::type = 0>
  std::vector<CallAndValue> get_trace(const X &x, const Y &y) const {
    return {{cov_func_.get_name(), std::to_string(cov_func_(x, y))}};
  }

  template <typename X, typename Y,
            typename std::enable_if<
                (!has_defined_call_impl<CovFunc, Y &, X &>::value &&
                 !has_defined_call_impl<CovFunc, X &, Y &>::value),
                int>::type = 0>
  std::vector<CallAndValue> get_trace(const X &, const Y &) const {
    return {{cov_func_.get_name(), "UNDEFINED"}};
  }

  CovFunc cov_func_;
};

template <typename LHS, typename RHS>
class CallTrace<SumOfCovarianceFunctions<LHS, RHS>>
    : public CallTraceBase<CallTrace<SumOfCovarianceFunctions<LHS, RHS>>> {
public:
  CallTrace(const SumOfCovarianceFunctions<LHS, RHS> &cov_func)
      : cov_func_(cov_func){};

  template <typename X, typename Y,
            typename std::enable_if<
                has_defined_call_impl<SumOfCovarianceFunctions<LHS, RHS>, X &,
                                      Y &>::value,
                int>::type = 0>
  std::string eval(const X &x, const Y &y) const {
    std::ostringstream oss;
    oss << cov_func_(x, y);
    return oss.str();
  }

  template <typename X, typename Y,
            typename std::enable_if<
                !has_defined_call_impl<SumOfCovarianceFunctions<LHS, RHS>, X &,
                                       Y &>::value,
                int>::type = 0>
  std::string eval(const X &, const Y &) const {
    return "UNDEFINED";
  }

  template <typename X, typename Y>
  std::vector<CallAndValue> get_trace(const X &x, const Y &y) const {
    std::vector<CallAndValue> calls;
    calls.push_back({SumOfCovarianceFunctions<LHS, RHS>().name_, eval(x, y)});

    std::vector<CallAndValue> lhs_calls =
        CallTrace<LHS>(this->cov_func_.lhs_).get_trace(x, y);
    append_child_call_trace(calls, lhs_calls, '+');

    std::vector<CallAndValue> rhs_calls =
        CallTrace<RHS>(this->cov_func_.rhs_).get_trace(x, y);
    append_child_call_trace(calls, rhs_calls, '+');

    return calls;
  }

  SumOfCovarianceFunctions<LHS, RHS> cov_func_;
};

template <typename LHS, typename RHS>
class CallTrace<ProductOfCovarianceFunctions<LHS, RHS>>
    : public CallTraceBase<CallTrace<ProductOfCovarianceFunctions<LHS, RHS>>> {
public:
  CallTrace(const ProductOfCovarianceFunctions<LHS, RHS> &cov_func)
      : cov_func_(cov_func){};

  template <typename X, typename Y,
            typename std::enable_if<
                has_defined_call_impl<ProductOfCovarianceFunctions<LHS, RHS>,
                                      X &, Y &>::value,
                int>::type = 0>
  std::string eval(const X &x, const Y &y) const {
    std::ostringstream oss;
    oss << cov_func_(x, y);
    return oss.str();
  }

  template <typename X, typename Y,
            typename std::enable_if<
                !has_defined_call_impl<ProductOfCovarianceFunctions<LHS, RHS>,
                                       X &, Y &>::value,
                int>::type = 0>
  std::string eval(const X &x, const Y &y) const {
    return "UNDEFINED";
  }

  template <typename X, typename Y>
  std::vector<CallAndValue> get_trace(const X &x, const Y &y) const {
    std::vector<CallAndValue> calls;
    calls.push_back(
        {ProductOfCovarianceFunctions<LHS, RHS>().name_, eval(x, y)});

    std::vector<CallAndValue> lhs_calls =
        CallTrace<LHS>(this->cov_func_.lhs_).get_trace(x, y);
    append_child_call_trace(calls, lhs_calls, '*');

    std::vector<CallAndValue> rhs_calls =
        CallTrace<RHS>(this->cov_func_.rhs_).get_trace(x, y);
    append_child_call_trace(calls, rhs_calls, '*');

    return calls;
  }

  ProductOfCovarianceFunctions<LHS, RHS> cov_func_;
};

// This defines the .call_trace method on CovarianceFunction
template <typename Derived>
inline CallTrace<Derived> CovarianceFunction<Derived>::call_trace() const {
  return CallTrace<Derived>(this->derived());
};
}
#endif
