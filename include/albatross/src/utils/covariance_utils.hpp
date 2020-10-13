/*
 * Copyright (C) 2020 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef INCLUDE_ALBATROSS_SRC_UTILS_COVARIANCE_UTILS_HPP_
#define INCLUDE_ALBATROSS_SRC_UTILS_COVARIANCE_UTILS_HPP_

namespace albatross {

namespace details {

template <typename Child, typename Parent> struct cov_func_contains {
  const static bool value = false;
};

template <typename Child> struct cov_func_contains<Child, Child> {
  const static bool value = true;
};

template <typename Child, template <typename...> class Composite, typename LHS,
          typename RHS>
struct cov_func_contains<Child, Composite<LHS, RHS>> {
  const static bool value = cov_func_contains<Child, LHS>::value ||
                            cov_func_contains<Child, RHS>::value;
};

template <typename Parent> struct GetChildImpl {
  template <typename Child,
            std::enable_if_t<std::is_same<Parent, Child>::value, int> = 0>
  static Child get_child(const Parent &parent) {
    return parent;
  }
};

template <template <typename...> class Composite, typename LHS, typename RHS>
struct GetChildImpl<Composite<LHS, RHS>> {
  template <typename Child,
            std::enable_if_t<cov_func_contains<Child, LHS>::value, int> = 0>
  static Child get_child(const Composite<LHS, RHS> &parent) {
    return GetChildImpl<LHS>::template get_child<Child>(parent.get_lhs());
  }

  template <typename Child,
            std::enable_if_t<cov_func_contains<Child, RHS>::value &&
                                 !cov_func_contains<Child, LHS>::value,
                             int> = 0>
  static Child get_child(const Composite<LHS, RHS> &parent) {
    return GetChildImpl<RHS>::template get_child<Child>(parent.get_rhs());
  }
};

template <typename T> struct GetChildIdentity { typedef T type; };

} // namespace details

// usage looks like:
//
//   Parent parent;
//   Child child = get_child_covariance_function<Child>(parent);
//
// which fails if Child is not actual contained in Parent;
template <typename Child, typename Parent>
Child get_child_covariance_function(
    const Parent &parent,
    details::GetChildIdentity<Child> && = details::GetChildIdentity<Child>()) {
  return details::GetChildImpl<Parent>::template get_child<Child>(parent);
}

} // namespace albatross

#endif /* INCLUDE_ALBATROSS_SRC_UTILS_COVARIANCE_UTILS_HPP_ */
