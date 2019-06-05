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

#ifndef ALBATROSS_COVARIANCE_FUNCTIONS_TRAITS_H
#define ALBATROSS_COVARIANCE_FUNCTIONS_TRAITS_H

namespace albatross {

MAKE_HAS_ANY_TRAIT(_call_impl);

// A helper rename to avoid duplicate underscores.
template <typename U> class has_any_call_impl : public has_any__call_impl<U> {};

HAS_METHOD_WITH_RETURN_TYPE(_call_impl);

template <typename U, typename... Args>
class has_valid_call_impl : public has__call_impl_with_return_type<
                                U, double, typename const_ref<Args>::type...> {
};

HAS_METHOD(_call_impl);

template <typename U, typename... Args>
class has_possible_call_impl : public has__call_impl<U, Args &...> {};

HAS_METHOD_WITH_RETURN_TYPE(call);

/*
 * This determines whether or not a class has a method defined for,
 *   `operator() (const X &x, const Y &y, const Z &z, ...)`
 * The result of the inspection gets stored in the member `value`.
 */
template <typename T, typename... Args> class has_call_operator {

  template <typename C, typename = decltype(std::declval<C>()(
                            std::declval<typename const_ref<Args>::type>()...))>
  static std::true_type test(C *);
  template <typename> static std::false_type test(...);

public:
  static constexpr bool value = decltype(test<T>(0))::value;
};

template <typename T, typename... Args> class has_invalid_call_impl {

public:
  static constexpr bool value = (has_possible_call_impl<T, Args...>::value &&
                                 !has_valid_call_impl<T, Args...>::value);
};

HAS_METHOD(solve);

} // namespace albatross

#endif
