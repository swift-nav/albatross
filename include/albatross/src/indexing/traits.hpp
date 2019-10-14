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

#ifndef ALBATROSS_INDEXING_TRAITS_HPP_
#define ALBATROSS_INDEXING_TRAITS_HPP_

namespace albatross {

namespace details {

template <typename T> class has_less_than_operator {
  template <typename C,
            typename std::enable_if<
                std::is_same<bool, decltype(std::declval<C>() <
                                            std::declval<C>())>::value,
                int>::type = 0>
  static std::true_type test(C *);
  template <typename> static std::false_type test(...);

public:
  static constexpr bool value = decltype(test<T>(0))::value;
};

template <typename T> class is_valid_map_key {
  template <typename C,
            typename std::enable_if<std::is_default_constructible<C>::value &&
                                        std::is_copy_assignable<C>::value &&
                                        has_less_than_operator<C>::value,
                                    int>::type = 0>
  static std::true_type test(C *);
  template <typename> static std::false_type test(...);

public:
  static constexpr bool value = decltype(test<T>(0))::value;
};

template <> class is_valid_map_key<void> {
public:
  static constexpr bool value = false;
};

/*
 * This checks that a given type can be called with the following
 * argument types.
 *
 * Only works for non-overloaded methods, but does work with
 * lambda functions and function pointers.
 */
template <typename T, typename... Args> class callable_traits {
  template <typename C, typename ReturnType = decltype(
                            std::declval<C>()(std::declval<Args>()...))>
  static TypePair<std::true_type, ReturnType> test(C *);
  template <typename> static TypePair<std::false_type, void> test(...);

public:
  static constexpr bool is_defined = decltype(test<T>(0))::first_type::value;
  using return_type = typename decltype(test<T>(0))::second_type;
};

/*
 * Here we only care about the `is_defined` trait.
 */
template <typename T, typename... Args> struct can_be_called_with {
  static constexpr bool value = callable_traits<T, Args...>::is_defined;
};

template <typename T, typename... Args>
class can_be_called_with_const_ref
    : public can_be_called_with<const T, typename const_ref<Args>::type...> {};

/*
 * Stores the return type resulting from calling T with Args, if the
 * call isn't valid the resulting type will be void.
 */
template <typename T, typename... Args> struct return_type_when_called_with {
  using type = typename callable_traits<T, Args...>::return_type;
};

/*
 * A GrouperFunction is a function which takes a single const ref argument
 * and returns a non-void type which will end up being the key used in group by.
 */
template <typename GrouperFunction, typename ValueType>
struct grouper_return_type
    : public return_type_when_called_with<GrouperFunction,
                                          typename const_ref<ValueType>::type> {
};

template <typename GrouperFunction, typename ValueType>
struct group_key_is_valid {
  static constexpr bool value = is_valid_map_key<
      typename grouper_return_type<GrouperFunction, ValueType>::type>::value;
};

template <typename GrouperFunction, typename ValueType>
struct is_valid_grouper {

  static constexpr bool value =
      can_be_called_with_const_ref<GrouperFunction, ValueType>::value &&
      group_key_is_valid<GrouperFunction, ValueType>::value;
};

/*
 * An ApplyFunction is a function which takes a key value pair and
 * returns a new type, they key type needs to remain unchanged but
 * the value can be modified.
 */
template <typename ApplyFunction, typename KeyType, typename ArgType>
struct key_value_apply_return_type
    : public return_type_when_called_with<
          ApplyFunction, typename const_ref<KeyType>::type, ArgType> {};

template <typename ApplyFunction, typename ArgType>
struct value_only_apply_return_type
    : public return_type_when_called_with<ApplyFunction, ArgType> {};

template <typename ApplyFunction, typename KeyType, typename ArgType>
struct is_valid_key_value_apply_function
    : public can_be_called_with_const_ref<
          ApplyFunction, typename const_ref<KeyType>::type, ArgType> {};

template <typename ApplyFunction, typename ArgType>
struct is_valid_value_only_apply_function
    : public can_be_called_with_const_ref<ApplyFunction, ArgType> {};

template <typename ApplyFunction, typename KeyType, typename ArgType>
struct is_valid_index_apply_function
    : public can_be_called_with_const_ref<
          ApplyFunction, typename const_ref<KeyType>::type,
          typename const_ref<GroupIndices>::type> {};

template <typename FilterFunction, typename... Args>
struct returns_bool_when_called_with {
  static constexpr bool value = std::is_same<
      typename return_type_when_called_with<FilterFunction, Args...>::type,
      bool>::value;
};

template <typename FilterFunction, typename KeyType, typename ArgType>
struct is_valid_key_value_filter_function {
  static constexpr bool value =
      can_be_called_with_const_ref<FilterFunction, KeyType, ArgType>::value &&
      returns_bool_when_called_with<FilterFunction, KeyType, ArgType>::value;
};

template <typename FilterFunction, typename ArgType>
struct is_valid_value_only_filter_function {
  static constexpr bool value =
      can_be_called_with_const_ref<FilterFunction, ArgType>::value &&
      returns_bool_when_called_with<FilterFunction, ArgType>::value;
};

/*
 * The following traits are required in order to allow inspection of
 * the only partially defined Derived types inside of GroupByBase.
 *
 * To get GroupByBase to work with other types you need to add a new
 * trait struct for the type.
 *
 * GrouperFunction is a callable type which should take
 */
template <typename T> struct group_by_traits {};

template <typename FeatureType, typename GrouperFunction>
struct group_by_traits<
    GroupBy<RegressionDataset<FeatureType>, GrouperFunction>> {
  using ValueType = RegressionDataset<FeatureType>;
  using IterableType = FeatureType;
  using KeyType =
      typename grouper_return_type<GrouperFunction, IterableType>::type;
  using GrouperType = GrouperFunction;
};

template <typename FeatureType, typename GrouperFunction>
struct group_by_traits<GroupBy<std::vector<FeatureType>, GrouperFunction>> {
  using ValueType = std::vector<FeatureType>;
  using IterableType = FeatureType;
  using KeyType =
      typename grouper_return_type<GrouperFunction, IterableType>::type;
  using GrouperType = GrouperFunction;
};

} // namespace details

} // namespace albatross

#endif /* ALBATROSS_INDEXING_TRAITS_HPP_ */
