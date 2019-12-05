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

template <typename T, typename... Args>
class is_invocable_const_ref
    : public is_invocable<const T, typename const_ref<Args>::type...> {};

/*
 * A GrouperFunction is a function which takes a single const ref argument
 * and returns a non-void type which will end up being the key used in group by.
 */
template <typename GrouperFunction, typename ValueType>
struct grouper_result
    : public invoke_result<GrouperFunction,
                           typename const_ref<ValueType>::type> {};

template <typename GroupKey> class group_key_is_valid {

  template <typename C, std::enable_if_t<is_valid_map_key<C>::value, int> = 0>
  static std::true_type test(C *);
  template <typename> static std::false_type test(...);

public:
  static constexpr bool value = decltype(test<GroupKey>(0))::value;
};

template <typename GrouperFunction, typename ValueType>
struct is_valid_grouper {

private:
  using GroupKey = typename grouper_result<GrouperFunction, ValueType>::type;

public:
  static constexpr bool value =
      is_invocable_const_ref<GrouperFunction, ValueType>::value &&
      group_key_is_valid<GroupKey>::value;
};

/*
 * An ApplyFunction is a function which takes a key value pair and
 * returns a new type, they key type needs to remain unchanged but
 * the value can be modified.
 */
template <typename ApplyFunction, typename KeyType, typename ArgType>
struct key_value_apply_result
    : public invoke_result<ApplyFunction, typename const_ref<KeyType>::type,
                           ArgType> {};

template <typename ApplyFunction, typename ArgType>
struct value_only_apply_result : public invoke_result<ApplyFunction, ArgType> {
};

template <typename ApplyFunction, typename KeyType, typename ArgType>
struct is_valid_key_value_apply_function
    : public is_invocable_const_ref<
          ApplyFunction, typename const_ref<KeyType>::type, ArgType> {};

template <typename ApplyFunction, typename ArgType>
struct is_valid_value_only_apply_function
    : public is_invocable_const_ref<ApplyFunction, ArgType> {};

template <typename ApplyFunction, typename KeyType, typename ArgType>
struct is_valid_index_apply_function
    : public is_invocable_const_ref<ApplyFunction,
                                    typename const_ref<KeyType>::type,
                                    typename const_ref<GroupIndices>::type> {};

template <typename Expected, typename FilterFunction, typename... Args>
struct invoke_result_is_same {

  template <typename C,
            typename std::enable_if_t<
                std::is_same<typename invoke_result<C, Args...>::type,
                             Expected>::value,
                int> = 0>
  static std::true_type test(C *);
  template <typename> static std::false_type test(...);

public:
  static constexpr bool value = decltype(test<FilterFunction>(0))::value;
};

/*
 * Check for valid filter functions.
 */
template <typename FilterFunction, typename... Args>
class is_valid_filter_function {
  template <typename C,
            typename std::enable_if_t<
                invoke_result_is_same<bool, C, Args...>::value, int> = 0>
  static std::true_type test(C *);
  template <typename> static std::false_type test(...);

public:
  static constexpr bool value = decltype(test<FilterFunction>(0))::value;
};

template <typename FilterFunction, typename KeyType, typename ArgType>
struct is_valid_key_value_filter_function
    : public is_valid_filter_function<FilterFunction, KeyType, ArgType> {};

template <typename FilterFunction, typename ArgType>
struct is_valid_value_only_filter_function
    : public is_valid_filter_function<FilterFunction, ArgType> {};

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
  static_assert(is_valid_grouper<GrouperFunction, IterableType>::value,
                "Invalid Grouper Function");
  using KeyType = typename grouper_result<GrouperFunction, IterableType>::type;
  using GrouperType = GrouperFunction;
};

template <typename FeatureType, typename GrouperFunction>
struct group_by_traits<GroupBy<std::vector<FeatureType>, GrouperFunction>> {
  using ValueType = std::vector<FeatureType>;
  using IterableType = FeatureType;
  static_assert(is_valid_grouper<GrouperFunction, IterableType>::value,
                "Invalid Grouper Function");
  using KeyType = typename grouper_result<GrouperFunction, IterableType>::type;
  using GrouperType = GrouperFunction;
};

} // namespace details

} // namespace albatross

#endif /* ALBATROSS_INDEXING_TRAITS_HPP_ */
