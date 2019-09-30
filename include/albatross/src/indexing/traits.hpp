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

/*
 * This checks that a give type can be called with the following
 * argument types.
 *
 * Only works for non-overloaded methods.
 */
template <typename T, typename... Args> class can_be_called_with {
  template <typename C,
            typename = decltype(std::declval<C>()(std::declval<Args>()...))>
  static std::true_type test(C *);
  template <typename> static std::false_type test(...);

public:
  static constexpr bool value = decltype(test<T>(0))::value;
};

template <typename T, typename... Args>
class can_be_called_with_const_ref
    : public can_be_called_with<T, typename const_ref<Args>::type...> {};

/*
 * Stores the return type resulting from calling T with Args, if the
 * call isn't valid the resulting type will be void.
 */
template <typename T, typename... Args> class return_type_when_called_with {
  template <typename C, typename ReturnType = decltype(
                            std::declval<C>()(std::declval<Args>()...))>
  static ReturnType test(C *);
  template <typename> static void test(...);

public:
  typedef decltype(test<T>(0)) type;
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
  static constexpr bool value = !std::is_same<
      void, typename grouper_return_type<GrouperFunction, ValueType>::type>::value;
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
class apply_return_type
    : public return_type_when_called_with<
          ApplyFunction, typename const_ref<KeyType>::type, ArgType> {};

template <typename ApplyFunction, typename KeyType, typename ArgType>
class is_valid_apply_function
    : public can_be_called_with<ApplyFunction,
                                typename const_ref<KeyType>::type, ArgType> {
};

template <typename ApplyFunction, typename KeyType, typename ArgType>
class is_valid_value_only_apply_function
    : public can_be_called_with<ApplyFunction, ArgType> {};

template <typename ApplyFunction, typename KeyType, typename ArgType>
class is_valid_index_apply_function
    : public can_be_called_with<ApplyFunction,
                                typename const_ref<KeyType>::type,
                                typename const_ref<GroupIndices>::type> {};

/*
 * The following traits are required in order to allow inspection of
 * the only partially defined Derived types inside of GroupByBase.
 *
 * To get GroupByBase to work with other types you need to add a new
 * trait struct for the type.
 */
template <typename T> struct traits {};

template <typename FeatureType, typename GrouperFunction>
struct traits<GroupBy<RegressionDataset<FeatureType>, GrouperFunction>> {
  using KeyType = typename grouper_return_type<GrouperFunction, FeatureType>::type;
  using ValueType = RegressionDataset<FeatureType>;
  using GrouperType = GrouperFunction;
};

template <typename FeatureType, typename GrouperFunction>
struct traits<GroupBy<std::vector<FeatureType>, GrouperFunction>> {
  using KeyType = typename grouper_return_type<GrouperFunction, FeatureType>::type;
  using ValueType = std::vector<FeatureType>;
  using GrouperType = GrouperFunction;
};

} // namespace details

} // namespace albatross

#endif /* ALBATROSS_INDEXING_TRAITS_HPP_ */
