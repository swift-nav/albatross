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

#ifndef ALBATROSS_MODELS_PATCHWORK_GP_DETAILS_HPP_
#define ALBATROSS_MODELS_PATCHWORK_GP_DETAILS_HPP_

namespace albatross {

namespace details {

DEFINE_CLASS_METHOD_TRAITS(boundary);

DEFINE_CLASS_METHOD_TRAITS(nearest_group);

DEFINE_CLASS_METHOD_TRAITS(grouper);

/*
 * Here we make sure that a given class T contains all the required methods
 * to be PatchworkFunctions and that the types involved are consistent.
 */
template <typename T, typename FeatureType>
class patchwork_functions_are_valid {

  using GroupKey = typename class_method_grouper_traits<
      T, typename const_ref<FeatureType>::type>::return_type;

  static constexpr bool has_valid_grouper = group_key_is_valid<GroupKey>::value;

  using ConstRefGroupKey = typename const_ref<GroupKey>::type;

  using BoundaryReturnType =
      typename class_method_boundary_traits<T, ConstRefGroupKey,
                                            ConstRefGroupKey>::return_type;

  static constexpr bool has_valid_boundary =
      is_vector<BoundaryReturnType>::value;

  template <typename Key, std::enable_if_t<!std::is_void<Key>::value, int> = 0,
            typename NearestGroupReturnType =
                typename class_method_nearest_group_traits<
                    T, typename const_ref<std::vector<Key>>::type,
                    ConstRefGroupKey>::return_type>
  static NearestGroupReturnType nearest_group_test(int);

  template <typename C> static void nearest_group_test(...);

  static constexpr bool has_valid_nearest_group =
      std::is_same<decltype(nearest_group_test<GroupKey>(0)), GroupKey>::value;

public:
  static constexpr bool value =
      has_valid_grouper && has_valid_boundary && has_valid_nearest_group;
};

} // namespace details

/*
 * A BoundaryFeature represents a pseudo observation of the difference
 * between predictions from two different models.  In other words,
 *
 *   BoundaryFeature(key_i, key_j, feature)
 *
 * represents the quantity
 *
 *   model_i.predict(feature) - model_j.predict(feature)
 *
 * Patchwork Krigging uses these to force equivalence between two
 * otherwise independent models.  These are the \delta_{k,l} variables
 * in Equation 2 from the paper referenced above.
 */
template <typename GroupKey, typename FeatureType> struct BoundaryFeature {

  BoundaryFeature(const GroupKey &left_group_key_,
                  const GroupKey &right_group_key_, const FeatureType &feature_)
      : left_group_key(left_group_key_), right_group_key(right_group_key_),
        feature(feature_){};

  BoundaryFeature(GroupKey &&left_group_key_, GroupKey &&right_group_key_,
                  FeatureType &&feature_)
      : left_group_key(std::move(left_group_key_)),
        right_group_key(std::move(right_group_key_)),
        feature(std::move(feature_)){};

  GroupKey left_group_key;
  GroupKey right_group_key;
  FeatureType feature;
};

template <typename GroupKey, typename FeatureType>
inline auto as_boundary_feature(GroupKey &&left_group_key,
                                GroupKey &&right_group_key,
                                FeatureType &&feature) {
  using BoundaryFeatureType =
      BoundaryFeature<typename std::decay<GroupKey>::type,
                      typename std::decay<FeatureType>::type>;
  return BoundaryFeatureType(std::forward<GroupKey>(left_group_key),
                             std::forward<GroupKey>(right_group_key),
                             std::forward<FeatureType>(feature));
}

template <typename GroupKey, typename FeatureType>
inline auto as_boundary_features(GroupKey &&left_group_key,
                                 GroupKey &&right_group_key,
                                 const std::vector<FeatureType> &features) {
  using BoundaryFeatureType =
      BoundaryFeature<typename std::decay<GroupKey>::type,
                      typename std::decay<FeatureType>::type>;

  std::vector<BoundaryFeatureType> boundary_features;
  for (const auto &f : features) {
    boundary_features.emplace_back(
        as_boundary_feature(left_group_key, right_group_key, f));
  }
  return boundary_features;
}

/*
 * GroupFeature
 *
 * This is used to indicate which model a particular Feature corresponds
 * to.  It corresponds (loosely) to the f_i in Equations 3 and 4 from
 * the paper referenced above.
 */

template <typename GroupKey, typename FeatureType> struct GroupFeature {

  GroupFeature() : group_key(), feature(){};

  GroupFeature(const GroupKey &key_, const FeatureType &feature_)
      : group_key(key_), feature(feature_){};

  GroupFeature(GroupKey &&key_, FeatureType &&feature_)
      : group_key(std::move(key_)), feature(std::move(feature_)){};

  GroupKey group_key;
  FeatureType feature;
};

template <typename GroupKey, typename FeatureType>
inline auto as_group_feature(GroupKey &&key, FeatureType &&feature) {
  using GroupFeatureType = GroupFeature<typename std::decay<GroupKey>::type,
                                        typename std::decay<FeatureType>::type>;
  return GroupFeatureType(std::forward<GroupKey>(key),
                          std::forward<FeatureType>(feature));
}

template <typename GroupKey, typename FeatureType>
inline auto as_group_feature(GroupKey &&key ALBATROSS_UNUSED,
                             GroupFeature<GroupKey, FeatureType> &feature) {
  return feature;
}

template <typename GrouperFunction, typename FeatureType>
inline auto as_group_features(const std::vector<FeatureType> &features,
                              const GrouperFunction &grouper_function) {

  using GroupKey =
      typename GroupBy<std::vector<FeatureType>, GrouperFunction>::KeyType;
  using GroupFeatureType =
      GroupFeature<GroupKey, typename std::decay<FeatureType>::type>;

  std::vector<GroupFeatureType> group_features(features.size());

  auto emplace_in_output = [&](const auto &key, const auto &idx) {
    for (const auto &i : idx) {
      group_features[i] = as_group_feature(key, features[i]);
    }
  };
  group_by(features, grouper_function).index_apply(emplace_in_output);

  return group_features;
}

template <typename GroupKey, typename FeatureType>
inline auto as_group_features(const GroupKey &key,
                              const std::vector<FeatureType> &features) {
  using GroupFeatureType =
      GroupFeature<GroupKey, typename std::decay<FeatureType>::type>;

  std::vector<GroupFeatureType> group_features;
  for (const auto &f : features) {
    group_features.emplace_back(as_group_feature(key, f));
  }
  return group_features;
}

namespace internal {

/*
 * Here we define the rules laid out in Equations 3 and 4 of the referenced
 * paper.  The rules consist of defining the covariance between boundary
 * features, group features and in the trivial case two normal features.
 */
template <typename SubCaller> struct PatchworkCallerBase {
  template <
      typename CovFunc, typename X, typename Y,
      typename std::enable_if<
          has_valid_cov_caller<CovFunc, SubCaller, X, Y>::value, int>::type = 0>
  static double call(const CovFunc &cov_func, const X &x, const Y &y) {
    // The trivial case, forward on to the underlying covariance.
    return SubCaller::call(cov_func, x, y);
  }

  template <typename CovFunc, typename GroupKey, typename FeatureTypeX,
            typename FeatureTypeY>
  static double call(const CovFunc &cov_func,
                     const GroupFeature<GroupKey, FeatureTypeX> &x,
                     const GroupFeature<GroupKey, FeatureTypeY> &y) {
    // The covariance between any two group features is only defined if
    // the two are in the same group.
    if (x.group_key == y.group_key) {
      return SubCaller::call(cov_func, x.feature, y.feature);
    } else {
      return 0.;
    }
  }

  template <typename CovFunc, typename GroupKey, typename FeatureTypeX,
            typename FeatureTypeY>
  static double call(const CovFunc &cov_func,
                     const GroupFeature<GroupKey, FeatureTypeX> &x,
                     const BoundaryFeature<GroupKey, FeatureTypeY> &y) {
    // This is Equation 3 in the referenced paper.
    if (x.group_key == y.left_group_key) {
      return SubCaller::call(cov_func, x.feature, y.feature);
    } else if (x.group_key == y.right_group_key) {
      return -SubCaller::call(cov_func, x.feature, y.feature);
    } else {
      return 0.;
    }
  }

  template <typename CovFunc, typename GroupKey, typename FeatureTypeX,
            typename FeatureTypeY>
  static double call(const CovFunc &cov_func,
                     const BoundaryFeature<GroupKey, FeatureTypeX> &x,
                     const BoundaryFeature<GroupKey, FeatureTypeY> &y) {
    // This is Equation 4 in the referenced paper.
    if (x.left_group_key == y.left_group_key &&
        x.right_group_key == y.right_group_key) {
      return 2 * SubCaller::call(cov_func, x.feature, y.feature);
    } else if (x.left_group_key == y.right_group_key &&
               x.right_group_key == y.left_group_key) {
      return -2 * SubCaller::call(cov_func, x.feature, y.feature);
    } else if (x.left_group_key == y.left_group_key &&
               x.right_group_key != y.right_group_key) {
      return SubCaller::call(cov_func, x.feature, y.feature);
    } else if (x.left_group_key != y.left_group_key &&
               x.right_group_key == y.right_group_key) {
      return SubCaller::call(cov_func, x.feature, y.feature);
    } else if (x.left_group_key == y.right_group_key &&
               x.right_group_key != y.left_group_key) {
      return -SubCaller::call(cov_func, x.feature, y.feature);
    } else if (x.left_group_key != y.right_group_key &&
               x.right_group_key == y.left_group_key) {
      return -SubCaller::call(cov_func, x.feature, y.feature);
    } else {
      return 0.;
    }
  }
};

} // namespace internal

// The PatchworkCaller tries symmetric versions of the PatchworkCallerBase
// and otherwise resorts to the DefaultCaller
using PatchworkCaller =
    internal::SymmetricCaller<internal::PatchworkCallerBase<DefaultCaller>>;

} // namespace albatross

#endif /* ALBATROSS_MODELS_PATCHWORK_GP_DETAILS_HPP_ */
