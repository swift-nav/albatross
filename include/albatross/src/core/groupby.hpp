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

#ifndef ALBATROSS_CORE_GROUPBY_HPP_
#define ALBATROSS_CORE_GROUPBY_HPP_

namespace albatross {

namespace details {
template <typename GrouperFunction, typename FeatureType>
class grouper_return_type {

  template <typename C,
            typename GroupType = decltype(std::declval<const C>()(
                std::declval<typename const_ref<FeatureType>::type>()))>
  static GroupType test(C *);
  template <typename> static void test(...);

public:
  typedef decltype(test<GrouperFunction>(0)) type;
};

template <typename ApplyFunction, typename GroupType, typename ParentType>
class apply_return_type {

  template <typename C, typename ApplyType = decltype(std::declval<const C>()(
                            std::declval<const GroupType &>(),
                            std::declval<const ParentType &>()))>
  static ApplyType test(C *);
  template <typename> static void test(...);

public:
  typedef decltype(test<ApplyFunction>(0)) type;
};

template <typename T> struct traits {};

template <typename FeatureType, typename GrouperFunction>
struct traits<GroupBy<RegressionDataset<FeatureType>, GrouperFunction>> {
  using GroupType =
      typename grouper_return_type<GrouperFunction, FeatureType>::type;
  using ParentType = RegressionDataset<FeatureType>;
  using GrouperType = GrouperFunction;
};

template <typename FeatureType, typename GrouperFunction>
struct traits<GroupBy<std::vector<FeatureType>, GrouperFunction>> {
  using GroupType =
      typename grouper_return_type<GrouperFunction, FeatureType>::type;
  using ParentType = std::vector<FeatureType>;
  using GrouperType = GrouperFunction;
};

} // namespace details

template <typename KeyType, typename FeatureType>
RegressionDataset<FeatureType>
combine(const std::map<KeyType, RegressionDataset<FeatureType>> &groups) {
  return concatenate_datasets(map_values(groups));
}

template <typename KeyType, typename FeatureType>
std::vector<FeatureType>
combine(const std::map<KeyType, std::vector<FeatureType>> &groups) {
  return concatenate(map_values(groups));
}

/*
 * GROUPED
 *
 * This is the output of a call to method such as
 *
 *   dataset.groupby(grouper).groups()
 *
 * which is basically just a map, but contains a
 * few helper functions.
 */

template <typename KeyType, typename ValueType> class Grouped;

template <typename KeyType> class Grouped<KeyType, void> {
public:
  void operator[](const KeyType &) const {};
};

template <typename KeyType, typename FeatureType>
class Grouped<KeyType, RegressionDataset<FeatureType>>
    : public std::map<KeyType, RegressionDataset<FeatureType>> {
public:
  RegressionDataset<FeatureType> combine() const {
    return albatross::combine(*this);
  }
};

template <typename KeyType, typename FeatureType>
class Grouped<KeyType, std::vector<FeatureType>>
    : public std::map<KeyType, std::vector<FeatureType>> {
public:
  std::vector<FeatureType> combine() const { return albatross::combine(*this); }
};

/*
 * GROUP BY BASE
 */
template <typename Derived> class GroupByBase {

public:
  using GroupType = typename details::traits<Derived>::GroupType;
  using ParentType = typename details::traits<Derived>::ParentType;
  using GrouperType = typename details::traits<Derived>::GrouperType;
  using IndexerType = std::unordered_map<GroupType, std::vector<std::size_t>>;

  GroupByBase(const ParentType &parent, const GrouperType &grouper)
      : parent_(parent), grouper_(grouper){};

  //  GroupByBase(const ParentType &parent, const GrouperType &&grouper)
  //  : parent_(parent), grouper_(grouper) {};

  IndexerType indexers() const {
    IndexerType output;
    const auto impl = derived();
    for (std::size_t i = 0; i < impl._get_parent_size(); i++) {
      const GroupType group_key = this->grouper_(impl._get_element(i));
      // Get the existing indices if we've already encountered this group_name
      // otherwise initialize a new one.
      FoldIndexer indices;
      if (output.find(group_key) == output.end()) {
        output[group_key] = FoldIndices();
      }
      // Add the current index.
      output[group_key].push_back(i);
    }
    return output;
  }

  Grouped<GroupType, ParentType> groups() const {
    Grouped<GroupType, ParentType> output;
    for (const auto &key_indexer_pair : indexers()) {
      output[key_indexer_pair.first] =
          albatross::subset(parent_, key_indexer_pair.second);
    }
    return output;
  }

  std::size_t size() const {
    std::unordered_set<GroupType> set;
    const auto impl = derived();
    for (std::size_t i = 0; i < impl._get_parent_size(); i++) {
      set.insert(this->grouper_(impl._get_element(i)));
    }
    return set.size();
  }

  template <typename ApplyFunction,
            typename ApplyType = typename details::apply_return_type<
                ApplyFunction, GroupType, ParentType>::type,
            typename std::enable_if<std::is_same<void, ApplyType>::value,
                                    int>::type = 0>
  void apply(const ApplyFunction &f) const {
    for (const auto &pair : groups()) {
      f(pair.first, pair.second);
    }
  }

  template <typename ApplyFunction,
            typename ApplyType = typename details::apply_return_type<
                ApplyFunction, GroupType, ParentType>::type,
            typename std::enable_if<!std::is_same<void, ApplyType>::value,
                                    int>::type = 0>
  auto apply(const ApplyFunction &f) const {
    Grouped<GroupType, ApplyType> output;
    for (const auto &pair : groups()) {
      output[pair.first] = f(pair.first, pair.second);
    }
    return output;
  }

  std::map<GroupType, std::size_t> counts() const {
    std::map<GroupType, std::size_t> output;
    for (const auto &key_indexer_pair : indexers()) {
      output[key_indexer_pair.first] = key_indexer_pair.second.size();
    }
    return output;
  }

  ParentType parent_;
  GrouperType grouper_;

  /*
   * CRTP Helpers
   */
  Derived &derived() { return *static_cast<Derived *>(this); }
  const Derived &derived() const { return *static_cast<const Derived *>(this); }
};

/*
 * GROUP BY (DATASET)
 *
 * This it the return type of a call to:
 *
 *   dataset.groupby(grouper)
 *
 * it is lazy, nothing happens until you ask for something
 * such as `.groups()`.
 */
template <typename FeatureType, typename GrouperFunction>
class GroupBy<RegressionDataset<FeatureType>, GrouperFunction>
    : public GroupByBase<
          GroupBy<RegressionDataset<FeatureType>, GrouperFunction>> {

public:
  using Base =
      GroupByBase<GroupBy<RegressionDataset<FeatureType>, GrouperFunction>>;
  using Base::Base;

  FeatureType _get_element(const std::size_t &i) const {
    return this->parent_.features[i];
  }

  std::size_t _get_parent_size() const { return this->parent_.size(); }
};

template <typename FeatureType, typename GrouperFunction>
class GroupBy<std::vector<FeatureType>, GrouperFunction>
    : public GroupByBase<GroupBy<std::vector<FeatureType>, GrouperFunction>> {

public:
  using Base = GroupByBase<GroupBy<std::vector<FeatureType>, GrouperFunction>>;
  using Base::Base;

  FeatureType _get_element(const std::size_t &i) const {
    return this->parent_[i];
  }

  std::size_t _get_parent_size() const { return this->parent_.size(); }
};

template <typename FeatureType>
template <typename GrouperFunc>
GroupBy<RegressionDataset<FeatureType>, GrouperFunc>
RegressionDataset<FeatureType>::group_by(const GrouperFunc &&grouper) const {
  return GroupBy<RegressionDataset<FeatureType>, GrouperFunc>(*this, grouper);
}

template <typename FeatureType, typename GrouperFunc>
auto group_by(const RegressionDataset<FeatureType> &dataset,
              const GrouperFunc &&grouper) {
  return dataset.group_by(std::forward<const GrouperFunc>(grouper));
}

template <typename FeatureType, typename GrouperFunc>
auto group_by(const std::vector<FeatureType> &vector,
              const GrouperFunc &grouper) {
  return GroupBy<std::vector<FeatureType>, GrouperFunc>(
      vector, std::forward<const GrouperFunc>(grouper));
}

} // namespace albatross

#endif /* ALBATROSS_CORE_GROUPBY_HPP_ */
