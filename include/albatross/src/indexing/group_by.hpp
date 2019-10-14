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

#ifndef ALBATROSS_INDEXING_GROUPBY_HPP_
#define ALBATROSS_INDEXING_GROUPBY_HPP_

/*
 * Group By
 *
 * This collection of classes facilitates the manipulation of data through
 * the use of split-apply-combine approach.
 */

namespace albatross {

/*
 * The Grouped class is the output of a call to method such as
 *
 *   dataset.groupby(grouper).groups()
 *
 * which is basically just a map with additional functionality
 * that facilitates applying a function to all values, filtering etc ...
 */

template <typename KeyType, typename ValueType> class GroupedBase {

public:
  GroupedBase() : map_(){};
  GroupedBase(const GroupedBase &other) = default;
  GroupedBase(const std::map<KeyType, ValueType> &map) : map_(map){};

  void emplace(const KeyType &k, ValueType &&v) {
    map_.emplace(k, std::move(v));
  }

  void emplace(const KeyType &k, const ValueType &v) { map_.emplace(k, v); }

  auto begin() { return map_.begin(); }

  auto begin() const { return map_.begin(); }

  auto end() { return map_.end(); }

  auto end() const { return map_.end(); }

  std::size_t size() const { return map_.size(); }

  ValueType &operator[](const KeyType &key) { return map_[key]; }

  bool operator==(const GroupedBase<KeyType, ValueType> &other) const {
    return map_ == other.map_;
  }

  auto find(const KeyType &key) const { return map_.find(key); }

  /*
   * Filtering a Grouped object consists of deciding which of the
   * groups you would like to keep.  This is done by providing a function which
   * returns bool when provided with a group (or group key and group)
   */
  template <
      typename FilterFunction,
      typename std::enable_if<details::is_valid_value_only_filter_function<
                                  FilterFunction, ValueType>::value,
                              int>::type = 0>
  auto filter(const FilterFunction &f) const {
    Grouped<KeyType, ValueType> output;
    for (const auto &pair : map_) {
      if (f(pair.second)) {
        output.emplace(pair.first, pair.second);
      }
    }
    return output;
  }

  template <
      typename FilterFunction,
      typename std::enable_if<details::is_valid_key_value_filter_function<
                                  FilterFunction, KeyType, ValueType>::value,
                              int>::type = 0>
  auto filter(const FilterFunction &f) const {
    Grouped<KeyType, ValueType> output;
    for (const auto &pair : map_) {
      if (f(pair.first, pair.second)) {
        output.emplace(pair.first, pair.second);
      }
    }
    return output;
  }

  /*
   * Using Apply with Grouped objects consists of performing some operation
   * to each group.  This is done by provided a function which takes a group
   * (or group key and group).  If the function returns something other than
   * void the results will be aggregated into a new Grouped object.
   */
  template <
      typename ApplyFunction,
      typename ApplyType = typename details::key_value_apply_return_type<
          ApplyFunction, KeyType, ValueType>::type,
      typename std::enable_if<details::is_valid_key_value_apply_function<
                                  ApplyFunction, KeyType, ValueType>::value &&
                                  std::is_same<void, ApplyType>::value,
                              int>::type = 0>
  void apply(const ApplyFunction &f) const {
    for (const auto &pair : map_) {
      f(pair.first, pair.second);
    }
  }

  template <
      typename ApplyFunction,
      typename ApplyType = typename details::key_value_apply_return_type<
          ApplyFunction, KeyType, ValueType>::type,
      typename std::enable_if<details::is_valid_key_value_apply_function<
                                  ApplyFunction, KeyType, ValueType>::value &&
                                  !std::is_same<void, ApplyType>::value,
                              int>::type = 0>
  auto apply(const ApplyFunction &f) const {
    Grouped<KeyType, ApplyType> output;
    for (const auto &pair : map_) {
      output.emplace(pair.first, f(pair.first, pair.second));
    }
    return output;
  }

  template <typename ApplyFunction,
            typename ApplyType = typename details::value_only_apply_return_type<
                ApplyFunction, ValueType>::type,
            typename std::enable_if<details::is_valid_value_only_apply_function<
                                        ApplyFunction, ValueType>::value &&
                                        !std::is_same<void, ApplyType>::value,
                                    int>::type = 0>
  auto apply(const ApplyFunction &f) const {
    Grouped<KeyType, ApplyType> output;
    for (const auto &pair : map_) {
      output.emplace(pair.first, f(pair.second));
    }
    return output;
  }

  template <typename ApplyFunction,
            typename ApplyType = typename details::value_only_apply_return_type<
                ApplyFunction, ValueType>::type,
            typename std::enable_if<details::is_valid_value_only_apply_function<
                                        ApplyFunction, ValueType>::value &&
                                        std::is_same<void, ApplyType>::value,
                                    int>::type = 0>
  auto apply(const ApplyFunction &f) const {
    for (const auto &pair : map_) {
      f(pair.second);
    }
  }

protected:
  std::map<KeyType, ValueType> map_;
};

template <typename KeyType, typename ValueType>
class Grouped : public GroupedBase<KeyType, ValueType> {};

template <typename KeyType>
class Grouped<KeyType, GroupIndices>
    : public GroupedBase<KeyType, GroupIndices> {
public:
  using Base = GroupedBase<KeyType, GroupIndices>;
  using Base::Base;

  template <typename ApplyFunction,
            typename ApplyType = typename details::key_value_apply_return_type<
                ApplyFunction, KeyType, GroupIndices>::type,
            typename std::enable_if<
                details::is_valid_index_apply_function<ApplyFunction, KeyType,
                                                       GroupIndices>::value &&
                    !std::is_same<void, ApplyType>::value,
                int>::type = 0>
  auto index_apply(const ApplyFunction &f) const {
    Grouped<KeyType, ApplyType> output;
    for (const auto &pair : this->map_) {
      output.emplace(pair.first, f(pair.first, pair.second));
    }
    return output;
  }
};

/*
 * combine
 *
 * Like concatenate, but works on map values.
 */
template <template <typename...> class Map, typename KeyType,
          typename FeatureType>
RegressionDataset<FeatureType>
combine(const Map<KeyType, RegressionDataset<FeatureType>> &groups) {
  return concatenate_datasets(map_values(groups));
}

template <template <typename...> class Map, typename KeyType,
          typename FeatureType>
std::vector<FeatureType>
combine(const Map<KeyType, std::vector<FeatureType>> &groups) {
  return concatenate(map_values(groups));
}

template <template <typename...> class Map, typename KeyType>
Eigen::VectorXd combine(const Map<KeyType, double> &groups) {
  Eigen::VectorXd output(static_cast<Eigen::Index>(groups.size()));
  Eigen::Index i = 0;
  for (const auto &x : map_values(groups)) {
    output[i] = x;
    ++i;
  }
  assert(i == output.size());
  return output;
}

/*
 * Combinable grouped objects support operations in which you can
 * collapse a `map<Key, Value>` into a single `Value`
 */
template <typename KeyType, typename ValueType>
class CombinableBase : public GroupedBase<KeyType, ValueType> {
public:
  ValueType combine() const { return albatross::combine(*this); }
};

template <typename KeyType, typename FeatureType>
class Grouped<KeyType, RegressionDataset<FeatureType>>
    : public CombinableBase<KeyType, RegressionDataset<FeatureType>> {};

template <typename KeyType, typename FeatureType>
class Grouped<KeyType, std::vector<FeatureType>>
    : public CombinableBase<KeyType, std::vector<FeatureType>> {};

/*
 * Not all GrouperFunctions actually take Values as input, the leave one
 * out approach (for example) requires knowledge of the total number of
 * indices not the actual values.  This `IndexBuilder` class is responsible
 * for distinguishing between these different approaches.
 */
template <typename GrouperFunction> struct IndexerBuilder {

  template <
      typename Iterable,
      typename IterableValue = typename const_ref<typename std::iterator_traits<
          typename Iterable::iterator>::value_type>::type,
      typename std::enable_if<
          details::is_valid_grouper<GrouperFunction, IterableValue>::value,
          int>::type = 0>
  static auto build(const GrouperFunction &grouper_function,
                    const Iterable &iterable) {
    using GroupKey = typename details::grouper_return_type<GrouperFunction,
                                                           IterableValue>::type;
    GroupIndexer<GroupKey> output;
    std::size_t i = 0;
    for (const auto &value : iterable) {
      const GroupKey group_key = grouper_function(value);
      // Get the existing indices if we've already encountered this group_name
      // otherwise initialize a new one.
      GroupIndices indices;
      if (output.find(group_key) == output.end()) {
        output[group_key] = GroupIndices();
      }
      // Add the current index.
      output[group_key].push_back(i);
      ++i;
    }
    return output;
  }
};

struct LeaveOneOutGrouper {
  template <typename Arg> std::size_t operator()(const Arg &) const {
    static_assert(
        delay_static_assert<Arg>::value,
        "You shouldn't be calling LeaveOneOutGrouper directly, pass it into "
        "group_by as a GrouperFunction");
    return 0;
  };
};

template <> struct IndexerBuilder<LeaveOneOutGrouper> {

  template <typename Iterable>
  static auto build(const LeaveOneOutGrouper &grouper_function,
                    const Iterable &iterable) {
    GroupIndexer<std::size_t> output;
    std::size_t i = 0;
    auto it = iterable.begin();
    while (it != iterable.end()) {
      // Add the current index.
      output[i] = {i};
      ++i;
      ++it;
    }
    return output;
  }
};

template <typename GrouperFunction, typename Iterable>
inline auto build_indexer(const GrouperFunction &grouper_function,
                          const Iterable &iterable) {
  return IndexerBuilder<GrouperFunction>::build(grouper_function, iterable);
}

/*
 * GroupByBase
 *
 * This is the base class holding common operations for classes which can
 * be grouped.
 */
template <typename Derived> class GroupByBase {

public:
  using KeyType = typename details::group_by_traits<Derived>::KeyType;
  using ValueType = typename details::group_by_traits<Derived>::ValueType;
  using GrouperType = typename details::group_by_traits<Derived>::GrouperType;
  using IndexerType = GroupIndexer<KeyType>;

  GroupByBase(const ValueType &parent, const GrouperType &grouper)
      : parent_(parent), grouper_(grouper) {
    indexers_ = build_indexers();
  };

  IndexerType indexers() const { return indexers_; }

  Grouped<KeyType, ValueType> groups() const {
    Grouped<KeyType, ValueType> output;
    for (const auto &key_indexer_pair : indexers()) {
      output.emplace(key_indexer_pair.first,
                     albatross::subset(parent_, key_indexer_pair.second));
    }
    return output;
  }

  std::vector<KeyType> keys() const { return map_keys(indexers()); }

  std::size_t size() const { return indexers().size(); }

  template <typename ApplyFunction> auto apply(const ApplyFunction &f) const {
    return groups().apply(f);
  }

  template <typename ApplyFunction,
            typename ApplyType = typename details::key_value_apply_return_type<
                ApplyFunction, KeyType, GroupIndices>::type,
            typename std::enable_if<
                details::is_valid_index_apply_function<ApplyFunction, KeyType,
                                                       GroupIndices>::value &&
                    !std::is_same<void, ApplyType>::value,
                int>::type = 0>
  auto index_apply(const ApplyFunction &f) const {
    Grouped<KeyType, ApplyType> output;
    for (const auto &pair : indexers()) {
      output.emplace(pair.first, f(pair.first, pair.second));
    }
    return output;
  }

  template <typename ApplyFunction,
            typename ApplyType = typename details::key_value_apply_return_type<
                ApplyFunction, KeyType, GroupIndices>::type,
            typename std::enable_if<
                details::is_valid_index_apply_function<ApplyFunction, KeyType,
                                                       GroupIndices>::value &&
                    std::is_same<void, ApplyType>::value,
                int>::type = 0>
  void index_apply(const ApplyFunction &f) const {
    for (const auto &pair : indexers()) {
      f(pair.first, pair.second);
    }
  }

  template <typename FilterFunction> auto filter(FilterFunction f) const {
    return groups().filter(f);
  }

  std::map<KeyType, std::size_t> counts() const {
    std::map<KeyType, std::size_t> output;
    for (const auto &key_indexer_pair : indexers()) {
      output.emplace(key_indexer_pair.first, key_indexer_pair.second.size());
    }
    return output;
  }

protected:
  ValueType parent_;
  GrouperType grouper_;
  IndexerType indexers_;

private:
  IndexerType build_indexers() const {
    return albatross::build_indexer(grouper_, derived()._get_iterable());
  }

  /*
   * CRTP Helpers
   */
  Derived &derived() { return *static_cast<Derived *>(this); }
  const Derived &derived() const { return *static_cast<const Derived *>(this); }
};

/*
 * GroupBy for RegressionDataset
 */
template <typename FeatureType, typename GrouperFunction>
class GroupBy<RegressionDataset<FeatureType>, GrouperFunction>
    : public GroupByBase<
          GroupBy<RegressionDataset<FeatureType>, GrouperFunction>> {

public:
  using Base =
      GroupByBase<GroupBy<RegressionDataset<FeatureType>, GrouperFunction>>;
  using Base::Base;

  auto &_get_iterable() const { return this->parent_.features; }
};

/*
 * GroupBy for std::vector
 */
template <typename FeatureType, typename GrouperFunction>
class GroupBy<std::vector<FeatureType>, GrouperFunction>
    : public GroupByBase<GroupBy<std::vector<FeatureType>, GrouperFunction>> {

public:
  using Base = GroupByBase<GroupBy<std::vector<FeatureType>, GrouperFunction>>;
  using Base::Base;

  auto &_get_iterable() const { return this->parent_; }
};

/*
 * Define the (already declared) group_by method for datasets.
 */
template <typename FeatureType>
template <typename GrouperFunc>
GroupBy<RegressionDataset<FeatureType>, GrouperFunc>
RegressionDataset<FeatureType>::group_by(GrouperFunc grouper) const {
  return GroupBy<RegressionDataset<FeatureType>, GrouperFunc>(
      *this, std::move(grouper));
}

/*
 * Free functions which create a common way of performing group_by on
 * datasets and vectors.  This is useful because we can't add a .group_by()
 * method to standard library vectors.
 */
template <typename FeatureType, typename GrouperFunc>
auto group_by(const RegressionDataset<FeatureType> &dataset,
              GrouperFunc grouper) {
  return dataset.group_by(std::move(grouper));
}

template <typename FeatureType, typename GrouperFunc>
auto group_by(const std::vector<FeatureType> &vector, GrouperFunc grouper) {
  return GroupBy<std::vector<FeatureType>, GrouperFunc>(vector,
                                                        std::move(grouper));
}

} // namespace albatross

#endif /* ALBATROSS_INDEXING_GROUPBY_HPP_ */
