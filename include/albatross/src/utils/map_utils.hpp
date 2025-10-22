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

#ifndef ALBATROSS_MAP_UTILS_H
#define ALBATROSS_MAP_UTILS_H

namespace albatross {

/*
 * Convenience function instead of using the find == end
 * method of determining if a key exists in a map.
 */
template <typename Map, typename K,
          std::enable_if_t<has_find_key_v<Map, K>, int> = 0>
inline bool map_contains(const Map &m, const K &k) {
  return m.find(k) != m.end();
}

/*
 * Convenience function which returns m.at(k) unless k is not in
 * the map in which case it uses either the default constructor
 * or the provided alternate value.
 */
template <template <typename...> class Map, typename K, typename SearchKey,
          typename V, typename... More,
          typename = std::enable_if_t<
              has_find_key_v<Map<K, V, More...>, SearchKey>, void>>
inline V map_at_or(const Map<K, V, More...> &m, const SearchKey &k,
                   const V &value_if_missing = V()) {
  const auto iter = m.find(k);
  if (iter == m.end()) {
    return value_if_missing;
  }
  return iter->second;
}

/*
 * Returns a vector consisting of all the keys in a map.
 */
template <template <typename...> class Map, typename K, typename V,
          typename... More>
inline std::vector<K> map_keys(const Map<K, V, More...> &m) {
  std::vector<K> keys;
  keys.reserve(m.size());
  for (const auto &[key, _] : m) {
    keys.push_back(key);
  }
  return keys;
}

/*
 * Returns a vector consisting of all the values in a map.
 */
template <template <typename...> class Map, typename K, typename V,
          typename... More>
inline std::vector<V> map_values(const Map<K, V, More...> &m) {
  std::vector<V> values;
  values.reserve(m.size());
  for (const auto &[_, value] : m) {
    values.push_back(value);
  }
  return values;
}

template <template <typename...> class Map, typename K, typename V,
          typename... More>
inline Map<K, V, More...> map_join(const Map<K, V, More...> &m,
                                   const Map<K, V, More...> &other) {
  Map<K, V, More...> join(other);
  // Note the order here is reversed since insert will not insert if a key
  // already exists, in this case we want the result to contain all elements of
  // m overwritten by any elements in other.
  join.insert(m.begin(), m.end());
  return join;
}

template <template <typename...> class Map, typename K, typename V,
          typename... More>
inline Map<K, V, More...> map_join_strict(const Map<K, V, More...> &m,
                                          const Map<K, V, More...> &other) {
  Map<K, V, More...> join(other);
  // Note the order here is reversed since insert will not insert if a key
  // already exists, in this case we want the result to contain all elements
  // of m overwritten by any elements in other.
  for (const auto &[key, value] : m) {
    if (map_contains(join, key)) {
      // duplicate key found in map_join.
      ALBATROSS_ASSERT(false && "duplicate key found");
    }
    join[key] = value;
  }
  return join;
}

namespace detail {

// Implementation notes
//
// These helpers take in consistently ordered maps / sequences and
// perform the desired operation in a single linear pass.
//
// Your functor will be called in key order, so you should use
// `emplace_hint()` or `push_back()`.
template <typename Map1, typename Map2, typename OnlyA, typename OnlyB,
          typename Merge>
void map_merge(const Map1 &a, const Map2 &b, OnlyA &&only_a, OnlyB &&only_b,
               Merge &&merge) {
  auto less = a.key_comp();
  auto ai = a.begin();
  auto bi = b.begin();
  while (ai != a.end() && bi != b.end()) {
    const auto &[ak, av] = *ai;
    const auto &[bk, bv] = *bi;
    if (less(ak, bk)) {
      only_a(ak, av);
      ++ai;
    } else if (less(bk, ak)) {
      only_b(bk, bv);
      ++bi;
    } else {
      merge(ak, av, bv);
      ++ai;
      ++bi;
    }
  }
  for (; ai != a.end(); ++ai) {
    only_a(ai->first, ai->second);
  }
  for (; bi != b.end(); ++bi) {
    only_b(bi->first, bi->second);
  }
}

template <typename Map, typename Set, typename Merge>
void map_subset_sequence(const Map &a, const Set &b, Merge &&merge) {
  auto less = a.key_comp();
  auto ai = a.begin();
  auto bi = b.begin();
  while (ai != a.end() && bi != b.end()) {
    const auto &[ak, av] = *ai;
    const auto bk = *bi;
    if (less(ak, bk)) {
      ++ai;
    } else if (less(bk, ak)) {
      ++bi;
    } else {
      merge(ak, av);
      ++ai;
      ++bi;
    }
  }
}

struct DoNothing {
  template <typename... Args>
  constexpr void operator()(const Args &...) const noexcept {}
};

template <typename Map1, typename Map2, typename F>
void map_difference(const Map1 &a, const Map2 &b, F &&f) {
  map_merge(a, b, std::forward<F>(f), DoNothing{}, DoNothing{});
}

template <typename Map1, typename Map2, typename F>
void map_symmetric_difference(const Map1 &a, const Map2 &b, F &&f) {
  map_merge(a, b, std::forward<F>(f), std::forward<F>(f), DoNothing{});
}

template <typename Map1, typename Map2, typename F>
void map_intersect(const Map1 &a, const Map2 &b, F &&f) {
  map_merge(a, b, DoNothing{}, DoNothing{}, std::forward<F>(f));
}

template <typename Map1, typename Map2, typename AddA, typename AddB,
          typename MergeAdd>
void map_union(const Map1 &a, const Map2 &b, AddA &&add_a, AddB &&add_b,
               MergeAdd &&merge_add) {
  map_merge(a, b, add_a, add_b, merge_add);
}

} // namespace detail

// Efficient map operations
//
// The following functions offer similar logic to
// `std::set_difference()`, `std::set_intersection()` and friends, but
// with slightly simpler behavior and interfaces (not dealing manually
// with iterators everywhere), better checking of ordering
// requirements and some map-specific operations like subsetting
// without having to do extra lambda gymnastics every time.
//
// The functions below operate via a linear merge pass over two
// ordered containers for efficiency; this means consistent ordering
// is crucial for correctness.
//
//  - Both containers (`a` and `b`) must have the same comparator type
//    where applicable, accessible via a member type `key_compare` or
//    as a third template parameter `Compare`.  Either this comparator
//    must be stateless, or both instances must have the same state.
//
//  - This value must be accessible at runtime via a `.key_comp()`
//    member.
//
//  - If the container has no inherent ordering (`std::vector` e.g.),
//    the elements of the container must be sorted according to the
//    `key_compare` of the other argument.
//
//  - Map types must have a `key_type` member type corresponding to
//    the key type and a `mapped_type` member type corresponding to
//    the value type of each association.
//
//  - Any argument type (map or sequence) must offer forward iterators
//    accessible via `.begin()` and `.end()`.
//
//  - Map types used for returned maps (i.e. `a`'s map type) must
//    offer `.emplace_hint(const_iterator, key, value)`.
//
// Since this comparator must be the same between both types, if `a`
// and `b` have different key types, this comparator must implement
// strict weak ordering between the two key types.  This basically
// means it must provide the following 4 operations (or equivalent via
// templated members):
//
//    bool operator()(const decltype(a)::key_type &ak,
//                    const decltype(b)::key_type &bk) const;
//
//    bool operator()(const decltype(b)::key_type &bk,
//                    const decltype(a)::key_type &ak) const;
//
//    bool operator()(const decltype(a)::key_type &ak1,
//                    const decltype(a)::key_type &ak2) const;
//
//    bool operator()(const decltype(b)::key_type &bk1,
//                    const decltype(b)::key_type &bk2) const;
//
// The functions are all "left-biased", in several senses:
//
//  1. The return type will be the same as the type of `a`, or, where
//      it doesn't exactly match, something drawn from `a`'s member
//      types / template arguments (e.g. a
//      `std::vector<decltype(a)::key_type>`).
//
//  2. For asymmetric operations (subset, difference), the function
//     name describes the logic relative to `a`, e.g. `map_subset(a,
//     b)` computes a subset of `a`.
//
// For `map_intersect()` and `map_union()`, the function to be called
// whenever an equivalent key appears in both sequences is defaulted
// but may be overriden by the user.  Several basic merge functors are
// provided (`MakePair`, `ReturnLeft`, `ReturnRight`).  You may define
// your own; it must support either
//
//    T operator()(const decltype(a)::key_type &ak,
//                 const decltype(a)::mapped_type &av,
//                 const decltype(b)::mapped_type &bv) const;
//
// or
//
//    T operator()(const decltype(a)::mapped_type &av,
//                 const decltype(b)::mapped_type &bv) const;
//
// if your merge does the same thing for any key.  `T` here is the
// appropriate return type for your operation.
//
// The asymptotic performance of all of these is O(|a| + |b|) in time.
// Each allocates a new map or other container (and its contents) and
// returns by value (using RVO).
//
// Basic safety is guaranteed -- if an exception is thrown, any
// partial results will be correctly destroyed.

// Returns a map containing the associations in `a` whose keys are not
// present in `b`.
template <typename Map1, typename Map2,
          typename = std::enable_if_t<has_same_key_compare_v<Map1, Map2>, void>>
[[nodiscard]] Map1 map_difference(const Map1 &a, const Map2 &b) {
  Map1 diff;
  const auto only_a = [&diff](const auto &k, const auto &v) {
    diff.emplace_hint(diff.end(), k, v);
  };
  detail::map_difference(a, b, only_a);
  return diff;
}

template <
    typename Map1, typename Map2,
    typename = std::enable_if_t<!has_same_key_compare_v<Map1, Map2>, void>,
    typename = void>
Map1 map_difference(const Map1 &, const Map2 &) {
  static_assert(
      has_same_key_compare_v<Map1, Map2>,
      "map_difference requires both maps to have the same comparator type. "
      "Check that Map1::key_compare and Map2::key_compare are the same type.");
  return {};
}

// Returns a map containing the associations in `a` whose keys are not
// present in `b` and those present in `b` whose keys are not present
// in `a`.
//
// If the maps have different key types, then `b`'s key type must be
// implicitly convertible to `a`'s (and the associations only in `b`
// will be inserted into the resulting map using the converted key
// values).  It is your responsibility to ensure this conversion is
// bijective (i.e. each value of `Map2::key_type` corresponds to only
// one value of `Map1::key_type`.
template <typename Map1, typename Map2,
          typename = std::enable_if_t<
              has_same_key_compare_v<Map1, Map2> &&
                  std::is_convertible_v<typename Map2::key_type,
                                        typename Map1::key_type> &&
                  std::is_same_v<typename Map1::mapped_type,
                                 typename Map2::mapped_type>,
              void>>
[[nodiscard]] Map1 map_symmetric_difference(const Map1 &a, const Map2 &b) {
  Map1 diff;
  const auto either = [&diff](const auto &k, const auto &v) {
    diff.emplace_hint(diff.end(), k, v);
  };
  detail::map_symmetric_difference(a, b, either);
  return diff;
}

template <typename Map1, typename Map2,
          typename = std::enable_if_t<
              !has_same_key_compare_v<Map1, Map2> ||
                  !std::is_convertible_v<typename Map2::key_type,
                                         typename Map1::key_type> ||
                  !std::is_same_v<typename Map1::mapped_type,
                                  typename Map2::mapped_type>,
              void>,
          typename = void>
Map1 map_symmetric_difference(const Map1 &, const Map2 &) {
  static_assert(
      has_same_key_compare_v<Map1, Map2>,
      "map_symmetric_difference requires both maps to have the same "
      "comparator type (Map1::key_compare must equal Map2::key_compare)");

  static_assert(
      std::is_convertible_v<typename Map2::key_type, typename Map1::key_type>,
      "map_symmetric_difference requires Map2::key_type to be implicitly "
      "convertible to Map1::key_type, because keys from `b` will be "
      "inserted into the result map");

  static_assert(
      std::is_same_v<typename Map1::mapped_type, typename Map2::mapped_type>,
      "map_symmetric_difference requires both maps to have the same "
      "value type (Map1::mapped_type must equal Map2::mapped_type)");

  return {};
}

// A functor to call `make_pair()`.  This is only useful for the C++
// reasons that make it awkward to pass `std::make_pair()` directly.
struct MakePair {
  template <typename V1, typename V2>
  constexpr auto operator()(V1 &&v1, V2 &&v2) const
      noexcept(noexcept(std::make_pair(std::forward<V1>(v1),
                                       std::forward<V2>(v2)))) {
    return std::make_pair(std::forward<V1>(v1), std::forward<V2>(v2));
  }
};

// A functor returning its first argument.  Useful for left-biased map
// intersection (when you just want the associations from `a` whose
// keys are in `b`) or union (when you want to keep `a`'s values on
// overlap).
struct ReturnLeft {
  template <typename L, typename R>
  constexpr auto operator()(L l, R &&) const noexcept {
    return l;
  }
};

// A functor returning its second argument.  Useful for right-biased
// map intersection (when you just want the associations from `b`
// whose keys are in `a`) or union (when you want to keep `b`'s values
// on overlap).
struct ReturnRight {
  template <typename L, typename R>
  constexpr auto operator()(L &&, R r) const noexcept {
    return r;
  }
};

// Union the maps `a` and `b`, returning a map containing the the
// associations in either input, with values for overlapping keys
// computed by calling `merge()`, either as `merge(key, val_a, val_b)`
// or `merge(val_a, val_b)` if the former is not available.  `merge()` must
// return the value type, which must be `::mapped_type` of both `a` and `b`.
//
// By default, `merge` is provided and simply preserves the value from
// `a`.  See also `ReturnRight` for the opposite behaviour, or provide
// your own merge function.
//
// If the maps have different key types, then `b`'s key type must be
// implicitly convertible to `a`'s.  It is your responsibility to
// ensure this conversion is bijective (i.e. each value of
// `Map2::key_type` corresponds to only one value of `Map1::key_type`.
template <typename Map1, typename Map2, typename Merge = ReturnLeft,
          typename = std::enable_if_t<
              has_same_key_compare_v<Map1, Map2> &&
                  std::is_convertible_v<typename Map2::key_type,
                                        typename Map1::key_type> &&
                  std::is_same_v<typename Map1::mapped_type,
                                 typename Map2::mapped_type> &&
                  can_call_map_union_v<Merge, typename Map1::key_type,
                                       typename Map1::mapped_type>,
              void>>
[[nodiscard]] Map1 map_union(const Map1 &a, const Map2 &b,
                             Merge &&merge = ReturnLeft{}) {
  Map1 result;
  const auto add_a = [&result](const auto &k, const auto &av) {
    result.emplace_hint(result.end(), k, av);
  };
  const auto add_b = [&result](const auto &bk, const auto &bv) {
    result.emplace_hint(result.end(), bk, bv);
  };
  const auto both = [&result, merge = std::forward<Merge>(merge)](
                        const auto &k, const auto &av, const auto &bv) {
    result.emplace_hint(result.end(), k, call_map_union(merge, k, av, bv));
  };
  detail::map_union(a, b, add_a, add_b, both);
  return result;
}

template <typename Map1, typename Map2, typename Merge = ReturnLeft,
          typename = std::enable_if_t<
              !has_same_key_compare_v<Map1, Map2> ||
                  !std::is_convertible_v<typename Map2::key_type,
                                         typename Map1::key_type> ||
                  !std::is_same_v<typename Map1::mapped_type,
                                  typename Map2::mapped_type> ||
                  !can_call_map_union_v<Merge, typename Map1::key_type,
                                        typename Map1::mapped_type>,
              void>,
          typename = void>
Map1 map_union(const Map1 &, const Map2 &, Merge &&) {
  static_assert(
      has_same_key_compare_v<Map1, Map2>,
      "map_union requires both maps to have the same comparator type. "
      "Check that Map1::key_compare and Map2::key_compare are the same type. "
      "Both containers must be sorted by the same comparison function.");

  static_assert(
      std::is_convertible_v<typename Map2::key_type, typename Map1::key_type>,
      "map_union requires Map2::key_type to be implicitly convertible to "
      "Map1::key_type. Keys from the second map will be inserted into the "
      "result (which has Map1's key type), so the conversion must be valid.");

  static_assert(
      std::is_same_v<typename Map1::mapped_type, typename Map2::mapped_type>,
      "map_union requires both maps to have the same value type. "
      "Check that Map1::mapped_type and Map2::mapped_type are identical.");

  static_assert(
      can_call_map_union_v<Merge, typename Map1::key_type,
                           typename Map1::mapped_type>,
      "map_union requires the merge function to be callable as either "
      "merge(key, val_a, val_b) or merge(val_a, val_b), where key is "
      "Map1::key_type and val_a/val_b are Map1::mapped_type.");

  return {};
}

// Intersect the maps `a` and `b`, returning a map containing the
// overlapping keys and values computed by calling `f()`, either as
// `merge(key, val_a, val_b)` or `merge(val_a, val_b)` if the former
// is not available.  The return type of `merge()` when this call is
// made is the `mapped_type` (second template argument) of the
// resulting map.
//
// By default, `merge` is provided and simply pairs up the values and
// returns a `Map<K, std::pair<V1, V2>, Compare>`.  See also
// `ReturnLeft` / `ReturnRight` for intersections that ignore one
// input map's values, or provide your own merge function.
template <template <typename...> typename Map, typename K, typename V,
          typename Compare, typename Map2, typename Merge = MakePair,
          typename = std::enable_if_t<
              has_same_key_compare_v<Map<K, V, Compare>, Map2> &&
                  can_call_map_intersection_v<Merge, K, V,
                                              typename Map2::mapped_type>,
              void>>
[[nodiscard]] IntersectedMapType<Map, K, V, typename Map2::mapped_type, Merge,
                                 Compare>
map_intersect(const Map<K, V, Compare> &a, const Map2 &b,
              Merge &&merge = MakePair{}) {
  IntersectedMapType<Map, K, V, typename Map2::mapped_type, Merge, Compare>
      intersection;
  const auto on_both = [&intersection, merge = std::forward<Merge>(merge)](
                           const auto &k, const auto &av, const auto &bv) {
    intersection.emplace_hint(intersection.end(), k,
                              call_map_intersection(merge, k, av, bv));
  };
  detail::map_intersect(a, b, on_both);
  return intersection;
}

template <template <typename...> typename Map, typename K, typename V,
          typename Compare, typename Map2, typename Merge = MakePair,
          typename = std::enable_if_t<
              !has_same_key_compare_v<Map<K, V, Compare>, Map2> ||
                  !can_call_map_intersection_v<Merge, K, V,
                                               typename Map2::mapped_type>,
              void>,
          typename = void>
IntersectedMapType<Map, K, V, typename Map2::mapped_type, Merge, Compare>
map_intersect(const Map<K, V, Compare> &, const Map2 &, Merge &&) {
  static_assert(
      has_same_key_compare_v<Map<K, V, Compare>, Map2>,
      "map_intersect requires both maps to have the same comparator type. "
      "Check that the comparators of both input maps are the same type.");

  static_assert(
      can_call_map_intersection_v<Merge, K, V, typename Map2::mapped_type>,
      "map_intersect requires the merge function to be callable as either "
      "merge(key, val_a, val_b) or merge(val_a, val_b), where key is the "
      "key type of the first map and val_a/val_b are the value types from "
      "each map. The return type of merge determines the value type of the "
      "result.");

  return {};
}

// Returns a sorted vector of the keys present in `a` but not `b`.
template <typename Map1, typename Map2,
          typename = std::enable_if_t<has_same_key_compare_v<Map1, Map2>, void>>
[[nodiscard]] std::vector<typename Map1::key_type>
map_difference_keys(const Map1 &a, const Map2 &b) {
  std::vector<typename Map1::key_type> diff;
  const auto only_a = [&diff](const auto &k, const auto &) {
    diff.push_back(k);
  };
  detail::map_difference(a, b, only_a);
  return diff;
}

template <
    typename Map1, typename Map2,
    typename = std::enable_if_t<!has_same_key_compare_v<Map1, Map2>, void>,
    typename = void>
std::vector<typename Map1::key_type> map_difference_keys(const Map1 &,
                                                         const Map2 &) {
  static_assert(
      has_same_key_compare_v<Map1, Map2>,
      "map_difference_keys requires both maps to have the same comparator "
      "type. "
      "Check that Map1::key_compare and Map2::key_compare are the same type.");
  return {};
}

// Returns a sorted vector of the keys present in both `a` and `b`.
template <typename Map1, typename Map2,
          typename = std::enable_if_t<has_same_key_compare_v<Map1, Map2>, void>>
[[nodiscard]] std::vector<typename Map1::key_type>
map_intersect_keys(const Map1 &a, const Map2 &b) {
  std::vector<typename Map1::key_type> intersection;
  const auto on_match = [&intersection](const auto &k, const auto &,
                                        const auto &) {
    intersection.push_back(k);
  };
  detail::map_intersect(a, b, on_match);
  return intersection;
}

template <
    typename Map1, typename Map2,
    typename = std::enable_if_t<!has_same_key_compare_v<Map1, Map2>, void>,
    typename = void>
std::vector<typename Map1::key_type> map_intersect_keys(const Map1 &,
                                                        const Map2 &) {
  static_assert(
      has_same_key_compare_v<Map1, Map2>,
      "map_intersect_keys requires both maps to have the same comparator type. "
      "Check that Map1::key_compare and Map2::key_compare are the same type.");
  return {};
}

// Returns a new map containing the associations from `m` whose keys
// were present in the sequence `keys`.  `keys` must be an ordered
// container with the usual interface of `::key_compare` /
// `.key_comp()` and whose `key_compare` type matches that of `m`.
//
// If you have something like a properly sorted
// `std::vector<m::key_type>` (with no inherent ordering), for example
// from calling `map_(intersect|difference)_keys()` or `map_keys()`,
// use `map_subset_sorted`.
template <
    typename Map, typename Sequence,
    typename = std::enable_if_t<has_same_key_compare_v<Map, Sequence>, void>>
[[nodiscard]] Map map_subset(const Map &m, const Sequence &keys) {
  Map results;
  const auto on_match = [&results](const auto &k, const auto &v) {
    results.emplace_hint(results.end(), k, v);
  };
  detail::map_subset_sequence(m, keys, on_match);
  return results;
}

template <
    typename Map, typename Sequence,
    typename = std::enable_if_t<!has_same_key_compare_v<Map, Sequence>, void>,
    typename = void>
Map map_subset(const Map &, const Sequence &) {
  static_assert(has_same_key_compare_v<Map, Sequence>,
                "map_subset requires the key sequence to have the same "
                "comparator type as the map. "
                "This typically means using std::set or std::map with the same "
                "comparator. "
                "If you have a std::vector or other unordered sequence, use "
                "map_subset_sorted() instead.");
  return {};
}

// Returns a new map containing the associations from `m` whose keys
// were present in the sequence `keys`.  This accepts any container of
// `Map::key_type`, whether or not it has an inherent ordering or a
// `::key_compare` member type.  `keys` *must* be sorted according to
// the comparator `m.key_comp()` (`Map::key_compare`); if they are
// not, this function will assert.
//
// If you have something like a `std::set<m::key_type>`, you should
// use `map_subset`.
template <typename Map, typename Sequence>
[[nodiscard]] Map map_subset_sorted(const Map &m, const Sequence &keys) {
  ALBATROSS_ASSERT(std::is_sorted(keys.begin(), keys.end(), m.key_comp()) &&
                   "You promised when you called this function that the `keys` "
                   "would be sorted!");
  Map results;
  const auto on_match = [&results](const auto &k, const auto &v) {
    results.emplace_hint(results.end(), k, v);
  };
  detail::map_subset_sequence(m, keys, on_match);
  return results;
}

} // namespace albatross
#endif
