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

template <typename Map1, typename Map2, typename F>
void map_difference(const Map1 &a, const Map2 &b, F &&f) {
  auto less = a.key_comp();
  auto ai = a.begin();
  for (auto bi = b.begin(); ai != a.end() && bi != b.end();) {
    const auto &[ak, av] = *ai;
    const auto &[bk, _] = *bi;
    if (less(ak, bk)) {
      f(ak, av);
      ++ai;
    } else if (less(bk, ak)) {
      ++bi;
    } else {
      ++ai;
      ++bi;
    }
  }
  for (; ai != a.end(); ++ai) {
    f(ai->first, ai->second);
  }
}

template <typename Map1, typename Map2, typename F>
void map_intersect(const Map1 &a, const Map2 &b, F &&f) {
  auto less = a.key_comp();
  auto ai = a.begin();
  auto bi = b.begin();
  while (ai != a.end() && bi != b.end()) {
    const auto &[ak, av] = *ai;
    const auto &[bk, bv] = *bi;
    if (less(ak, bk)) {
      ++ai;
    } else if (less(bk, ak)) {
      ++bi;
    } else {
      f(ak, av, bv);
      ++ai;
      ++bi;
    }
  }
}

template <typename Map, typename Set, typename F>
void map_subset_sequence(const Map &a, const Set &b, F &&f) {
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
      f(ak, av);
      ++ai;
      ++bi;
    }
  }
}

struct MapCompare {
  template <typename A, typename B, typename Compare>
  bool operator()(const A &a, const B &b, Compare &&cmp) const {
    return std::forward<Compare>(cmp)(a.first, b.first);
  }

  template <typename Iterator, typename Compare>
  void sort(Iterator, Iterator, Compare &&) const {};
};

struct SetCompare {
  template <typename A, typename B, typename Compare>
  bool operator()(const A &a, const B &b, Compare &&cmp) const {
    return std::forward<Compare>(cmp)(a, b);
  }

  template <typename Iterator, typename Compare>
  void sort(Iterator begin, Iterator end, Compare &&cmp) const {
    std::sort(begin, end, std::forward<Compare>(cmp));
  };
};

// This helper makes sure `b` is ordered consistently with `a` before
// calling `F` on it.  It tries to do this as efficiently as it can.
template <typename SortedB, typename CompareB, typename A, typename B,
          typename F>
auto call_on_sorted_b(const A &a, const B &b, F &&f) {
  CompareB cmp;
  const auto compare = [&a, &cmp](const auto &ak, const auto &bk) {
    return cmp(ak, bk, a.key_comp());
  };
  if constexpr (has_same_key_compare_v<A, B>) {
    // Happy path: `a` and `b` have the same comparator object
    f(a, b);
  } else {
    // Need a runtime check whether `b` is consistent
    if (!std::is_sorted(b.begin(), b.end(), compare)) {
      // Sadly, `b` is out of order, and we must copy before calling.
      SortedB b_sorted(b.begin(), b.end());
      cmp.sort(b_sorted.begin(), b_sorted.end(), a.key_comp());
      f(a, b_sorted);
    } else {
      // Happily, `b` was still in the right order, so we don't need
      // any sorting.
      f(a, b);
    }
  }
}

} // namespace detail

// Returns a map containing the associations in `a` whose keys are not
// present in `b`.
//
// If `a` and `b` have different key types, `a`'s comparator must
// implement strict weak ordering between the two key types,
// _including_ supporting a comparison between two values of the key
// type of `b`.  The resulting map will have the same key type and
// comparator type as `a`.
//
// If `b` has a different comparator than `a`, a runtime checkwill be
// run to ensure `b` is sorted according to `a`'s comparator, at cost
// O(|b|). If it's not, this function will internally create a sorted
// copy at cost O(|b| log|b|).
template <template <typename...> typename Map1, typename K, typename V,
          typename Compare, template <typename...> typename Map2, typename K2,
          typename V2, typename Compare2>
Map1<K, V, Compare> map_difference(const Map1<K, V, Compare> &a,
                                   const Map2<K2, V2, Compare2> &b) {
  static constexpr bool can_compare_key_values =
      std::is_invocable_r_v<bool, const Compare &, const V2 &, const V2 &>;
  static_assert(can_compare_key_values,
                "`map_difference()` needs to be able to compare the keys of "
                "`b` using `a.key_comp()`");
  Map1<K, V, Compare> diff;
  const auto difference_on_insert = [&diff](const auto &a,
                                            const auto &b_sorted) {
    detail::map_difference(a, b_sorted, [&diff](const auto &k, const auto &v) {
      diff.emplace_hint(diff.end(), k, v);
    });
  };

  detail::call_on_sorted_b<Map2<K2, V2, Compare>, detail::MapCompare>(
      a, b, difference_on_insert);

  return diff;
}

// Returns a sorted vector of the keys in `a` that are not present in
// `b`.
//
// If `a` and `b` have different key types, `a`'s comparator must
// implement strict weak ordering between the two key types,
// _including_ supporting a comparison between two values of the key
// type of `b`.  The resulting map will have the same key type and
// comparator type as `a`.
//
// If `b` has a different comparator than `a`, a runtime checkwill be
// run to ensure `b` is sorted according to `a`'s comparator, at cost
// O(|b|). If it's not, this function will internally create a sorted
// copy at cost O(|b| log|b|).
template <template <typename...> typename Map1, typename K, typename V,
          typename Compare, template <typename...> typename Map2, typename K2,
          typename V2, typename Compare2>
std::vector<K> map_difference_keys(const Map1<K, V, Compare> &a,
                                   const Map2<K2, V2, Compare2> &b) {
  static constexpr bool can_compare_key_values =
      std::is_invocable_r_v<bool, const Compare &, const V2 &, const V2 &>;
  static_assert(can_compare_key_values,
                "`map_difference_keys()` needs to be able to compare the keys "
                "of `b` using `a.key_comp()`");
  std::vector<K> diff;
  const auto difference_on_insert = [&diff](const auto &a,
                                            const auto &b_sorted) {
    detail::map_difference(a, b_sorted, [&diff](const auto &k, const auto &) {
      diff.push_back(k);
    });
  };
  detail::call_on_sorted_b<Map2<K2, V2, Compare>, detail::MapCompare>(
      a, b, difference_on_insert);
  return diff;
}

// Returns a sorted vector of the the keys present in both `a` and
// `b`.
//
// If `a` and `b` have different key types, `a`'s comparator must
// implement strict weak ordering between the two key types,
// _including_ supporting a comparison between two values of the key
// type of `b`.  The resulting map will have the same key type and
// comparator type as `a`.
//
// If `b` has a different comparator than `a`, a runtime checkwill be
// run to ensure `b` is sorted according to `a`'s comparator, at cost
// O(|b|). If it's not, this function will internally create a sorted
// copy at cost O(|b| log|b|).
template <template <typename...> typename Map1, typename K, typename V,
          typename Compare, template <typename...> typename Map2, typename K2,
          typename V2, typename Compare2>
std::vector<K> map_intersect_keys(const Map1<K, V, Compare> &a,
                                  const Map2<K2, V2, Compare2> &b) {
  static constexpr bool can_compare_key_values =
      std::is_invocable_r_v<bool, const Compare &, const V2 &, const V2 &>;
  static_assert(can_compare_key_values,
                "`map_difference_keys()` needs to be able to compare the keys "
                "of `b` using `a.key_comp()`");
  std::vector<K> intersection;
  const auto intersect_on_match = [&intersection](const auto &a,
                                                  const auto &b_sorted) {
    detail::map_intersect(
        a, b_sorted,
        [&intersection](const auto &k, const auto &, const auto &) {
          intersection.push_back(k);
        });
  };
  detail::call_on_sorted_b<Map2<K2, V2, Compare>, detail::MapCompare>(
      a, b, intersect_on_match);
  return intersection;
}

// Removes all associations from `a` whose keys are not present in
// `b`.
//
// If `a` and `b` have different key types, `a`'s comparator must
// implement strict weak ordering between the two key types,
// _including_ supporting a comparison between two values of the key
// type of `b`.  The resulting map will have the same key type and
// comparator type as `a`.
//
// If `b` has a different comparator than `a`, a runtime checkwill be
// run to ensure `b` is sorted according to `a`'s comparator, at cost
// O(|b|). If it's not, this function will internally create a sorted
// copy at cost O(|b| log|b|).
template <typename Map1, typename Map2>
void map_intersect_subset(Map1 &a, const Map2 &b) {
  for (const auto &k : map_difference_keys(a, b)) {
    a.erase(k);
  }
}

// Returns a new map containing the associations from `m` whose keys
// were present in the sequence `keys`.
//
// If `m`'s key type differs from the element type of `keys`, then
// `m`'s comparator must implement strict weak ordering between them,
// _including_ supporting a comparison between two values of the
// element type of `keys`.
//
// You should already have this fully general comparison on hand,
// since you needed it to sort the incoming sequence (for
// e.g. `std::vector`) or construct it (for e.g. `std::set`).
//
// If `keys` is not sorted according to this ordering, it will be
// copied.
template <typename Map, typename Set>
Map map_subset(const Map &m, const Set &keys) {
  static constexpr bool can_compare_key_values =
      std::is_invocable_r_v<bool, const typename Map::key_compare &,
                            const typename Set::value_type &,
                            const typename Set::value_type &>;
  static_assert(can_compare_key_values,
                "`map_subset()` needs to be able to compare the elements of "
                "`keys` using `m.key_comp()`");
  Map results;
  const auto subset_sequence_on_match = [&results](const auto &m,
                                                   const auto &keys_sorted) {
    detail::map_subset_sequence(m, keys_sorted,
                                [&results](const auto &k, const auto &v) {
                                  results.emplace_hint(results.end(), k, v);
                                });
  };
  detail::call_on_sorted_b<std::vector<typename Set::value_type>,
                           detail::SetCompare>(m, keys,
                                               subset_sequence_on_match);
  return results;
}

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
// keys are in `b`)
struct ReturnLeft {
  template <typename L, typename R>
  constexpr auto operator()(L l, R &&) const noexcept {
    return l;
  }
};

// A functor returning its second argument.  Useful for right-biased
// map intersection (when you just want the associations from `b`
// whose keys are in `a`).
struct ReturnRight {
  template <typename L, typename R>
  constexpr auto operator()(L &&, R r) const noexcept {
    return r;
  }
};

// Intersect the maps `a` and `b`, returning a map containing the
// overlapping keys and values computed by calling `f()`, either as
// `f(key, val_a, val_b)` or `f(val_a, val_b)` if the former is not
// available.
//
// By default, `f` is provided and simply pairs up the values and
// returns a `Map<K, std::pair<V1, V2>, Compare>`.  See also
// `ReturnLeft` / `ReturnRight` for intersections that ignore one
// input map's values, or provide your own merge function.
//
// If `a` and `b` have different key types, `a`'s comparator must
// implement strict weak ordering between the two key types,
// _including_ supporting a comparison between two values of the key
// type of `b`.  The resulting map will have the same key type and
// comparator type as `a`.
//
// If `b` has a different comparator than `a`, a runtime check will be
// run to ensure `b` is sorted according to `a`'s comparator, at cost
// O(|b|). If it's not, this function will internally create a sorted
// copy at cost O(|b| log|b|).
template <
    template <typename...> typename Map, typename K, typename V,
    template <typename...> typename Map2, typename K2, typename V2,
    typename Compare, typename Compare2, typename F = MakePair,
    typename = std::enable_if_t<can_call_map_intersection_v<F, K, V, V2>, void>>
auto map_intersect(const Map<K, V, Compare> &a, const Map2<K2, V2, Compare2> &b,
                   F &&f = F{}) {
  static constexpr bool can_compare_key_values =
      std::is_invocable_r_v<bool, Compare &, const K2 &, const K2 &>;
  static_assert(can_compare_key_values,
                "`map_intersect()` needs to be able to compare the elements of "
                "`b` using `a.key_comp()`");
  IntersectedMapType<Map, K, V, V2, F, Compare> intersection;
  const auto intersect_on_match = [&intersection, fn = std::forward<F>(f)](
                                      const auto &a, const auto &b_sorted) {
    detail::map_intersect(
        a, b_sorted,
        [&intersection, fn = std::move(fn)](const auto &k, const auto &av,
                                            const auto &bv) {
          intersection.emplace_hint(intersection.end(), k,
                                    call_map_intersection(fn, k, av, bv));
        });
  };
  detail::call_on_sorted_b<Map2<K2, V2, Compare>, detail::MapCompare>(
      a, b, intersect_on_match);
  return intersection;
}

} // namespace albatross
#endif
