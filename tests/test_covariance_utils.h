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

#ifndef TESTS_TEST_COVARIANCE_UTILS_H_
#define TESTS_TEST_COVARIANCE_UTILS_H_

namespace albatross {

struct V {};
struct W {};

struct X {
  bool operator==(const X &) const { return false; }
};

struct Y {};
struct Z {};

class HasXX : public CovarianceFunction<HasXX> {
public:
  double _call_impl(const X &, const X &) const { return 1.; };
};

class HasXY : public CovarianceFunction<HasXY> {
public:
  double _call_impl(const X &, const Y &) const { return 1.; };
};

class HasNone : public CovarianceFunction<HasNone> {};

class HasMultiple : public CovarianceFunction<HasMultiple> {
public:
  double _call_impl(const X &, const X &) const { return 1.; };

  double _call_impl(const X &, const Y &) const { return 3.; };

  double _call_impl(const Y &, const Y &) const { return 5.; };

  double _call_impl(const W &, const W &) const { return 7.; };

  double _call_impl(const V &, const V &) const { return 11.; };

  // These are all invalid:
  double _call_impl(const Z &, const X &) { return 1.; };

  double _call_impl(Z &, const Y &) const { return 1.; };

  int _call_impl(const Z &, const Z &) const { return 1.; };

  std::string name() const { return "has_multiple"; };
};

class HasPublicCallImpl {
public:
  double _call_impl(const X &, const Y &) const { return 1.; };
};

class HasProtectedCallImpl {
protected:
  double _call_impl(const X &, const Y &) const { return 1.; };
};

class HasPrivateCallImpl {
  double _call_impl(const X &, const Y &) const { return 1.; };
};

class HasNoCallImpl {};

/*
 * Test classes for get_ssr_features
 */

struct TestSSR {};
struct OtherSSR {};

class HasTestSSR : public CovarianceFunction<HasTestSSR> {
public:
  std::vector<TestSSR> _ssr_features(const std::vector<X> &) const {
    return {TestSSR()};
  }
};

class AlsoHasTestSSR : public CovarianceFunction<AlsoHasTestSSR> {
public:
  std::vector<TestSSR> _ssr_features(const std::vector<X> &) const {
    return {TestSSR(), TestSSR(), TestSSR()};
  }
};

class HasOtherSSR : public CovarianceFunction<HasOtherSSR> {
public:
  std::vector<OtherSSR> _ssr_features(const std::vector<X> &) const {
    return {OtherSSR(), OtherSSR(), OtherSSR(), OtherSSR(), OtherSSR()};
  }
};

} // namespace albatross

#endif /* TESTS_TEST_COVARIANCE_UTILS_H_ */
