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

#ifndef INCLUDE_ALBATROSS_SRC_DETAILS_HAS_ANY_MACROS_HPP_
#define INCLUDE_ALBATROSS_SRC_DETAILS_HAS_ANY_MACROS_HPP_

namespace albatross {

#define HAS_METHOD(fname)                                                      \
  template <typename T, typename... Args> class has_##fname {                  \
    template <typename C, typename = decltype(std::declval<C>().fname(         \
                              std::declval<Args>()...))>                       \
    static std::true_type test(C *);                                           \
    template <typename> static std::false_type test(...);                      \
                                                                               \
  public:                                                                      \
    static constexpr bool value = decltype(test<T>(0))::value;                 \
  };

#define HAS_METHOD_WITH_RETURN_TYPE(fname)                                     \
  template <typename T, typename ReturnType, typename... Args>                 \
  class has_##fname##_with_return_type {                                       \
    template <typename C,                                                      \
              typename ActualReturnType = decltype(                            \
                  std::declval<const C>().fname(std::declval<Args>()...))>     \
    static typename std::is_same<ActualReturnType, ReturnType>::type           \
    test(C *);                                                                 \
    template <typename> static std::false_type test(...);                      \
                                                                               \
  public:                                                                      \
    static constexpr bool value = decltype(test<T>(0))::value;                 \
  };

/*
 * This set of macros creates a trait which can check for the existence
 * of any method with some name `fname` (including private methods).
 * This is done by hijacking name hiding, Namely if a derived class overloads a
 * method the base methods will be hidden.  So by starting with a base class
 * with a known method then extending that class you can determine if the
 * derived class included any other methods with that name.
 * https://stackoverflow.com/questions/1628768/why-does-an-overridden-function-in-the-derived-class-hide-other-overloads-of-the
 */

namespace detail {
struct DummyType {};
} // namespace detail

/*
 * Creates a base class with a public method with name `fname` this is
 * included via inheritance to check for name hiding.
 */
#define BASE_WITH_PUBLIC_METHOD(fname)                                         \
  namespace detail {                                                           \
  struct BaseWithPublic##fname {                                               \
    DummyType fname() const { return DummyType(); }                            \
  };                                                                           \
  }

/*
 * Creates a templated class which inherits from a given class as well
 * as the Base class above.  If U contains a method with name `fname` then
 * the Base class definition of that function will be hidden.
 */
#define MULTI_INHERIT(fname)                                                   \
  namespace detail {                                                           \
  template <typename U>                                                        \
  struct MultiInherit##fname : public U, public BaseWithPublic##fname {};      \
  }

/*
 * Creates a trait which checks to see if the dummy implementation in
 * the Base class exists or not, used to determine if name hiding is
 * active.
 */
#define HAS_DUMMY_DEFINITION(fname)                                            \
  namespace detail {                                                           \
  template <typename T> class has_dummy_definition_##fname {                   \
    template <typename C>                                                      \
    static typename std::is_same<decltype(std::declval<const C>().fname()),    \
                                 DummyType>::type                              \
    test(C *);                                                                 \
    template <typename> static std::false_type test(...);                      \
                                                                               \
  public:                                                                      \
    static constexpr bool value = decltype(test<T>(0))::value;                 \
  };                                                                           \
  }

/*
 * This creates the final trait which will check if any method named
 * `fname` exists in type U.
 */
#define HAS_ANY_DEFINITION(fname)                                              \
  template <typename U> class has_any_##fname {                                \
    template <typename T>                                                      \
    static typename std::enable_if<detail::has_dummy_definition_##fname<       \
                                       detail::MultiInherit##fname<T>>::value, \
                                   std::false_type>::type                      \
    test(int);                                                                 \
    template <typename T> static std::true_type test(...);                     \
                                                                               \
  public:                                                                      \
    static constexpr bool value = decltype(test<U>(0))::value;                 \
  }

#define MAKE_HAS_ANY_TRAIT(fname)                                              \
  BASE_WITH_PUBLIC_METHOD(fname);                                              \
  MULTI_INHERIT(fname);                                                        \
  HAS_DUMMY_DEFINITION(fname);                                                 \
  HAS_ANY_DEFINITION(fname);

} // namespace albatross

#endif /* INCLUDE_ALBATROSS_SRC_DETAILS_HAS_ANY_MACROS_HPP_ */
