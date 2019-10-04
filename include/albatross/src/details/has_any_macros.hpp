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

template <typename First, typename Second> struct TypePair {
  using first_type = First;
  using second_type = Second;
};

/*
 * Defines a couple of traits which help inspect whether a type has a method
 * which is callable whne provided with specific arguments.
 *
 *   template <typename T, typename ... Args>
 *   struct class_method_FNAME_traits {
 *     static constexpr bool is_defined;
 *     typename return_type;
 *   }
 *
 * So for example:
 *
 *   class_method_foo_traits<Bar, int, double>::is_defined;
 *
 * would be a static bool which would tell you if you can call:
 *
 *   Bar.foo(int(1), double(3.14159));
 *
 * similarly:
 *
 *   class_method_foo_traits<Bar, int, double>::return_type
 *
 * would tell you the return type.  Note that if the method is not
 * defined `return_type` will be void, be sure to check `is_defined`
 * if you want to make sure the call is feasible, or use the
 * helper class:
 *
 *   has_foo_with_return_type<Bar, ExpectedReturnType, int, double>::value
 *
 * Similarly there is a helper class if you want to check if the
 * method exists in a (slightly) more concise way:
 *
 *   has_foo<Bar, int, double>::value
 */
#define DEFINE_CLASS_METHOD_TRAITS(fname)                                      \
  template <typename T, typename... Args>                                      \
  class class_method_##fname##_traits {                                        \
    template <typename C,                                                      \
              typename ReturnType =                                            \
                  decltype(std::declval<C>().fname(std::declval<Args>()...))>  \
    static TypePair<std::true_type, ReturnType> test(C *);                     \
    template <typename> static TypePair<std::false_type, void> test(...);      \
                                                                               \
  public:                                                                      \
    static constexpr bool is_defined =                                         \
        decltype(test<T>(0))::first_type::value;                               \
    using return_type = typename decltype(test<T>(0))::second_type;            \
  };                                                                           \
                                                                               \
  template <typename T, typename... Args> struct has_##fname {                 \
    static constexpr bool value =                                              \
        class_method_##fname##_traits<T, Args...>::is_defined;                 \
  };                                                                           \
  template <typename T, typename ReturnType, typename... Args>                 \
  struct has_##fname##_with_return_type {                                      \
    using MethodTraits = class_method_##fname##_traits<T, Args...>;            \
    static constexpr bool value =                                              \
        MethodTraits::is_defined &&                                            \
        std::is_same<ReturnType, typename MethodTraits::return_type>::value;   \
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
