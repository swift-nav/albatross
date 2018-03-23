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

#ifndef ALBATROSS_CORE_MAGIC_H
#define ALBATROSS_CORE_MAGIC_H

namespace albatross {

/*
 * This struct determines whether or not a class has a method defined for,
 *   `operator() (X x, Y y, Z z, ...)`
 * The result of the inspection gets stored in the member `value`.
 */
template <typename T, typename... Args>
class has_call_operator
{
    template <typename C,
              typename = decltype( std::declval<C>()(std::declval<Args>()...))>
    static std::true_type test(int);
    template <typename C>
    static std::false_type test(...);

public:
    static constexpr bool value = decltype(test<T>(0))::value;
};

}

#endif
