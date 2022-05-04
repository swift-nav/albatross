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

#ifndef INCLUDE_ALBATROSS_SRC_DETAILS_ERROR_HANDLING_HPP_
#define INCLUDE_ALBATROSS_SRC_DETAILS_ERROR_HANDLING_HPP_

/*
 * The fact that assert() behaves differently in debug and release mode can
 * cause a number of headaches.  For example, if you do something like:
 *
 *   assert(function_with_side_effects());
 *
 * that function call will have side effects during debug, but not release.
 * Alternatively you may want to instead do:
 *
 *   const bool success = function_with_side_effects();
 *   assert(success);
 *
 * but that will results in compiler complaints about unused variables.
 *
 * Here we use a workaround proposed here:
 *
 * https://web.archive.org/web/20201129200055/http://cnicholson.net/2009/02/stupid-c-tricks-adventures-in-assert/
 *
 * in which we define a separate macro which always evaluates (in even release
 * mode) and avoids the unused variable problem if you want to go that route.
 */
#ifdef NDEBUG
#define ALBATROSS_ASSERT(x)                                                    \
  do {                                                                         \
    (void)(x);                                                                 \
  } while (0)
#else
#include <assert.h>
#define ALBATROSS_ASSERT(x) assert((x))
#endif

namespace albatross {

#define ALBATROSS_FAIL(dummy, msg)                                             \
  { static_assert(delay_static_assert<dummy>::value, msg); }

/*
 * Setting ALBATROSS_FAIL to "= delete" as below will slightly
 * change the behavior of failures.  In some situations
 * inspection of return types can trigger the delay_static_assert
 * approach above, while the deleted function approach may
 * work fine.  In general however the deleted function approach
 * leads to slightly more confusing compile errors since it
 * isn't possible to include an error message.
 */

//#define ALBATROSS_FAIL(dummy, msg) = delete

} // namespace albatross

#endif /* INCLUDE_ALBATROSS_SRC_DETAILS_ERROR_HANDLING_HPP_ */
