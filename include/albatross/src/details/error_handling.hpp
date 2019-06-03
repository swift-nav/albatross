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

namespace albatross {

#define ALBATROSS_FAIL(dummy, msg)                                             \
  { static_assert(delay_static_assert<dummy>::value, msg); }

//#define ALBATROSS_FAIL(dummy, msg) = delete

} // namespace albatross

#endif /* INCLUDE_ALBATROSS_SRC_DETAILS_ERROR_HANDLING_HPP_ */
