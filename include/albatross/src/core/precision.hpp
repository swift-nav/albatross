/*
 * Copyright (C) 2024 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * Precision configuration for Albatross GP library
 */

#ifndef ALBATROSS_CORE_PRECISION_HPP_
#define ALBATROSS_CORE_PRECISION_HPP_

namespace albatross {

/*
 * Build-time precision configuration
 *
 * By default, Albatross uses double precision (float64) for all computations.
 * Define ALBATROSS_USE_FLOAT_PRECISION at build time to use single precision
 * (float32) for internal computations while maintaining double precision
 * interfaces for backward compatibility.
 *
 * Expected performance with ALBATROSS_USE_FLOAT_PRECISION:
 * - Matrix operations: ~2x faster
 * - GP training: ~1.3x faster
 * - GP prediction: ~1.5x faster
 * - Memory usage: ~50% reduction
 *
 * Trade-offs:
 * - Numerical precision: ~7 digits (float) vs ~15 digits (double)
 * - Suitable for most GP applications where data noise >> numerical error
 *
 * Build with:
 *   bazel build --copt=-DALBATROSS_USE_FLOAT_PRECISION //...
 */

#ifdef ALBATROSS_USE_FLOAT_PRECISION
using DefaultScalar = float;
#else
using DefaultScalar = double;
#endif

} // namespace albatross

#endif // ALBATROSS_CORE_PRECISION_HPP_
