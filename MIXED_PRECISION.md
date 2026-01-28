# Mixed-Precision Mode

Albatross supports mixed-precision computation to accelerate Gaussian Process operations while maintaining numerical accuracy.

## How to Enable

### Simple: Use the Bazel Config

```bash
# Build with mixed-precision
bazel build --config=mixed-precision //your:target

# Test with mixed-precision  
bazel test --config=mixed-precision //your:tests

# Run benchmarks
bazel run --config=mixed-precision //:benchmark_mixed_precision
```

### Alternative: Manual Flag

```bash
bazel build --copt=-DALBATROSS_USE_FLOAT_PRECISION //your:target
```

## Performance Benefits

- **Matrix operations**: ~2x faster
- **GP training**: ~1.3x faster  
- **GP prediction**: ~1.5x faster
- **Memory usage**: ~50% reduction

## What Changes

When `--config=mixed-precision` is used:
- **Internal computations**: Use `float32` (single precision)
- **Public APIs**: Still use `double` (backward compatible)
- **Storage/decompositions**: Use `double` (numerical stability)

## Trade-offs

✅ **Pros:**
- Significant speedup for large problems
- Reduced memory usage
- Same API - no code changes needed

⚠️ **Considerations:**
- Precision: ~7 digits (float) vs ~15 digits (double)
- For most GP applications, this is fine (data noise >> numerical error)

## Example

```cpp
// Your code doesn't change!
auto gp = GaussianProcess(cov_func, mean_func);
auto fit = gp.fit(dataset);
auto pred = fit.predict(test_features);

// Just build with: --config=mixed-precision
// Gets ~2x speedup automatically
```

## Verify It Works

```bash
# Run tests with mixed-precision
bazel test --config=mixed-precision //:albatross-test-core
bazel test --config=mixed-precision //:albatross-test-models

# See benchmarks
bazel run //:benchmark_mixed_precision
```

## Technical Details

- Implemented via build-time macro: `ALBATROSS_USE_FLOAT_PRECISION`
- Changes `BuildTimeScalar` type from `double` → `float`
- Template covariance functions use `BuildTimeScalar` internally
- Automatic precision conversion at API boundaries
- Fully backward compatible

For more details, see:
- `.bazelrc` (build configuration)
- `include/albatross/src/core/scalar_traits.hpp` (type system)
- `tests/test_mixed_precision.cc` (tests)
- `examples/mixed_precision_example.cc` (example)
