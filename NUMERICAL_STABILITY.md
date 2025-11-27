# Numerical Stability in Mixed-Precision Mode

## Operations That Always Use Double Precision

The following operations **always use `double` precision**, regardless of the `ALBATROSS_USE_FLOAT_PRECISION` macro:

### ✅ Protected (Always Double):

1. **Matrix Decompositions**
   - LDLT: `SerializableLDLT` uses `double` explicitly
   - All decomposition storage uses `MatrixXd`

2. **Linear System Solves**
   - `train_covariance.solve()` operates on `MatrixXd`
   - All solve operations use double precision

3. **Variance/Covariance Storage**
   - GP fit information: `Eigen::VectorXd`
   - Covariance matrices: `Eigen::MatrixXd`
   - Variance computations: `Eigen::VectorXd`

4. **Critical GP Operations**
   - Prediction means: `Eigen::VectorXd`
   - Prediction variances: `Eigen::VectorXd`
   - Cross-covariances in predictions: `Eigen::MatrixXd`

5. **Hyperparameters**
   - `Parameter` type always uses `double`
   - Length scales, sigmas: always double precision

## What BuildTimeScalar Affects (When Macro Is Enabled)

Currently, `BuildTimeScalar` is **infrastructure-only** and doesn't affect actual computations. It's prepared for future use in:

- Covariance function evaluation (e.g., `exp(-distance²/length_scale²)`)
- Distance computations
- Intermediate temporary calculations

## Verification

All core GP operations verified to use double:

```bash
# Verify LDLT uses double
grep "using.*Scalar" include/albatross/src/eigen/serializable_ldlt.hpp
# Output: using RealScalar = double; using Scalar = double;

# Verify GP uses double
grep "Eigen::VectorXd\|Eigen::MatrixXd" include/albatross/src/models/gp.hpp
# Output: All GP operations use MatrixXd/VectorXd
```

## Testing

Tests pass with both configurations:

```bash
# Default (double precision)
bazel test //:albatross-test-core //:albatross-test-models
# ✅ PASSED

# Mixed-precision mode
bazel test --config=mixed-precision //:albatross-test-core //:albatross-test-models
# ✅ PASSED
```

## Conclusion

**The mixed-precision macro is currently safe** because:

1. All numerically-sensitive operations explicitly use `double`
2. The macro only affects type definitions, not actual code paths
3. Tests verify correctness in both modes
4. Infrastructure is ready for future float32 optimizations

When float32 is added to covariance functions in the future, critical operations will remain protected by their explicit `double` types.
