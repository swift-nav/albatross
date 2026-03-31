/*
 * Copyright (C) 2026 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef ALBATROSS_COVARIANCE_FUNCTIONS_BLOCK_WORKSPACE_HPP_
#define ALBATROSS_COVARIANCE_FUNCTIONS_BLOCK_WORKSPACE_HPP_

namespace albatross {

/*
 * The operation to apply when accumulating block results into an output
 * matrix.  The user's _call_impl block method receives a CovarianceOp
 * and should apply it element-wise:
 *
 *   Assign:   out = value
 *   Add:      out += value
 *   Multiply: out *= value
 */
enum class CovarianceOp { Assign, Add, Multiply };

/*
 * Apply a CovarianceOp element-wise: out op= value.
 */
inline void apply_op(Eigen::Ref<Eigen::ArrayXXd> out,
                     const Eigen::Ref<const Eigen::ArrayXXd> &value,
                     CovarianceOp op) {
  switch (op) {
  case CovarianceOp::Assign:
    out = value;
    break;
  case CovarianceOp::Add:
    out += value;
    break;
  case CovarianceOp::Multiply:
    out *= value;
    break;
  }
}

/*
 * Default block dimensions for the blocked covariance computation.
 */
struct BlockSize {
  Eigen::Index rows{256};
  Eigen::Index cols{128};
};

/*
 * A stack-based workspace for temporary block matrices.
 *
 * Each thread in the pool gets its own thread_local BlockWorkspace.
 * The workspace pre-allocates buffers at the configured maximum block
 * dimensions.  Callers acquire() a Handle to get a correctly-sized
 * Ref into a buffer; the Handle's destructor releases the buffer
 * back to the stack (LIFO order).
 *
 * The workspace auto-grows if the stack depth is exceeded (e.g. for
 * deeply nested covariance compositions).
 */
class BlockWorkspace {
  std::vector<Eigen::ArrayXXd> bufs_;
  std::size_t top_{0};
  Eigen::Index max_r_, max_c_;

public:
  BlockWorkspace(Eigen::Index max_r, Eigen::Index max_c, std::size_t depth)
      : bufs_(depth), max_r_(max_r), max_c_(max_c) {
    for (auto &b : bufs_) {
      b.resize(max_r, max_c);
    }
  }

  struct Handle {
    BlockWorkspace *ws;
    Eigen::Ref<Eigen::ArrayXXd> ref;

    Handle(BlockWorkspace *ws_, Eigen::Ref<Eigen::ArrayXXd> ref_)
        : ws(ws_), ref(ref_) {}
    Handle(const Handle &) = delete;
    Handle &operator=(const Handle &) = delete;
    Handle(Handle &&other) noexcept : ws(other.ws), ref(other.ref) {
      other.ws = nullptr;
    }
    Handle &operator=(Handle &&) = delete;

    ~Handle() {
      if (ws) {
        ALBATROSS_ASSERT(ws->top_ > 0);
        ws->top_--;
      }
    }
  };

  /*
   * Ensure the workspace can hold blocks of at least (r, c).
   * Called by the top-level blocked loop before dispatching into the
   * caller chain. This is the only place resizing should happen;
   * user code inside _call_impl should never trigger a resize.
   */
  void ensure_capacity(Eigen::Index r, Eigen::Index c) {
    if (r <= max_r_ && c <= max_c_) {
      return;
    }
    max_r_ = std::max(max_r_, r);
    max_c_ = std::max(max_c_, c);
    for (auto &b : bufs_) {
      b.resize(max_r_, max_c_);
    }
  }

  Handle acquire(Eigen::Index r, Eigen::Index c) {
    ALBATROSS_ASSERT(r <= max_r_ && c <= max_c_);
    if (top_ == bufs_.size()) {
      bufs_.emplace_back(max_r_, max_c_);
    }
    auto &buf = bufs_[top_++];
    return Handle(this, buf.topLeftCorner(r, c));
  }
};

} // namespace albatross

#endif // ALBATROSS_COVARIANCE_FUNCTIONS_BLOCK_WORKSPACE_HPP_
