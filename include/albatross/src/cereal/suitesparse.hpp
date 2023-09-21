/*
 * Copyright (C) 2023 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef ALBATROSS_CEREAL_SUITESPARSE_H
#define ALBATROSS_CEREAL_SUITESPARSE_H

namespace cereal {

template <class Archive>
inline void serialize(Archive &ar, cholmod_common::cholmod_method_struct &cms,
                      std::uint32_t version ALBATROSS_UNUSED) {
  ar(CEREAL_NVP(cms.lnz));
  ar(CEREAL_NVP(cms.fl));
  ar(CEREAL_NVP(cms.prune_dense));
  ar(CEREAL_NVP(cms.prune_dense2));
  ar(CEREAL_NVP(cms.nd_oksep));
  static_assert(sizeof(cms.other_1) == 4 * sizeof(double),
                "cholmod_common::cholmod_method_struct::other_1 expected to "
                "have 4 elements.");
  ar(::cereal::make_nvp("cms.other_1_0", cms.other_1[0]));
  ar(::cereal::make_nvp("cms.other_1_1", cms.other_1[1]));
  ar(::cereal::make_nvp("cms.other_1_2", cms.other_1[2]));
  ar(::cereal::make_nvp("cms.other_1_3", cms.other_1[3]));
  ar(CEREAL_NVP(cms.nd_small));
  static_assert(sizeof(cms.other_2) == 4 * sizeof(std::size_t),
                "cholmod_common::cholmod_method_struct::other_2 expected to "
                "have 4 elements.");
  ar(::cereal::make_nvp("cms.other_2_0", cms.other_2[0]));
  ar(::cereal::make_nvp("cms.other_2_1", cms.other_2[1]));
  ar(::cereal::make_nvp("cms.other_2_2", cms.other_2[2]));
  ar(::cereal::make_nvp("cms.other_2_3", cms.other_2[3]));
  ar(CEREAL_NVP(cms.aggressive));
  ar(CEREAL_NVP(cms.order_for_lu));
  ar(CEREAL_NVP(cms.nd_compress));
  ar(CEREAL_NVP(cms.nd_camd));
  ar(CEREAL_NVP(cms.nd_components));
  ar(CEREAL_NVP(cms.ordering));
  static_assert(sizeof(cms.other_3) == 4 * sizeof(std::size_t),
                "cholmod_common::cholmod_method_struct::other_3 expected to "
                "have 4 elements.");
  ar(::cereal::make_nvp("cms.other_3_0", cms.other_3[0]));
  ar(::cereal::make_nvp("cms.other_3_1", cms.other_3[1]));
  ar(::cereal::make_nvp("cms.other_3_2", cms.other_3[2]));
  ar(::cereal::make_nvp("cms.other_3_3", cms.other_3[3]));
}

namespace detail {

template <class Archive>
inline void encode_array(Archive &ar, const char *name, void *source,
                         std::size_t size_bytes) {
  std::string payload =
      gzip::compress(reinterpret_cast<const char *>(source), size_bytes);
  if (::cereal::traits::is_text_archive<Archive>::value) {
    payload =
        base64::encode(reinterpret_cast<const unsigned char *>(payload.data()),
                       payload.size());
  }
  ar(::cereal::make_nvp(name, payload));
}

template <class Archive>
inline void decode_array(Archive &ar, const char *name, void *destination,
                         std::size_t expected_bytes) {
  ALBATROSS_ASSERT(nullptr != destination);
  std::string payload;
  ar(::cereal::make_nvp(name, payload));
  if (::cereal::traits::is_text_archive<Archive>::value) {
    payload = base64::decode(payload);
  }
  const std::string decompressed =
      gzip::decompress(payload.data(), payload.size());
  ALBATROSS_ASSERT(decompressed.size() == expected_bytes);
  std::memcpy(destination, decompressed.data(), decompressed.size());
}

// This exists because 1) the suitesparse memory functions require a
// cholmod_common pointer for statistics 2) the suitesparse realloc
// helpers have difficult state properties.
inline void suitesparse_cereal_realloc(void **p, std::size_t count,
                                       std::size_t elem_size) {
  if (nullptr != *p) {
    free(*p);
  }
  *p = malloc(count * elem_size);
  ALBATROSS_ASSERT(nullptr != *p);
}

} // namespace detail

template <class Archive>
inline void save(Archive &ar, cholmod_common const &cc,
                 std::uint32_t version ALBATROSS_UNUSED) {
  ar(CEREAL_NVP(cc.dbound));
  ar(CEREAL_NVP(cc.grow0));
  ar(CEREAL_NVP(cc.grow1));
  ar(CEREAL_NVP(cc.grow2));
  ar(CEREAL_NVP(cc.maxrank));
  ar(CEREAL_NVP(cc.supernodal_switch));
  ar(CEREAL_NVP(cc.supernodal));
  ar(CEREAL_NVP(cc.final_asis));
  ar(CEREAL_NVP(cc.final_super));
  ar(CEREAL_NVP(cc.final_ll));
  ar(CEREAL_NVP(cc.final_pack));
  ar(CEREAL_NVP(cc.final_monotonic));
  ar(CEREAL_NVP(cc.final_resymbol));
  static_assert(sizeof(cc.zrelax) == 3 * sizeof(double),
                "cholmod_common::zrelax expected to have 3 elements.");
  ar(::cereal::make_nvp("cc.zrelax0", cc.zrelax[0]));
  ar(::cereal::make_nvp("cc.zrelax1", cc.zrelax[1]));
  ar(::cereal::make_nvp("cc.zrelax2", cc.zrelax[2]));
  static_assert(sizeof(cc.nrelax) == 3 * sizeof(std::size_t),
                "cholmod_common::nrelax expected to have 3 elements.");
  ar(::cereal::make_nvp("cc.nrelax0", cc.nrelax[0]));
  ar(::cereal::make_nvp("cc.nrelax1", cc.nrelax[1]));
  ar(::cereal::make_nvp("cc.nrelax2", cc.nrelax[2]));
  ar(CEREAL_NVP(cc.prefer_zomplex));
  ar(CEREAL_NVP(cc.prefer_upper));
  ar(CEREAL_NVP(cc.quick_return_if_not_posdef));
  ar(CEREAL_NVP(cc.prefer_binary));
  ar(CEREAL_NVP(cc.print));
  ar(CEREAL_NVP(cc.precise));
  ar(CEREAL_NVP(cc.try_catch));
  // Do not serialise the user error handler.
  ar(CEREAL_NVP(cc.nmethods));
  ar(CEREAL_NVP(cc.current));
  ar(CEREAL_NVP(cc.selected));
  constexpr std::size_t max_methods = sizeof(cc.method) / sizeof(cc.method[0]);
  for (std::size_t m = 0; m < max_methods; ++m) {
    ar(cc.method[m]);
  }
  // method
  ar(CEREAL_NVP(cc.postorder));
  ar(CEREAL_NVP(cc.default_nesdis));
  ar(CEREAL_NVP(cc.metis_memory));
  ar(CEREAL_NVP(cc.metis_dswitch));
  ar(CEREAL_NVP(cc.metis_nswitch));
  // This and the "worksize" variables are serialised under different
  // names so that we can deserialise them to stack variables and
  // calculate things with them in `load()`.
  ar(::cereal::make_nvp("nrow", cc.nrow));
  ar(CEREAL_NVP(cc.mark));
  ar(::cereal::make_nvp("iworksize", cc.iworksize));
  // The comments in `cholmod_core.h` define this as a size in bytes.
  // The code, however, considers it to be a number of elements.
  ar(::cereal::make_nvp("xworksize", cc.xworksize));

  const std::size_t cholmod_int_size =
      cc.itype == CHOLMOD_LONG ? sizeof(SuiteSparse_long) : sizeof(int);

  detail::encode_array(ar, "cc.Flag", cc.Flag, cc.nrow * cholmod_int_size);
  detail::encode_array(ar, "cc.Head", cc.Head,
                       (cc.nrow + 1) * cholmod_int_size);
  detail::encode_array(ar, "cc.Xwork", cc.Xwork, cc.xworksize * sizeof(double));

  // It is safe to discard the CHOLMOD workspace in between calls to
  // the solver, so rather than serialize these temporary arrays, we
  // just leave them out and reinitialize them properly when we
  // deserialize the object (see below how `load()` fills in
  // `cc.Iwork`).
  ar(CEREAL_NVP(cc.itype));
  ar(CEREAL_NVP(cc.dtype));
  ar(CEREAL_NVP(cc.no_workspace_reallocate));
  ar(CEREAL_NVP(cc.status));
  ar(CEREAL_NVP(cc.fl));
  ar(CEREAL_NVP(cc.lnz));
  ar(CEREAL_NVP(cc.anz));
  ar(CEREAL_NVP(cc.modfl));
  ar(CEREAL_NVP(cc.malloc_count));
  ar(CEREAL_NVP(cc.memory_usage));
  ar(CEREAL_NVP(cc.memory_inuse));
  ar(CEREAL_NVP(cc.nrealloc_col));
  ar(CEREAL_NVP(cc.nrealloc_factor));
  ar(CEREAL_NVP(cc.ndbounds_hit));
  ar(CEREAL_NVP(cc.rowfacfl));
  ar(CEREAL_NVP(cc.aatfl));
  ar(CEREAL_NVP(cc.called_nd));
  ar(CEREAL_NVP(cc.blas_ok));
  ar(CEREAL_NVP(cc.SPQR_grain));
  ar(CEREAL_NVP(cc.SPQR_small));
  ar(CEREAL_NVP(cc.SPQR_shrink));
  ar(CEREAL_NVP(cc.SPQR_nthreads));
  ar(CEREAL_NVP(cc.SPQR_flopcount));
  ar(CEREAL_NVP(cc.SPQR_analyze_time));
  ar(CEREAL_NVP(cc.SPQR_factorize_time));
  ar(CEREAL_NVP(cc.SPQR_solve_time));
  ar(CEREAL_NVP(cc.SPQR_flopcount_bound));
  ar(CEREAL_NVP(cc.SPQR_tol_used));
  ar(CEREAL_NVP(cc.SPQR_norm_E_fro));
  static_assert(sizeof(cc.SPQR_istat) == 10 * sizeof(SuiteSparse_long),
                "cholmod_common.SPQR_istat expected to have 10 elements.");
  ar(::cereal::make_nvp("cc.SPQR_istat0", cc.SPQR_istat[0]));
  ar(::cereal::make_nvp("cc.SPQR_istat1", cc.SPQR_istat[1]));
  ar(::cereal::make_nvp("cc.SPQR_istat2", cc.SPQR_istat[2]));
  ar(::cereal::make_nvp("cc.SPQR_istat3", cc.SPQR_istat[3]));
  ar(::cereal::make_nvp("cc.SPQR_istat4", cc.SPQR_istat[4]));
  ar(::cereal::make_nvp("cc.SPQR_istat5", cc.SPQR_istat[5]));
  ar(::cereal::make_nvp("cc.SPQR_istat6", cc.SPQR_istat[6]));
  ar(::cereal::make_nvp("cc.SPQR_istat7", cc.SPQR_istat[7]));
  ar(::cereal::make_nvp("cc.SPQR_istat8", cc.SPQR_istat[8]));
  ar(::cereal::make_nvp("cc.SPQR_istat9", cc.SPQR_istat[9]));

  // Completely ignore all the GPU stuff.
#ifdef GPU_BLAS
  static_assert(false,
                "This codec will not work for CHOLMOD / SPQR using GPU!");
#endif // GPU_BLAS
  ALBATROSS_ASSERT(cc.useGPU == 0);
  ar(CEREAL_NVP(cc.useGPU));

  ar(CEREAL_NVP(cc.syrkStart));
  ar(CEREAL_NVP(cc.cholmod_cpu_gemm_time));
  ar(CEREAL_NVP(cc.cholmod_cpu_syrk_time));
  ar(CEREAL_NVP(cc.cholmod_cpu_trsm_time));
  ar(CEREAL_NVP(cc.cholmod_cpu_potrf_time));
  ar(CEREAL_NVP(cc.cholmod_assemble_time));
  ar(CEREAL_NVP(cc.cholmod_assemble_time2));
  ar(CEREAL_NVP(cc.cholmod_cpu_gemm_calls));
  ar(CEREAL_NVP(cc.cholmod_cpu_syrk_calls));
  ar(CEREAL_NVP(cc.cholmod_cpu_trsm_calls));
  ar(CEREAL_NVP(cc.cholmod_cpu_potrf_calls));
}

template <class Archive>
inline void load(Archive &ar, cholmod_common &cc,
                 std::uint32_t version ALBATROSS_UNUSED) {
  ar(CEREAL_NVP(cc.dbound));
  ar(CEREAL_NVP(cc.grow0));
  ar(CEREAL_NVP(cc.grow1));
  ar(CEREAL_NVP(cc.grow2));
  ar(CEREAL_NVP(cc.maxrank));
  ar(CEREAL_NVP(cc.supernodal_switch));
  ar(CEREAL_NVP(cc.supernodal));
  ar(CEREAL_NVP(cc.final_asis));
  ar(CEREAL_NVP(cc.final_super));
  ar(CEREAL_NVP(cc.final_ll));
  ar(CEREAL_NVP(cc.final_pack));
  ar(CEREAL_NVP(cc.final_monotonic));
  ar(CEREAL_NVP(cc.final_resymbol));
  static_assert(sizeof(cc.zrelax) == 3 * sizeof(double),
                "cholmod_common::zrelax expected to have 3 elements.");
  ar(::cereal::make_nvp("cc.zrelax0", cc.zrelax[0]));
  ar(::cereal::make_nvp("cc.zrelax1", cc.zrelax[1]));
  ar(::cereal::make_nvp("cc.zrelax2", cc.zrelax[2]));
  static_assert(sizeof(cc.nrelax) == 3 * sizeof(std::size_t),
                "cholmod_common::nrelax expected to have 3 elements.");
  ar(::cereal::make_nvp("cc.nrelax0", cc.nrelax[0]));
  ar(::cereal::make_nvp("cc.nrelax1", cc.nrelax[1]));
  ar(::cereal::make_nvp("cc.nrelax2", cc.nrelax[2]));
  ar(CEREAL_NVP(cc.prefer_zomplex));
  ar(CEREAL_NVP(cc.prefer_upper));
  ar(CEREAL_NVP(cc.quick_return_if_not_posdef));
  ar(CEREAL_NVP(cc.prefer_binary));
  ar(CEREAL_NVP(cc.print));
  ar(CEREAL_NVP(cc.precise));
  ar(CEREAL_NVP(cc.try_catch));
  // Do not serialise the user error handler.
  ar(CEREAL_NVP(cc.nmethods));
  ar(CEREAL_NVP(cc.current));
  ar(CEREAL_NVP(cc.selected));
  constexpr std::size_t max_methods = sizeof(cc.method) / sizeof(cc.method[0]);
  for (std::size_t m = 0; m < max_methods; ++m) {
    ar(cc.method[m]);
  }
  // method
  ar(CEREAL_NVP(cc.postorder));
  ar(CEREAL_NVP(cc.default_nesdis));
  ar(CEREAL_NVP(cc.metis_memory));
  ar(CEREAL_NVP(cc.metis_dswitch));
  ar(CEREAL_NVP(cc.metis_nswitch));
  std::size_t nrow = 0;
  std::size_t iworksize = 0;
  std::size_t xworksize = 0;
  ar(CEREAL_NVP(nrow));
  ar(CEREAL_NVP(cc.mark));
  ar(CEREAL_NVP(iworksize));
  ar(CEREAL_NVP(xworksize));
  const std::size_t cholmod_int_size =
      cc.itype == CHOLMOD_LONG ? sizeof(SuiteSparse_long) : sizeof(int);
  ALBATROSS_ASSERT(cholmod_l_free_work(&cc) == 1);
  ALBATROSS_ASSERT(cholmod_l_allocate_work(nrow, iworksize, xworksize, &cc) ==
                   1);
  detail::decode_array(ar, "cc.Flag", cc.Flag, cc.nrow * cholmod_int_size);
  detail::decode_array(ar, "cc.Head", cc.Head,
                       (cc.nrow + 1) * cholmod_int_size);
  detail::decode_array(ar, "cc.Xwork", cc.Xwork, cc.xworksize * sizeof(double));

  cc.Iwork = cholmod_l_malloc(cc.iworksize, cholmod_int_size, &cc);
  ar(CEREAL_NVP(cc.itype));
  ar(CEREAL_NVP(cc.dtype));
  ar(CEREAL_NVP(cc.no_workspace_reallocate));
  ar(CEREAL_NVP(cc.status));
  ar(CEREAL_NVP(cc.fl));
  ar(CEREAL_NVP(cc.lnz));
  ar(CEREAL_NVP(cc.anz));
  ar(CEREAL_NVP(cc.modfl));
  ar(CEREAL_NVP(cc.malloc_count));
  ar(CEREAL_NVP(cc.memory_usage));
  ar(CEREAL_NVP(cc.memory_inuse));
  ar(CEREAL_NVP(cc.nrealloc_col));
  ar(CEREAL_NVP(cc.nrealloc_factor));
  ar(CEREAL_NVP(cc.ndbounds_hit));
  ar(CEREAL_NVP(cc.rowfacfl));
  ar(CEREAL_NVP(cc.aatfl));
  ar(CEREAL_NVP(cc.called_nd));
  ar(CEREAL_NVP(cc.blas_ok));
  ar(CEREAL_NVP(cc.SPQR_grain));
  ar(CEREAL_NVP(cc.SPQR_small));
  ar(CEREAL_NVP(cc.SPQR_shrink));
  ar(CEREAL_NVP(cc.SPQR_nthreads));
  ar(CEREAL_NVP(cc.SPQR_flopcount));
  ar(CEREAL_NVP(cc.SPQR_analyze_time));
  ar(CEREAL_NVP(cc.SPQR_factorize_time));
  ar(CEREAL_NVP(cc.SPQR_solve_time));
  ar(CEREAL_NVP(cc.SPQR_flopcount_bound));
  ar(CEREAL_NVP(cc.SPQR_tol_used));
  ar(CEREAL_NVP(cc.SPQR_norm_E_fro));
  static_assert(sizeof(cc.SPQR_istat) == 10 * sizeof(SuiteSparse_long),
                "cholmod_common.SPQR_istat expected to have 10 elements.");
  ar(::cereal::make_nvp("cc.SPQR_istat0", cc.SPQR_istat[0]));
  ar(::cereal::make_nvp("cc.SPQR_istat1", cc.SPQR_istat[1]));
  ar(::cereal::make_nvp("cc.SPQR_istat2", cc.SPQR_istat[2]));
  ar(::cereal::make_nvp("cc.SPQR_istat3", cc.SPQR_istat[3]));
  ar(::cereal::make_nvp("cc.SPQR_istat4", cc.SPQR_istat[4]));
  ar(::cereal::make_nvp("cc.SPQR_istat5", cc.SPQR_istat[5]));
  ar(::cereal::make_nvp("cc.SPQR_istat6", cc.SPQR_istat[6]));
  ar(::cereal::make_nvp("cc.SPQR_istat7", cc.SPQR_istat[7]));
  ar(::cereal::make_nvp("cc.SPQR_istat8", cc.SPQR_istat[8]));
  ar(::cereal::make_nvp("cc.SPQR_istat9", cc.SPQR_istat[9]));

  // Completely ignore all the GPU stuff.
#ifdef GPU_BLAS
  static_assert(false,
                "This codec will not work for CHOLMOD / SPQR using GPU!");
#endif // GPU_BLAS
  ar(CEREAL_NVP(cc.useGPU));
  ALBATROSS_ASSERT(cc.useGPU == 0);

  ar(CEREAL_NVP(cc.syrkStart));
  ar(CEREAL_NVP(cc.cholmod_cpu_gemm_time));
  ar(CEREAL_NVP(cc.cholmod_cpu_syrk_time));
  ar(CEREAL_NVP(cc.cholmod_cpu_trsm_time));
  ar(CEREAL_NVP(cc.cholmod_cpu_potrf_time));
  ar(CEREAL_NVP(cc.cholmod_assemble_time));
  ar(CEREAL_NVP(cc.cholmod_assemble_time2));
  ar(CEREAL_NVP(cc.cholmod_cpu_gemm_calls));
  ar(CEREAL_NVP(cc.cholmod_cpu_syrk_calls));
  ar(CEREAL_NVP(cc.cholmod_cpu_trsm_calls));
  ar(CEREAL_NVP(cc.cholmod_cpu_potrf_calls));
}

template <class Matrix>
inline std::size_t get_element_size_bytes(const Matrix &m) {
  return m.dtype == CHOLMOD_DOUBLE ? sizeof(double) : sizeof(float);
}

template <class Matrix> inline std::size_t get_num_x_elements(const Matrix &m) {
  // Complex matrices store twice as many elements.
  // https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/Core/cholmod_sparse.c#L209
  return m.nzmax * (m.xtype == CHOLMOD_COMPLEX ? 2 : 1);
}

template <class Matrix>
inline std::size_t get_p_element_size_bytes(const Matrix &m) {
  assert((m.itype == CHOLMOD_INT || m.itype == CHOLMOD_LONG) &&
         "we only support int and long indices");
  return m.itype == CHOLMOD_INT ? sizeof(int) : sizeof(SuiteSparse_long);
}

template <class Matrix>
inline std::size_t get_integer_size_bytes(const Matrix &m) {
  assert((m.itype == CHOLMOD_INT || m.itype == CHOLMOD_LONG) &&
         "we only support int and long indices");
  return m.itype == CHOLMOD_LONG ? sizeof(SuiteSparse_long) : sizeof(int);
}

template <class Archive>
inline void save(Archive &ar, cholmod_sparse const &m,
                 std::uint32_t version ALBATROSS_UNUSED) {
  assert((m.itype == CHOLMOD_INT || m.itype == CHOLMOD_LONG) &&
         "we only support int and long indices");
  ar(CEREAL_NVP(m.nrow));
  ar(CEREAL_NVP(m.ncol));
  ar(CEREAL_NVP(m.nzmax));
  ar(CEREAL_NVP(m.stype));
  ar(CEREAL_NVP(m.itype));
  ar(CEREAL_NVP(m.xtype));
  ar(CEREAL_NVP(m.dtype));
  ar(CEREAL_NVP(m.sorted));
  ar(CEREAL_NVP(m.packed));

  const std::size_t element_size_bytes = get_element_size_bytes(m);
  const std::size_t x_elements = get_num_x_elements(m);
  const std::size_t p_size_bytes = get_p_element_size_bytes(m);
  const std::size_t integer_size_bytes = get_integer_size_bytes(m);

  detail::encode_array(ar, "m.p", m.p, (m.ncol + 1) * p_size_bytes);
  detail::encode_array(ar, "m.i", m.i, m.nzmax * integer_size_bytes);
  if (nullptr != m.nz && !m.packed) {
    detail::encode_array(ar, "m.nz", m.nz, m.ncol * integer_size_bytes);
  }
  if (nullptr != m.x) {
    detail::encode_array(ar, "m.x", m.x, x_elements * element_size_bytes);
  }
  // This should only be non-null for "zomplex" elements
  ALBATROSS_ASSERT(m.z == nullptr);
}

template <class Archive>
inline void load(Archive &ar, cholmod_sparse &m,
                 std::uint32_t version ALBATROSS_UNUSED) {
  ar(CEREAL_NVP(m.nrow));
  ar(CEREAL_NVP(m.ncol));
  ar(CEREAL_NVP(m.nzmax));
  ar(CEREAL_NVP(m.stype));
  ar(CEREAL_NVP(m.itype));
  ar(CEREAL_NVP(m.xtype));
  ar(CEREAL_NVP(m.dtype));
  ar(CEREAL_NVP(m.sorted));
  ar(CEREAL_NVP(m.packed));

  const std::size_t element_size_bytes = get_element_size_bytes(m);
  const std::size_t x_elements = get_num_x_elements(m);
  const std::size_t p_size_bytes = get_p_element_size_bytes(m);
  const std::size_t integer_size_bytes = get_integer_size_bytes(m);

  detail::suitesparse_cereal_realloc(&m.p, m.ncol + 1, p_size_bytes);
  detail::suitesparse_cereal_realloc(&m.i, m.nzmax, integer_size_bytes);
  if (!m.packed) {
    detail::suitesparse_cereal_realloc(&m.nz, m.ncol, integer_size_bytes);
  }
  detail::suitesparse_cereal_realloc(&m.x, m.nzmax, element_size_bytes);

  detail::decode_array(ar, "m.p", m.p, (m.ncol + 1) * p_size_bytes);
  detail::decode_array(ar, "m.i", m.i, m.nzmax * integer_size_bytes);
  if (!m.packed) {
    detail::decode_array(ar, "m.nz", m.nz, m.ncol * integer_size_bytes);
  }
  // TODO(@peddie): in what cases may `x` be absent?
  detail::decode_array(ar, "m.x", m.x, x_elements * element_size_bytes);
  // This should only be non-null for "zomplex" elements
  ALBATROSS_ASSERT(m.z == nullptr);
}

template <class Archive>
inline void save(Archive &ar, cholmod_dense const &m,
                 std::uint32_t version ALBATROSS_UNUSED) {
  ar(CEREAL_NVP(m.nrow));
  ar(CEREAL_NVP(m.ncol));
  ar(CEREAL_NVP(m.nzmax));
  ar(CEREAL_NVP(m.d));
  ar(CEREAL_NVP(m.xtype));
  ar(CEREAL_NVP(m.dtype));
  ALBATROSS_ASSERT(m.d >= m.nrow);

  const std::size_t element_size_bytes = get_element_size_bytes(m);
  const std::size_t x_elements = get_num_x_elements(m);

  if (nullptr != m.x) {
    detail::encode_array(ar, "m.x", m.x, x_elements * element_size_bytes);
  }
  // "zomplex" only
  ALBATROSS_ASSERT(nullptr == m.z);
}

template <class Archive>
inline void load(Archive &ar, cholmod_dense &m,
                 std::uint32_t version ALBATROSS_UNUSED) {
  ar(CEREAL_NVP(m.nrow));
  ar(CEREAL_NVP(m.ncol));
  ar(CEREAL_NVP(m.nzmax));
  ar(CEREAL_NVP(m.d));
  ar(CEREAL_NVP(m.xtype));
  ar(CEREAL_NVP(m.dtype));
  ALBATROSS_ASSERT(m.d >= m.nrow);

  const std::size_t element_size_bytes = get_element_size_bytes(m);
  const std::size_t x_elements = get_num_x_elements(m);

  detail::suitesparse_cereal_realloc(&m.x, x_elements, element_size_bytes);
  detail::decode_array(ar, "m.x", m.x, x_elements * element_size_bytes);
  // "zomplex" only
  ALBATROSS_ASSERT(nullptr == m.z);
}

} // namespace cereal

#endif // ALBATROSS_CEREAL_SUITESPARSE_H
