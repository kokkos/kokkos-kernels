/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef _KOKKOSSPGEMMROCSPARSE_HPP
#define _KOKKOSSPGEMMROCSPARSE_HPP

#include <KokkosKernels_config.h>

#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCSPARSE

#include "KokkosKernels_Controls.hpp"
#include "KokkosSparse_Utils_rocsparse.hpp"

namespace KokkosSparse {

namespace Impl {

//=============================================================================
// Overload rocsparse_Xcsrgemm_buffer_size() over scalar types
#define ROCSPARSE_XCSRGEMM_BUFFER_SIZE_SPEC(scalar_type, TOKEN)               \
  inline rocsparse_status rocsparse_Xcsrgemm_buffer_size(                     \
      rocsparse_handle handle, rocsparse_operation trans_A,                   \
      rocsparse_operation trans_B, rocsparse_int m, rocsparse_int n,          \
      rocsparse_int k, const scalar_type *alpha,                              \
      const rocsparse_mat_descr descr_A, rocsparse_int nnz_A,                 \
      const rocsparse_int *csr_row_ptr_A, const rocsparse_int *csr_col_ind_A, \
      const rocsparse_mat_descr descr_B, rocsparse_int nnz_B,                 \
      const rocsparse_int *csr_row_ptr_B, const rocsparse_int *csr_col_ind_B, \
      const scalar_type *beta, const rocsparse_mat_descr descr_D,             \
      rocsparse_int nnz_D, const rocsparse_int *csr_row_ptr_D,                \
      const rocsparse_int *csr_col_ind_D, rocsparse_mat_info info_C,          \
      size_t *buffer_size) {                                                  \
    return rocsparse_##TOKEN##csrgemm_buffer_size(                            \
        handle, trans_A, trans_B, m, n, k, alpha, descr_A, nnz_A,             \
        csr_row_ptr_A, csr_col_ind_A, descr_B, nnz_B, csr_row_ptr_B,          \
        csr_col_ind_B, beta, descr_D, nnz_D, csr_row_ptr_D, csr_col_ind_D,    \
        info_C, buffer_size);                                                 \
  }
ROCSPARSE_XCSRGEMM_BUFFER_SIZE_SPEC(float, s)
ROCSPARSE_XCSRGEMM_BUFFER_SIZE_SPEC(double, d)
ROCSPARSE_XCSRGEMM_BUFFER_SIZE_SPEC(rocsparse_float_complex, c)
ROCSPARSE_XCSRGEMM_BUFFER_SIZE_SPEC(rocsparse_double_complex, z)

//=============================================================================
// Overload rocsparse_Xcsrgemm_numeric() over scalar types
#define ROCSPARSE_XCSRGEMM_NUMERIC_SPEC(scalar_type, TOKEN)                   \
  inline rocsparse_status rocsparse_Xcsrgemm_numeric(                         \
      rocsparse_handle handle, rocsparse_operation trans_A,                   \
      rocsparse_operation trans_B, rocsparse_int m, rocsparse_int n,          \
      rocsparse_int k, const scalar_type *alpha,                              \
      const rocsparse_mat_descr descr_A, rocsparse_int nnz_A,                 \
      const scalar_type *csr_val_A, const rocsparse_int *csr_row_ptr_A,       \
      const rocsparse_int *csr_col_ind_A, const rocsparse_mat_descr descr_B,  \
      rocsparse_int nnz_B, const scalar_type *csr_val_B,                      \
      const rocsparse_int *csr_row_ptr_B, const rocsparse_int *csr_col_ind_B, \
      const scalar_type *beta, const rocsparse_mat_descr descr_D,             \
      rocsparse_int nnz_D, const scalar_type *csr_val_D,                      \
      const rocsparse_int *csr_row_ptr_D, const rocsparse_int *csr_col_ind_D, \
      const rocsparse_mat_descr descr_C, rocsparse_int nnz_C,                 \
      scalar_type *csr_val_C, const rocsparse_int *csr_row_ptr_C,             \
      const rocsparse_int *csr_col_ind_C, const rocsparse_mat_info info_C,    \
      void *buffer) {                                                         \
    return rocsparse_##TOKEN##csrgemm_numeric(                                \
        handle, trans_A, trans_B, m, n, k, alpha, descr_A, nnz_A, csr_val_A,  \
        csr_row_ptr_A, csr_col_ind_A, descr_B, nnz_B, csr_val_B,              \
        csr_row_ptr_B, csr_col_ind_B, beta, descr_D, nnz_D, csr_val_D,        \
        csr_row_ptr_D, csr_col_ind_D, descr_C, nnz_C, csr_val_C,              \
        csr_row_ptr_C, csr_col_ind_C, info_C, buffer);                        \
  }
ROCSPARSE_XCSRGEMM_NUMERIC_SPEC(float, s)
ROCSPARSE_XCSRGEMM_NUMERIC_SPEC(double, d)
ROCSPARSE_XCSRGEMM_NUMERIC_SPEC(rocsparse_float_complex, c)
ROCSPARSE_XCSRGEMM_NUMERIC_SPEC(rocsparse_double_complex, z)

  /*
    Rocsparse has its own datatype for complex numbers.
    The datatype, however, has the exact form of the Kokkos::complex<T> type
    (i.e. struct {T real,imaginary;} comlex_type;), so we can reinterpret_cast 
    between the types. Note that this is not a 100% safe operation since the 
    compiler may not preserve the ordering of the real/imaginary entries.
  */
template<typename T> struct rocsparse_type_conversion 
{
  using Type=T; 
  static Type convert(T val){return val;}
};
template<> struct rocsparse_type_conversion<Kokkos::complex<float>> 
{
  using Type=rocsparse_float_complex;
  static Type convert(Kokkos::complex<float> val){
    return *reinterpret_cast<Type*>(&val);
  }
};
template<> struct rocsparse_type_conversion<Kokkos::complex<double>> 
{
  using Type=rocsparse_double_complex;
  static Type convert(Kokkos::complex<double> val){
    return *reinterpret_cast<Type*>(&val);
  }
};

  /*
    The rocsparse kernels only run for "rocsparse_int" (int) index datatypes.
    Downstream applications may use a variety of signed/unsigned ordinal and 
    size types. To get around this we use a set of macros to copy/convert back
    and forth between datatypes. Naturally this comes at a cost, but the 
    runtime of the conversion kernels is negligible compared to the SpGEMM 
    operation itself.
  */

#define KOKKOSKERNELS_ROCSPARSE_USE_OR_CONVERT_VIEW(TYPE,                     \
    PTR_NAME, VIEW_TYPE, VIEW_NAME)                                           \
  TYPE * PTR_NAME = nullptr;                                                  \
  if(std::is_same<std::remove_const<TYPE>::type,                              \
                  typename VIEW_TYPE::non_const_value_type>::value) {         \
    PTR_NAME = (TYPE*) VIEW_NAME.data();                                      \
  } else {                                                                    \
    if(VIEW_NAME.size() > 0 && (spgemm_handle.PTR_NAME.size() !=              \
        VIEW_NAME.size())) {                                                  \
      auto local_##PTR_NAME = spgemm_handle.PTR_NAME =                        \
        typename InputHandle::rocSparseSpgemmHandleType::                     \
          rocsparse_index_array_t(                                            \
            Kokkos::ViewAllocateWithoutInitializing("local copy " #VIEW_NAME),\
              VIEW_NAME.size());                                              \
      Kokkos::parallel_for("local copy from array " #VIEW_NAME ,              \
          VIEW_NAME.size(),                                                   \
        KOKKOS_LAMBDA(const size_t i){local_##PTR_NAME[i] =                   \
          static_cast<std::remove_const<TYPE>::type>(VIEW_NAME[i]);});        \
      Kokkos::fence();                                                        \
    }                                                                         \
    PTR_NAME = (TYPE*) spgemm_handle.PTR_NAME.data();                         \
  }

#define KOKKOSKERNELS_ROCSPARSE_USE_OR_ALLOC_VIEW(TYPE,                       \
    PTR_NAME, VIEW_TYPE, VIEW_NAME)                                           \
  TYPE * PTR_NAME = nullptr;                                                  \
  if(std::is_same<std::remove_const<TYPE>::type,                              \
                  typename VIEW_TYPE::non_const_value_type>::value) {         \
    PTR_NAME = (TYPE*) VIEW_NAME.data();                                      \
  } else {                                                                    \
    if(VIEW_NAME.size() > 0 && (spgemm_handle.PTR_NAME.size() !=              \
        VIEW_NAME.size()))                                                    \
      spgemm_handle.PTR_NAME =                                                \
        typename InputHandle::rocSparseSpgemmHandleType::                     \
            rocsparse_index_array_t(                                          \
          Kokkos::ViewAllocateWithoutInitializing("local copy " #VIEW_NAME),  \
              VIEW_NAME.size());                                              \
    PTR_NAME = (TYPE*) spgemm_handle.PTR_NAME.data();                         \
  }

#define KOKKOSKERNELS_ROCSPARSE_COPY_VIEW(TYPE,                               \
    PTR_NAME, VIEW_TYPE, VIEW_NAME)                                           \
  if(! std::is_same<std::remove_const<TYPE>::type,                            \
                    typename VIEW_TYPE::non_const_value_type>::value) {       \
    auto local_##VIEW_NAME = spgemm_handle.PTR_NAME;                          \
    Kokkos::parallel_for("local copy to array " #VIEW_NAME , VIEW_NAME.size(),\
      KOKKOS_LAMBDA(const size_t i){VIEW_NAME[i] = local_##VIEW_NAME[i];});   \
    Kokkos::fence();                                                          \
  }

/**
 * @brief Symbolic call for rocsparse SpGEMM
 * 
 * Call is used to count the nnz for C = A*B, and fill C's row map.
 *
 * @tparam InputHandle KernelHandle used to store descriptor data
 * @tparam ScalarType Input scalar type (float, double, etc)
 * @tparam InputRowView Row map view type
 * @tparam InputColumnView Column index view type
 * @tparam OutputRowView Output row map view type
 * @param input_handle SpGEMM handle that will store various rocsparse content
 * @param m Number of rows in A
 * @param n Number of columns in A and rows in B
 * @param k Number of columns in B
 * @param row_map_A Row map for A
 * @param columns_A Column indexes for A
 * @param trans_A Use transpose of A (not supported by rocsparse)
 * @param row_map_B Row map for B
 * @param columns_B Column indexes for B
 * @param trans_B Use transpose of B (not supported by rocsparse)
 * @param row_map_C Output rowmap for C
 */
template <class InputHandle, class ScalarType, class InputRowView, 
          class InputColumnView, class OutputRowView>
void
spgemm_symbolic_rocsparse(InputHandle *       input_handle,
                          const rocsparse_int m,
                          const rocsparse_int n,
                          const rocsparse_int k,
                          InputRowView        row_map_A,
                          InputColumnView     columns_A,
                          const bool          trans_A,
                          InputRowView        row_map_B,
                          InputColumnView     columns_B,
                          const bool          trans_B,
                          OutputRowView       row_map_C)
{

  // rocSPARSE solves C = alpha * A * B + beta * D
  // For our purposes, we only care about C = A * B
  // If Jacobi SpGEMM is enabled, we instead use : 
  //   C = B - omega * D^{-1} * A * B, 
  // where D is a diagonal matrix (no effect on A's sparsity pattern)

  // In the symbolic call we will only count nnz for C, 
  // fill the C row map, and allocate a buffer space.

  using RocsparseScalarType = 
    typename rocsparse_type_conversion<ScalarType>::Type;
  using OrdinalType = rocsparse_int;

  // This handle contains some useful rocsparse components
  if(input_handle->get_rocsparse_spgemm_handle() == NULL)
    input_handle->create_rocsparse_spgemm_handle(trans_A,trans_B);
  auto & spgemm_handle = *input_handle->get_rocsparse_spgemm_handle();

  const bool enable_jacobi = spgemm_handle.enable_jacobi;

  // Initialize scalar multipliers - not used at this point, 
  // but we need beta=0 to ignore matrix D (or 1 if Jacobi is enabled)
  RocsparseScalarType alpha = 1;
  RocsparseScalarType beta  = enable_jacobi ? 1 : 0;

  // Create matrix descriptors
  rocsparse_mat_descr descr_A = spgemm_handle.descr_A;
  rocsparse_mat_descr descr_B = spgemm_handle.descr_B;
  rocsparse_mat_descr descr_C = spgemm_handle.descr_C;
  rocsparse_mat_descr descr_D = spgemm_handle.descr_D;

  // Acquire info object for C
  rocsparse_mat_info & info_C = spgemm_handle.info_C;

  const int nnz_A = columns_A.size();
  const int nnz_B = columns_B.size();
  const int nnz_D = enable_jacobi ? columns_B.size() : 0;

  // row_map_C needs to have been sized externally
  if(row_map_C.size() != static_cast<size_t>(m+1))
    throw std::runtime_error("spgemm_symbolic_rocsparse : " \
        "row_map_C has not been properly allocated. row_map_C.size() = " \
        +std::to_string(row_map_C.size())+", expected "+std::to_string(m+1));

  rocsparse_handle handle = spgemm_handle.handle;

  auto operation_A = trans_A ? rocsparse_operation_transpose : 
                               rocsparse_operation_none;
  auto operation_B = trans_B ? rocsparse_operation_transpose : 
                               rocsparse_operation_none;

  if((spgemm_handle.opA != operation_A) ||
     (spgemm_handle.opB != operation_B))
    throw std::runtime_error("spgemm_symbolic_rocsparse : " \
        "Reusing handle with incorrect transpose state.");

  // Matrix CSR row offset arrays
  KOKKOSKERNELS_ROCSPARSE_USE_OR_CONVERT_VIEW(const OrdinalType, csr_row_ptr_A, 
    InputRowView,  row_map_A)
  KOKKOSKERNELS_ROCSPARSE_USE_OR_CONVERT_VIEW(const OrdinalType, csr_row_ptr_B, 
    InputRowView,  row_map_B)
  KOKKOSKERNELS_ROCSPARSE_USE_OR_ALLOC_VIEW(        OrdinalType, csr_row_ptr_C,   
    OutputRowView, row_map_C)
  const OrdinalType * csr_row_ptr_D = (enable_jacobi) ? csr_row_ptr_B : nullptr;

  // Matrix CSR column index arrays
  KOKKOSKERNELS_ROCSPARSE_USE_OR_CONVERT_VIEW(const OrdinalType, csr_col_ind_A, 
    InputColumnView, columns_A)
  KOKKOSKERNELS_ROCSPARSE_USE_OR_CONVERT_VIEW(const OrdinalType, csr_col_ind_B, 
    InputColumnView, columns_B)
//        OrdinalType * csr_col_ind_C = nullptr;
  const OrdinalType * csr_col_ind_D = (enable_jacobi) ? csr_col_ind_B : nullptr;

  // Make sure device is ready
  Kokkos::fence();

  // Set pointer mode (applies to scalar values only)
  rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host);

  // Make sure buffer is allocated
  {
    // Get the size of the buffer
    size_t buffer_size;
    KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(
      rocsparse_Xcsrgemm_buffer_size(handle,
                                     operation_A,operation_B,
                                     m,n,k,
                                     &alpha,
                                     descr_A,nnz_A,csr_row_ptr_A,csr_col_ind_A,
                                     descr_B,nnz_B,csr_row_ptr_B,csr_col_ind_B,
                                     &beta,
                                     descr_D,nnz_D,csr_row_ptr_D,csr_col_ind_D,
                                     info_C,
                                     &buffer_size));

    // Allocate buffer
    using buffer_t = typename InputHandle::
        rocSparseSpgemmHandleType::buffer_t;
    spgemm_handle.buffer = buffer_t(
          Kokkos::ViewAllocateWithoutInitializing("rocsparse_buffer"),
          buffer_size);
  }

  void* buffer = static_cast<void*>(spgemm_handle.buffer.data());

  // Find number of nonzeros in C
  rocsparse_int nnz_C = 0;
  KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(
    rocsparse_csrgemm_nnz(handle,
                          operation_A,operation_B,
                          m,n,k,
                          descr_A,nnz_A,csr_row_ptr_A,csr_col_ind_A,
                          descr_B,nnz_B,csr_row_ptr_B,csr_col_ind_B,
                          descr_D,nnz_D,csr_row_ptr_D,csr_col_ind_D,
                          descr_C,csr_row_ptr_C,
                          &nnz_C,info_C,
                          buffer));

  KOKKOSKERNELS_ROCSPARSE_COPY_VIEW(OrdinalType, csr_row_ptr_C, 
    OutputRowView, row_map_C);

  // We pass the nnz_C out through the KernelHandle
  input_handle->set_c_nnz(nnz_C);

  // Here is an interesting corner case that testing caught 
  // that is a side-effect of not initializing arrays to zero
  if(nnz_C == 0) Kokkos::deep_copy(row_map_C,0);

  // While we have set nnz_C, and filled row_map_C,
  // we have not filled C column indexes, which is
  // handled in the numeric call. We reset the flag 
  // governing the evaluation of C column indexes here.
  spgemm_handle.C_populated = false;

}

/**
 * @brief Numeric call for rocsparse SpGEMM 
 * 
 * Used to fill values for C = A*B
 * Will also fill column indexes in C if that has not already been done
 *
 * @note The symbolic call must be called first or rocsparse will error out
 *
 * @tparam InputHandle KernelHandle used to store handle information
 * @tparam InputRowView Row map view type for A and B
 * @tparam InputColumnView Column index view type for A and B
 * @tparam InputDataView Data view type for A and B
 * @tparam OutputColumnView Output column index view type for C
 * @tparam OutputRowView Output row map view type for C
 * @tparam OutputDataView Data view type for C
 * @param input_handle SpGEMM handle that will store various rocsparse content
 * @param m Number of rows in A
 * @param n Number of columns in A and rows in B
 * @param k Number of columns in B
 * @param row_map_A Row map for A
 * @param columns_A Column indexes for A
 * @param values_A Values for A
 * @param trans_A Use transpose of A (not supported by rocsparse)
 * @param row_map_B Row map for B
 * @param columns_B Column indexes for B
 * @param values_B Values for B
 * @param trans_B Use transpose of B (not supported by rocsparse)
 * @param row_map_C Input row map for C
 * @param columns_C Output column indexes for C
 * @param values_C Output values for C
 */
template <class InputHandle, class InputRowView, class InputColumnView, 
          class InputDataView, class OutputColumnView, class OutputDataView>
void
spgemm_numeric_rocsparse(InputHandle *       input_handle,
                         const rocsparse_int m,
                         const rocsparse_int n,
                         const rocsparse_int k,
                         InputRowView        row_map_A,
                         InputColumnView     columns_A,
                         InputDataView       values_A,
                         const bool          trans_A,
                         InputRowView        row_map_B,
                         InputColumnView     columns_B,
                         InputDataView       values_B,
                         const bool          trans_B,
                         InputRowView        row_map_C,
                         OutputColumnView    columns_C,
                         OutputDataView      values_C)
{

  // rocSPARSE solves C = alpha * A * B + beta * D
  // For our purposes, we only care about C = A * B

  // In the numeric call we will fill columns_C and values_C using what was 
  // learned in the symbolic call.

  using OrdinalType = rocsparse_int;
  using ScalarType = typename InputDataView::non_const_value_type;
  using RocsparseScalarType = 
    typename rocsparse_type_conversion<ScalarType>::Type;
  
  // Initialize scalar multipliers - not used at this point, 
  // but we need beta=0 to ignore matrix D
  RocsparseScalarType alpha = 1;
  RocsparseScalarType beta  = 0;

  // This handle contains some useful rocsparse components
  auto & spgemm_handle = *input_handle->get_rocsparse_spgemm_handle();

  // Create matrix descriptors
  rocsparse_mat_descr descr_A = spgemm_handle.descr_A;
  rocsparse_mat_descr descr_B = spgemm_handle.descr_B;
  rocsparse_mat_descr descr_C = spgemm_handle.descr_C;
  rocsparse_mat_descr descr_D = spgemm_handle.descr_D;

  // Acquire info object for C
  rocsparse_mat_info & info_C = spgemm_handle.info_C;

  // Get the pre-allocated buffer (allocated in the symbolic call)
  void* buffer = static_cast<void*>(spgemm_handle.buffer.data());
  if(buffer == nullptr)
    throw std::runtime_error("spgemm_numeric_rocsparse : " \
      "Buffer has not been allocated.");

  const int nnz_A = columns_A.size();
  const int nnz_B = columns_B.size();
  const int nnz_C = columns_C.size();
  const int nnz_D = 0;

  // Make sure output allocations are properly sized
  if(input_handle->get_c_nnz() != nnz_C)
    throw std::runtime_error("spgemm_numeric_rocsparse : " \
      "columns_C has not been properly allocated. columns_C.size() = " \
      +std::to_string(columns_C.size())+", expected " \
      +std::to_string(input_handle->get_c_nnz()));
  if(static_cast<size_t>(nnz_C) != values_C.size())
    throw std::runtime_error("spgemm_numeric_rocsparse : " \
      "values_C has not been properly allocated. values_C.size() = " \
      +std::to_string(values_C.size())+", expected "+std::to_string(nnz_C));

  rocsparse_handle & handle = spgemm_handle.handle;

  auto operation_A = trans_A ? rocsparse_operation_transpose : 
                               rocsparse_operation_none;
  auto operation_B = trans_B ? rocsparse_operation_transpose : 
                               rocsparse_operation_none;

  if((spgemm_handle.opA != operation_A) ||
     (spgemm_handle.opB != operation_B))
    throw std::runtime_error("spgemm_numeric_rocsparse : " \
        "Using handle with incorrect transpose state.");

  // Matrix CSR row offset arrays
  KOKKOSKERNELS_ROCSPARSE_USE_OR_CONVERT_VIEW(const OrdinalType, csr_row_ptr_A, 
    InputRowView, row_map_A)
  KOKKOSKERNELS_ROCSPARSE_USE_OR_CONVERT_VIEW(const OrdinalType, csr_row_ptr_B, 
    InputRowView, row_map_B)
  KOKKOSKERNELS_ROCSPARSE_USE_OR_CONVERT_VIEW(const OrdinalType, csr_row_ptr_C, 
    InputRowView, row_map_C)
  const OrdinalType * csr_row_ptr_D = nullptr;

  // Matrix CSR column index arrays
  KOKKOSKERNELS_ROCSPARSE_USE_OR_CONVERT_VIEW(const OrdinalType,csr_col_ind_A, 
    InputColumnView,  columns_A)
  KOKKOSKERNELS_ROCSPARSE_USE_OR_CONVERT_VIEW(const OrdinalType,csr_col_ind_B, 
    InputColumnView,  columns_B)
  KOKKOSKERNELS_ROCSPARSE_USE_OR_ALLOC_VIEW(        OrdinalType,csr_col_ind_C,
    OutputColumnView, columns_C)
  const OrdinalType * csr_col_ind_D = nullptr;

  // Matrix CSR values arrays
  const RocsparseScalarType * csr_val_A = 
    reinterpret_cast<const RocsparseScalarType*>(values_A.data());
  const RocsparseScalarType * csr_val_B = 
    reinterpret_cast<const RocsparseScalarType*>(values_B.data());
        RocsparseScalarType * csr_val_C = 
    reinterpret_cast<      RocsparseScalarType*>(values_C.data());
  const RocsparseScalarType * csr_val_D = nullptr;

  // Set pointer mode (applies to scalar values only)
  KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(
    rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

  // Make sure device is ready
  Kokkos::fence();

  // Symbolic spgemm call - fills columns_C
  if(!spgemm_handle.C_populated){
    KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(
      rocsparse_csrgemm_symbolic(handle,
                                 operation_A,operation_B,
                                 m,n,k,
                                 descr_A,nnz_A,csr_row_ptr_A,csr_col_ind_A,
                                 descr_B,nnz_B,csr_row_ptr_B,csr_col_ind_B,
                                 descr_D,nnz_D,csr_row_ptr_D,csr_col_ind_D,
                                 descr_C,nnz_C,csr_row_ptr_C,csr_col_ind_C,
                                 info_C,
                                 buffer));

    KOKKOSKERNELS_ROCSPARSE_COPY_VIEW(OrdinalType, csr_col_ind_C, 
      OutputColumnView, columns_C);

    // Make sure device is ready
    Kokkos::fence();

    spgemm_handle.C_populated = true;
  }

  // Numeric spgemm call - fills values_C
  KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(
    rocsparse_Xcsrgemm_numeric(handle,
                          operation_A,operation_B,
                          m,n,k,
                          &alpha,
                          descr_A,nnz_A,csr_val_A,csr_row_ptr_A,csr_col_ind_A,
                          descr_B,nnz_B,csr_val_B,csr_row_ptr_B,csr_col_ind_B,
                          &beta,
                          descr_D,nnz_D,csr_val_D,csr_row_ptr_D,csr_col_ind_D,
                          descr_C,nnz_C,csr_val_C,csr_row_ptr_C,csr_col_ind_C,
                          info_C,
                          buffer));

  // Tell user the result is sorted
  input_handle->set_sort_option(1);

}

/**
 * @brief Numeric call for rocsparse SpGEMM
 * 
 * Used to fill values for C = B - omega * (D^{-1}*A) * B
 * Will also fill column indexes in C if that has not already been done
 *
 * @note The symbolic call must be called first or rocsparse will error out
 * @note In the symbolic call make sure the enable_jacobi flag is turned on
 *
 * @tparam InputHandle KernelHandle used to store handle information
 * @tparam InputRowView Row map view type for A and B
 * @tparam InputColumnView Column index view type for A and B
 * @tparam InputDataView Data view type for A and B
 * @tparam OutputColumnView Output column index view type for C
 * @tparam OutputRowView Output row map view type for C
 * @tparam OutputDataView Data view type for C
 * @param input_handle SpGEMM handle that will store various rocsparse content
 * @param m Number of rows in A
 * @param n Number of columns in A and rows in B
 * @param k Number of columns in B
 * @param row_map_A Row map for A
 * @param columns_A Column indexes for A
 * @param values_A Values for A
 * @param trans_A Use transpose of A (not supported by rocsparse)
 * @param row_map_B Row map for B
 * @param columns_B Column indexes for B
 * @param values_B Values for B
 * @param trans_B Use transpose of B (not supported by rocsparse)
 * @param row_map_C Input row map for C
 * @param columns_C Output column indexes for C
 * @param values_C Output values for C
 */
template <class InputHandle, class InputRowView, class InputColumnView, 
          class InputDataView, class OutputColumnView, class OutputDataView, 
          class InvDView>
void
jacobi_spgemm_numeric_rocsparse(InputHandle *       input_handle,
                                const rocsparse_int m,
                                const rocsparse_int n,
                                const rocsparse_int k,
                                InputRowView        row_map_A,
                                InputColumnView     columns_A,
                                InputDataView       values_A,
                                const bool          trans_A,
                                InputRowView        row_map_B,
                                InputColumnView     columns_B,
                                InputDataView       values_B,
                                const bool          trans_B,
                                typename InputDataView::const_value_type omega,
                                InvDView            values_invD,
                                InputRowView        row_map_C,
                                OutputColumnView    columns_C,
                                OutputDataView      values_C)
{

  // rocSPARSE solves C = alpha * A * B + beta * D
  // We need to evaluate C = B - omega * (D^{-1}*A) * B

  // In the numeric call we will fill columns_C and values_C.

  using OrdinalType = rocsparse_int;
  using ScalarType =  typename InputDataView::non_const_value_type;
  using RocsparseScalarType = 
    typename rocsparse_type_conversion<ScalarType>::Type;

  // Initialize scalar multipliers - not used at this point, 
  //  but we need beta=0 to ignore matrix D
  RocsparseScalarType alpha = 
    rocsparse_type_conversion<ScalarType>::convert(-omega);
  RocsparseScalarType beta  = 1;

  // This handle contains some useful rocsparse components
  auto & spgemm_handle = *input_handle->get_rocsparse_spgemm_handle();

  // Create matrix descriptors
  rocsparse_mat_descr descr_A = spgemm_handle.descr_A;
  rocsparse_mat_descr descr_B = spgemm_handle.descr_B;
  rocsparse_mat_descr descr_C = spgemm_handle.descr_C;
  rocsparse_mat_descr descr_D = spgemm_handle.descr_D;

  // Acquire info object for C
  rocsparse_mat_info & info_C = spgemm_handle.info_C;

  // Get the pre-allocated buffer (allocated in the symbolic call)
  void* buffer = static_cast<void*>(spgemm_handle.buffer.data());
  if(buffer == nullptr)
    throw std::runtime_error("jacobi_spgemm_numeric_rocsparse : " \
      "Buffer has not been allocated.");

  // If this is not set, then it is likely the allocations and 
  // row map for C are incorrect (defined by rocsparse_spgemm_symbolic call)
  // FIXME: We currently lack a clean way to implementent this flag. 
  // Hopefully if a user comes accross this issue the exception is verbose 
  // enough to describe the fix.
  if(! spgemm_handle.enable_jacobi)
    throw std::runtime_error("jacobi_spgemm_numeric_rocsparse : " \
      "Handle is not setup for Jacobi SpGEMM, this can lead to " \
      "undefined behavior so we do not allow it. Please set enable_jacobi " \
      "in rocsparse spgemm handle to true prior to running symbolic spgemm " \
      "call. i.e.:\nKokkosKernels::Experimental::KokkosKernelHandle::" \
      "get_spgemm_handle()->get_rocsparse_spgemm_handle()->enable_jacobi " \
      " = true");

  // invD needs to have num_rows entries
  if(static_cast<OrdinalType>(values_invD.size()) != m)
    throw std::runtime_error("jacobi_spgemm_numeric_rocsparse : "\
      "invD should be of size A_num_rows ("+std::to_string(m)+ \
      "), but is instead size "+std::to_string(values_invD.size())+".");

  const int nnz_A = columns_A.size();
  const int nnz_B = columns_B.size();
  const int nnz_C = columns_C.size();
  const int nnz_D = columns_B.size();

  // Make sure output allocations are properly sized
  if(input_handle->get_c_nnz() != nnz_C)
    throw std::runtime_error("jacobi_spgemm_numeric_rocsparse : " \
      "columns_C has not been properly allocated. columns_C.size() = " \
      +std::to_string(columns_C.size())+", expected " \
      +std::to_string(input_handle->get_c_nnz()));
  if(static_cast<size_t>(nnz_C) != values_C.size())
    throw std::runtime_error("jacobi_spgemm_numeric_rocsparse : " \
      "values_C has not been properly allocated. values_C.size() = " \
      +std::to_string(values_C.size())+", expected "+std::to_string(nnz_C));

  rocsparse_handle & handle = spgemm_handle.handle;

  auto operation_A = trans_A ? rocsparse_operation_transpose : 
                               rocsparse_operation_none;
  auto operation_B = trans_B ? rocsparse_operation_transpose : 
                               rocsparse_operation_none;

  if((spgemm_handle.opA != operation_A) ||
     (spgemm_handle.opB != operation_B))
    throw std::runtime_error("jacobi_spgemm_numeric_rocsparse : " \
        "Using handle with incorrect transpose state.");

  // Matrix CSR row offset arrays
  KOKKOSKERNELS_ROCSPARSE_USE_OR_CONVERT_VIEW(const OrdinalType, csr_row_ptr_A,
    InputRowView, row_map_A)
  KOKKOSKERNELS_ROCSPARSE_USE_OR_CONVERT_VIEW(const OrdinalType, csr_row_ptr_B,
    InputRowView, row_map_B)
  KOKKOSKERNELS_ROCSPARSE_USE_OR_CONVERT_VIEW(const OrdinalType, csr_row_ptr_C,
    InputRowView, row_map_C)
  const OrdinalType * csr_row_ptr_D = csr_row_ptr_B;

  // Matrix CSR column index arrays
  KOKKOSKERNELS_ROCSPARSE_USE_OR_CONVERT_VIEW(const OrdinalType, csr_col_ind_A,
    InputColumnView,  columns_A)
  KOKKOSKERNELS_ROCSPARSE_USE_OR_CONVERT_VIEW(const OrdinalType, csr_col_ind_B,
    InputColumnView,  columns_B)
  KOKKOSKERNELS_ROCSPARSE_USE_OR_ALLOC_VIEW(        OrdinalType, csr_col_ind_C,
    OutputColumnView, columns_C)
  const OrdinalType * csr_col_ind_D = csr_col_ind_B;

  // Before we call the numeric call for rocsparse, we need to fill out
  // the invD*A entries. Note that D is diagonal so it has no impact on 
  // the row_map or columns of the resulting operation

  // Allocate invDA_values if it not correctly sized
  if(spgemm_handle.csr_values_invDA.size() != nnz_A){
    spgemm_handle.csr_values_invDA =
      typename InputHandle::scalar_temp_work_view_t(
          Kokkos::ViewAllocateWithoutInitializing("local copy invDA"), nnz_A);
  }

  // Matrix CSR values arrays
        RocsparseScalarType * csr_val_A = 
    reinterpret_cast<      RocsparseScalarType*>(
      spgemm_handle.csr_values_invDA.data());
  const RocsparseScalarType * csr_val_B = 
    reinterpret_cast<const RocsparseScalarType*>(values_B.data());
        RocsparseScalarType * csr_val_C = 
    reinterpret_cast<      RocsparseScalarType*>(values_C.data());
  const RocsparseScalarType * csr_val_D = 
    reinterpret_cast<const RocsparseScalarType*>(csr_val_B);

  // Now we need to fill the csr_val_A with the results of invD * A
  {

    // Each team is a single wave/warp
    const OrdinalType num_rows = row_map_A.size()-1;
    const OrdinalType num_threads_per_row = 16;
    const OrdinalType num_rows_per_team = 16;
    using policy_t=Kokkos::TeamPolicy<Kokkos::Experimental::HIP>;
    using team_t=typename policy_t::member_type;

    // invD can be a high rank view with a single value per row, 
    // but it should be flat so this is "safe"
    const ScalarType * csr_val_A = values_A.data();
    const ScalarType * csr_val_invD = values_invD.data();
          ScalarType * csr_val_invDA = spgemm_handle.csr_values_invDA.data();

    policy_t policy((num_rows + num_rows_per_team - 1) / num_rows_per_team, 
                    num_rows_per_team, num_threads_per_row);
    Kokkos::parallel_for("Filling invD*A values", policy,
      KOKKOS_LAMBDA(const team_t & team){
        const OrdinalType row_begin = team.league_rank() * num_rows_per_team;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 
          row_begin, min(row_begin+num_rows_per_team, num_rows)),
          [&](const OrdinalType & row) {
            const auto mult = csr_val_invD[row];
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, 
              csr_row_ptr_A[row], csr_row_ptr_A[row + 1]),
              [&](const OrdinalType & idx) {
                csr_val_invDA[idx] = mult*csr_val_A[idx];
              });
          });
      });

    // Wait for kokkos to finish
    Kokkos::fence();
  }

  // Set pointer mode (applies to scalar values only)
  KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(
    rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

  // Symbolic spgemm call - fills columns_C
  if(!spgemm_handle.C_populated){
    KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(
      rocsparse_csrgemm_symbolic(handle,
                                 operation_A,operation_B,
                                 m,n,k,
                                 descr_A,nnz_A,csr_row_ptr_A,csr_col_ind_A,
                                 descr_B,nnz_B,csr_row_ptr_B,csr_col_ind_B,
                                 descr_D,nnz_D,csr_row_ptr_D,csr_col_ind_D,
                                 descr_C,nnz_C,csr_row_ptr_C,csr_col_ind_C,
                                 info_C,
                                 buffer));

    KOKKOSKERNELS_ROCSPARSE_COPY_VIEW(OrdinalType, csr_col_ind_C, 
      OutputColumnView, columns_C);

    spgemm_handle.C_populated = true;
  }

  // Numeric spgemm call - fills values_C
  KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(
    rocsparse_Xcsrgemm_numeric(handle,
                          operation_A, operation_B,
                          m, n, k,
                          &alpha,
                          descr_A,nnz_A,csr_val_A,csr_row_ptr_A,csr_col_ind_A,
                          descr_B,nnz_B,csr_val_B,csr_row_ptr_B,csr_col_ind_B,
                          &beta,
                          descr_D,nnz_D,csr_val_D,csr_row_ptr_D,csr_col_ind_D,
                          descr_C,nnz_C,csr_val_C,csr_row_ptr_C,csr_col_ind_C,
                          info_C,
                          buffer));

  // Tell user the result is sorted
  input_handle->set_sort_option(1);

}

}  // namespace Impl
}  // namespace KokkosSparse

#undef KOKKOSKERNELS_ROCSPARSE_USE_OR_CONVERT_VIEW
#undef KOKKOSKERNELS_ROCSPARSE_USE_OR_ALLOC_VIEW
#undef KOKKOSKERNELS_ROCSPARSE_COPY_VIEW

#endif
#endif


