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


#include "KokkosKernels_Controls.hpp"
#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCSPARSE
#include "rocsparse.h"
#endif

namespace KokkosSparse{

namespace Impl{


  template <typename KernelHandle,
  typename ain_row_index_view_type,
  typename ain_nonzero_index_view_type,
  typename bin_row_index_view_type,
  typename bin_nonzero_index_view_type,
  typename cin_row_index_view_type>
  void rocSPARSE_symbolic(KernelHandle *handle,
			  typename KernelHandle::nnz_lno_t m,
			  typename KernelHandle::nnz_lno_t n,
			  typename KernelHandle::nnz_lno_t k,
			  ain_row_index_view_type row_mapA,
			  ain_nonzero_index_view_type entriesA,
			  bool transposeA,
			  bin_row_index_view_type row_mapB,
			  bin_nonzero_index_view_type entriesB,
			  bool transposeB,
			  cin_row_index_view_type row_mapC) {

#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCSPARSE

    using device1   = typename ain_row_index_view_type::device_type;
    using device2   = typename ain_nonzero_index_view_type::device_type;
    using idx       = typename KernelHandle::nnz_lno_t;
    using size_type = typename KernelHandle::size_type;
    using scalar    = typename KernelHandle::nnz_scalar_t;


    //TODO this is not correct, check memory space.
    if (std::is_same<Kokkos::Experimental::HIP, device1 >::value){
      throw std::runtime_error ("MEMORY IS NOT ALLOCATED IN GPU DEVICE for CUSPARSE\n");
      //return;
    }
    if (std::is_same<Kokkos::Experimental::HIP, device2 >::value){
      throw std::runtime_error ("MEMORY IS NOT ALLOCATED IN GPU DEVICE for CUSPARSE\n");
      //return;
    }

    typename KernelHandle::SPGEMMrocSPARSEHandleType *spgemmTPLHandle = handle->get_rocSPARSEHandle();

    // Fake alpha and beta, just setting as 1.0 and 0.0
    // to ensure that the sparsity pattern of C is computed
    // correctly!
    const scalar alpha = 1.0;
    const scalar beta  = 0.0;

    int nnzA = entriesA.extent(0);
    int nnzB = entriesB.extent(0);
    int nnzD = 0;

    size_t buffer_size;

    if(std::is_same<size_type, int>::value && std::is_same<idx, int>::value) {
      if(std::is_same<scalar, float>::value) {
	rocsparse_scsrgemm_buffer_size(spgemmTPLHandle->handle,
				       rocsparse_operation_none, rocsparse_operation_none,
				       int(m), int(n), int(k),
				       (const float *) &alpha,
				       spgemmTPLHandle->a_descr, nnzA,
				       (const int *) row_mapA.data(),
				       (const int *) entriesA.data(),
				       spgemmTPLHandle->b_descr, nnzB,
				       (const int *) row_mapB.data(),
				       (const int *) entriesB.data(),
				       (const float *) &beta,
				       spgemmTPLHandle->d_descr, nnzD,
				       spgemmTPLHandle->row_mapD,
				       spgemmTPLHandle->entriesD,
				       spgemmTPLHandle->c_info,
				       &buffer_size);
 
      } else if(std::is_same<scalar, double>::value) {
	rocsparse_dcsrgemm_buffer_size(spgemmTPLHandle->handle,
				       rocsparse_operation_none, rocsparse_operation_none,
				       int(m), int(n), int(k),
				       (const double *) &alpha,
				       spgemmTPLHandle->a_descr, nnzA,
				       (const int *) row_mapA.data(),
				       (const int *) entriesA.data(),
				       spgemmTPLHandle->b_descr, nnzB,
				       (const int *) row_mapB.data(),
				       (const int *) entriesB.data(),
				       (const double *) &beta,
				       spgemmTPLHandle->d_descr, nnzD,
				       spgemmTPLHandle->row_mapD,
				       spgemmTPLHandle->entriesD,
				       spgemmTPLHandle->c_info,
				       &buffer_size);
      }

      // Allocate buffer
      hipMalloc(&(spgemmTPLHandle->buffer), buffer_size);

      // Obtain number of total non-zero entries in C and row pointers of C
      rocsparse_int nnzC;
      rocsparse_csrgemm_nnz(spgemmTPLHandle->handle,
			    rocsparse_operation_none, rocsparse_operation_none,
			    rocsparse_int(m), rocsparse_int(n), rocsparse_int(k),
			    spgemmTPLHandle->a_descr, nnzA,
			    (const rocsparse_int *) row_mapA.data(),
			    (const rocsparse_int *) entriesA.data(),
			    spgemmTPLHandle->b_descr, nnzB,
			    (const rocsparse_int *) row_mapB.data(),
			    (const rocsparse_int *) entriesB.data(),
			    spgemmTPLHandle->d_descr, nnzD,
			    spgemmTPLHandle->row_mapD,
			    spgemmTPLHandle->entriesD,
			    spgemmTPLHandle->c_descr,
			    (rocsparse_int *) row_mapC.data(),
			    &nnzC,
			    spgemmTPLHandle->c_info,
			    spgemmTPLHandle->buffer);
      handle->set_c_nnz(nnzC);
    }
#endif /* KOKKOSKERNELS_ENABLE_TPL_ROCSPARSE */
  } // rocSPARSE_symbolic



  template <typename KernelHandle,
  typename ain_row_index_view_type,
  typename ain_nonzero_index_view_type,
  typename ain_nonzero_value_view_type,
  typename bin_row_index_view_type,
  typename bin_nonzero_index_view_type,
  typename bin_nonzero_value_view_type,
  typename cin_row_index_view_type,
  typename cin_nonzero_index_view_type,
  typename cin_nonzero_value_view_type>
  void rocSPARSE_apply(KernelHandle *handle,
		       typename KernelHandle::nnz_lno_t m,
		       typename KernelHandle::nnz_lno_t n,
		       typename KernelHandle::nnz_lno_t k,
		       ain_row_index_view_type row_mapA,
		       ain_nonzero_index_view_type entriesA,
		       ain_nonzero_value_view_type valuesA,
		       bool /* transposeA */,
		       bin_row_index_view_type row_mapB,
		       bin_nonzero_index_view_type entriesB,
		       bin_nonzero_value_view_type valuesB,
		       bool /* transposeB */,
		       cin_row_index_view_type row_mapC,
		       cin_nonzero_index_view_type entriesC,
		       cin_nonzero_value_view_type valuesC) {

#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCSPARSE

    using device1   = typename ain_row_index_view_type::device_type;
    using device2   = typename ain_nonzero_index_view_type::device_type;
    using idx       = typename KernelHandle::nnz_lno_t;
    using size_type = typename KernelHandle::size_type;
    using scalar    = typename KernelHandle::nnz_scalar_t;


    //TODO this is not correct, check memory space.
    if (std::is_same<Kokkos::Experimental::HIP, device1 >::value){
      throw std::runtime_error ("MEMORY IS NOT ALLOCATED IN GPU DEVICE for CUSPARSE\n");
      //return;
    }
    if (std::is_same<Kokkos::Experimental::HIP, device2 >::value){
      throw std::runtime_error ("MEMORY IS NOT ALLOCATED IN GPU DEVICE for CUSPARSE\n");
      //return;
    }

    typename KernelHandle::SPGEMMrocSPARSEHandleType *spgemmTPLHandle = handle->get_rocSPARSEHandle();

    const scalar alpha = 1.0;
    const scalar beta  = 0.0;
    rocsparse_int nnzA = static_cast<rocsparse_int>(entriesA.extent(0));
    rocsparse_int nnzB = static_cast<rocsparse_int>(entriesB.extent(0));
    rocsparse_int nnzD = static_cast<rocsparse_int>(0);

    if(std::is_same<size_type, int>::value && std::is_same<idx, int>::value) {
      if(std::is_same<scalar, float>::value) {
	rocsparse_scsrgemm(spgemmTPLHandle->handle,
			   rocsparse_operation_none, rocsparse_operation_none,
			   rocsparse_int(m), rocsparse_int(n), rocsparse_int(k),
			   (const float *) &alpha,
			   spgemmTPLHandle->a_descr, nnzA,
			   (float *) valuesA.data(),
			   (const rocsparse_int *) row_mapA.data(),
			   (const rocsparse_int *) entriesA.data(),
			   spgemmTPLHandle->b_descr, nnzB,
			   (float *) valuesB.data(),
			   (const rocsparse_int *) row_mapB.data(),
			   (const rocsparse_int *) entriesB.data(),
			   (const float *) &beta,
			   spgemmTPLHandle->d_descr, nnzD,
			   spgemmTPLHandle->valuesDf,
			   spgemmTPLHandle->row_mapD,
			   spgemmTPLHandle->entriesD,
			   spgemmTPLHandle->c_descr,
			   (float *) valuesC.data(),
			   (const rocsparse_int *) row_mapC.data(),
			   (const rocsparse_int *) entriesC.data(),
			   spgemmTPLHandle->c_info,
			   spgemmTPLHandle->buffer);

      } else if(std::is_same<scalar, float>::value) {
	rocsparse_scsrgemm(spgemmTPLHandle->handle,
			   rocsparse_operation_none, rocsparse_operation_none,
			   rocsparse_int(m), rocsparse_int(n), rocsparse_int(k),
			   (const double *) &alpha,
			   spgemmTPLHandle->a_descr, nnzA,
			   (const double *) valuesA.data(),
			   (const rocsparse_int *) row_mapA.data(),
			   (const rocsparse_int *) entriesA.data(),
			   spgemmTPLHandle->b_descr, nnzB,
			   (const double *) valuesB.data(),
			   (const rocsparse_int *) row_mapB.data(),
			   (const rocsparse_int *) entriesB.data(),
			   (const double *) &beta,
			   spgemmTPLHandle->d_descr, nnzD,
			   spgemmTPLHandle->valuesDd,
			   spgemmTPLHandle->row_mapD,
			   spgemmTPLHandle->entriesD,
			   spgemmTPLHandle->c_descr,
			   (double *) valuesC.data(),
			   (const rocsparse_int *) row_mapC.data(),
			   (const rocsparse_int *) entriesC.data(),
			   spgemmTPLHandle->c_info,
			   spgemmTPLHandle->buffer);
	hipFree(spgemmTPLHandle->buffer);
      }
    }

#endif /* KOKKOSKERNELS_ENABLE_TPL_ROCSPARSE */

  } // rocSPARSE_apply
} // namespace Impl
} // namespace KokkosSparse
#endif /* _KOKKOSSPGEMMROCSPARSE_HPP */
