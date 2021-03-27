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

#ifndef ROCSPARSE_SPMV_HPP_
#define ROCSPARSE_SPMV_HPP_

#include "KokkosKernels_spmv_data.hpp"

#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCSPARSE
#include <rocsparse.h>

template<typename Scalar>
void rocsparse_matvec_wrapper(rocsparse_handle& handle, rocsparse_mat_descr& descr,
			      rocsparse_mat_info& info,
			      int numRows, int numCols, int nnz,
			      Scalar* values, int* rowptrs, int* entries,
			      Scalar* x, Scalar* y)
{
  throw std::runtime_error("Can't use rocSPARSE mat-vec for scalar types other than double and float.");
}

template<>
void rocsparse_matvec_wrapper<double>(rocsparse_handle& handle, rocsparse_mat_descr& descr,
				      rocsparse_mat_info& info,
				      int numRows, int numCols, int nnz,
				      double* values, int* rowptrs, int* entries,
				      double* x, double* y)
{
  double s_a = 1.0;
  double s_b = 0.0;
  rocsparse_dcsrmv(handle, rocsparse_operation_none,
		   numRows, numCols, nnz,
		   &s_a,
		   descr,
		   values, rowptrs, entries,
		   info,
		   x,
		   &s_b,
		   y);
}

template<>
void rocsparse_matvec_wrapper<float>(rocsparse_handle& handle, rocsparse_mat_descr& descr,
				     rocsparse_mat_info& info,
				     int numRows, int numCols, int nnz,
				     float* values, int* rowptrs, int* entries,
				     float* x, float* y)
{
  float s_a = 1.0f;
  float s_b = 0.0f;
  rocsparse_scsrmv(handle, rocsparse_operation_none,
		   numRows, numCols, nnz,
		   &s_a,
		   descr,
		   values, rowptrs, entries,
		   info,
		   x,
		   &s_b,
		   y);
}

template<typename AType, typename XType, typename YType>
void rocsparse_matvec(AType A, XType x, YType y, spmv_additional_data* data) {
  using Scalar = typename AType::non_const_value_type;

  // Run rocSPARSE spmv corresponding to scalar type
  rocsparse_matvec_wrapper<Scalar>(data->handle, data->descr, data->info,
				   A.numRows(), A.numCols(), A.nnz(),
				   A.values.data(),
				   const_cast<int *>(A.graph.row_map.data()),
				   A.graph.entries.data(),
				   x.data(), y.data());
}

#endif /* KOKKOSKERNELS_ENABLE_TPL_ROCSPARSE */

#endif /* ROCSPARSE_SPMV_HPP_ */
