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

#ifndef KOKKOSKERNELS_SPMV_DATA_HPP_
#define KOKKOSKERNELS_SPMV_DATA_HPP_

#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCSPARSE
#include <rocsparse.h>
#endif

struct spmv_additional_data {
  int test;
#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCSPARSE
  rocsparse_handle    handle;
  rocsparse_mat_descr descr;
  rocsparse_mat_info  info;
#endif

  spmv_additional_data(int test_) : test(test_) {
#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCSPARSE
    rocsparse_status    status;
    status = rocsparse_create_handle(&handle);
    status = rocsparse_create_mat_descr(&descr);
    status = rocsparse_create_mat_info(&info);
#endif
  };
  
  ~spmv_additional_data() {
#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCSPARSE
    rocsparse_status    status;
    status = rocsparse_destroy_mat_info(info);
    status = rocsparse_destroy_mat_descr(descr);
    status = rocsparse_destroy_handle(handle);
#endif
  }
};

#endif /* KOKKOSKERNELS_SPMV_DATA_HPP_ */
