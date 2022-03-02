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

#include "KokkosSparse_csc2csr.hpp"
#include "KokkosKernels_TestUtils.hpp"

namespace Test {
template <class ScalarType, class LayoutType, class ExeSpaceType>
void doCsc2Csr(size_t m, size_t n, ScalarType min_val, ScalarType max_val) {
  RandCscMat<ScalarType, LayoutType, ExeSpaceType> cscMat(m, n, min_val,
                                                          max_val);

  auto csrMat = KokkosSparse::csc2csr(
      cscMat.get_m(), cscMat.get_n(), cscMat.get_nnz(), cscMat.get_vals(),
      cscMat.get_row_ids(), cscMat.get_col_map());

  // TODO check csrMat against cscMat
}

template <class LayoutType, class ExeSpaceType>
void doAllScalarsCsc2Csr(size_t m, size_t n, int min, int max) {
  doCsc2Csr<float, LayoutType, ExeSpaceType>(m, n, min, max);
  doCsc2Csr<double, LayoutType, ExeSpaceType>(m, n, min, max);
  doCsc2Csr<Kokkos::complex<float>, LayoutType, ExeSpaceType>(m, n, min, max);
  doCsc2Csr<Kokkos::complex<double>, LayoutType, ExeSpaceType>(m, n, min, max);
}

template <class ExeSpaceType>
void doAllLayoutsCsc2Csr(size_t m, size_t n, int min, int max) {
  doAllScalarsCsc2Csr<Kokkos::LayoutLeft, ExeSpaceType>(m, n, min, max);
  doAllScalarsCsc2Csr<Kokkos::LayoutRight, ExeSpaceType>(m, n, min, max);
}

template <class ExeSpaceType>
void doAllCsc2csr(size_t m, size_t n) {
  int min = 1, max = 10;
  doAllLayoutsCsc2Csr<ExeSpaceType>(m, n, min, max);
}

TEST_F(TestCategory, sparse_csc2csr) {
  // Square cases
  for (size_t dim = 1; dim < 1024; dim *= 4)
    doAllCsc2csr<TestExecSpace>(dim, dim);

  // Non-square cases
  for (size_t dim = 1; dim < 1024; dim *= 4) {
    doAllCsc2csr<TestExecSpace>(dim * 3, dim);
    doAllCsc2csr<TestExecSpace>(dim, dim * 3);
  }
}
}  // namespace Test