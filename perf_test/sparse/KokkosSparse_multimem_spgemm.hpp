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

#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosSparse_BsrMatrix.hpp"
#include "KokkosSparse_run_spgemm.hpp"
#include "KokkosSparse_IOUtils.hpp"

namespace KokkosKernels {

namespace Experiment {

namespace Impl {

// generic utility to convert CSR matrix type into similar BSR matrix type
template <typename crsMat_t>
using bsrMat_t = KokkosSparse::Experimental::BsrMatrix<
    typename crsMat_t::value_type, typename crsMat_t::ordinal_type,
    typename crsMat_t::device_type, typename crsMat_t::memory_traits,
    typename crsMat_t::size_type>;

template <typename exec_space, typename mem_space1, typename mem_space0,
          typename fast_mat_t, typename slow_mat_t>
struct SpGEMMRunHelper {
  template <typename ret_mat_t>
  static KOKKOS_INLINE_FUNCTION void run(const Parameters &params,
                                         fast_mat_t a_fast, slow_mat_t a_slow,
                                         fast_mat_t b_fast, slow_mat_t b_slow,
                                         ret_mat_t &c) {
    if (params.work_mem_space == 1) {
      c = run_mem<ret_mat_t, mem_space1>(params, a_fast, a_slow, b_fast,
                                         b_slow);
    } else {
      c = run_mem<ret_mat_t, mem_space0>(params, a_fast, a_slow, b_fast,
                                         b_slow);
    }
  }

 private:
  template <typename ret_mat_t, typename mem_space>
  static KOKKOS_INLINE_FUNCTION ret_mat_t run_mem(const Parameters &params,
                                                  fast_mat_t a_fast,
                                                  slow_mat_t a_slow,
                                                  fast_mat_t b_fast,
                                                  slow_mat_t b_slow) {
    if (params.a_mem_space == 1) {
      return run_a<ret_mat_t, mem_space>(params, a_fast, b_fast, b_slow);
    } else {
      return run_a<ret_mat_t, mem_space>(params, a_slow, b_fast, b_slow);
    }
  }
  template <typename ret_mat_t, typename mem_space, typename a_mat_t>
  static KOKKOS_INLINE_FUNCTION ret_mat_t run_a(const Parameters &params,
                                                a_mat_t a, fast_mat_t b_fast,
                                                slow_mat_t b_slow) {
    if (params.b_mem_space == 1) {
      return run_b<ret_mat_t, mem_space, a_mat_t>(params, a, b_fast);
    } else {
      return run_b<ret_mat_t, mem_space, a_mat_t>(params, a, b_slow);
    }
  }
  template <typename ret_mat_t, typename mem_space, typename a_mat_t,
            typename b_mat_t>
  static KOKKOS_INLINE_FUNCTION ret_mat_t run_b(const Parameters &params,
                                                a_mat_t a, b_mat_t b) {
    return KokkosKernels::Experiment::run_experiment<
        exec_space, ret_mat_t, a_mat_t, b_mat_t, mem_space, mem_space>(a, b,
                                                                       params);
  }
};

}  // namespace Impl

template <typename size_type, typename lno_t, typename scalar_t,
          typename exec_space, typename hbm_mem_space, typename sbm_mem_space>
void run_multi_mem_spgemm(Parameters params) {
  typedef exec_space myExecSpace;
  typedef Kokkos::Device<exec_space, hbm_mem_space> myFastDevice;
  typedef Kokkos::Device<exec_space, sbm_mem_space> mySlowExecSpace;

  typedef typename KokkosSparse::CrsMatrix<scalar_t, lno_t, myFastDevice, void,
                                           size_type>
      fast_crstmat_t;
  typedef typename KokkosSparse::CrsMatrix<scalar_t, lno_t, mySlowExecSpace,
                                           void, size_type>
      slow_crstmat_t;

  char *a_mat_file = params.a_mtx_bin_file;
  char *b_mat_file = params.b_mtx_bin_file;
  char *c_mat_file = params.c_mtx_bin_file;

  slow_crstmat_t a_slow_crsmat, b_slow_crsmat, c_slow_crsmat;
  fast_crstmat_t a_fast_crsmat, b_fast_crsmat, c_fast_crsmat;

  // read a and b matrices and store them on slow or fast memory.

  if (params.a_mem_space == 1) {
    a_fast_crsmat =
        KokkosSparse::Impl::read_kokkos_crst_matrix<fast_crstmat_t>(a_mat_file);
  } else {
    a_slow_crsmat =
        KokkosSparse::Impl::read_kokkos_crst_matrix<slow_crstmat_t>(a_mat_file);
  }

  if ((b_mat_file == NULL || strcmp(b_mat_file, a_mat_file) == 0) &&
      params.b_mem_space == params.a_mem_space) {
    std::cout << "Using A matrix for B as well" << std::endl;
    b_fast_crsmat = a_fast_crsmat;
    b_slow_crsmat = a_slow_crsmat;
  } else if (params.b_mem_space == 1) {
    if (b_mat_file == NULL) b_mat_file = a_mat_file;
    b_fast_crsmat =
        KokkosSparse::Impl::read_kokkos_crst_matrix<fast_crstmat_t>(b_mat_file);
  } else {
    if (b_mat_file == NULL) b_mat_file = a_mat_file;
    b_slow_crsmat =
        KokkosSparse::Impl::read_kokkos_crst_matrix<slow_crstmat_t>(b_mat_file);
  }

  if (params.block_size > 0) {
    using fast_bsr_t = Impl::bsrMat_t<fast_crstmat_t>;
    using slow_bsr_t = Impl::bsrMat_t<slow_crstmat_t>;
    using Helper     = Impl::SpGEMMRunHelper<myExecSpace, hbm_mem_space,
                                         sbm_mem_space, fast_bsr_t, slow_bsr_t>;
    const auto bs    = std::max(1, params.block_size);
    auto a_fast      = fast_bsr_t(a_fast_crsmat, bs);
    auto a_slow      = slow_bsr_t(a_slow_crsmat, bs);
    auto b_fast      = fast_bsr_t(b_fast_crsmat, bs);
    auto b_slow      = slow_bsr_t(b_slow_crsmat, bs);
    // Note: C result is temporary as BSR export is not yet supported
    if (params.c_mem_space == 1) {
      fast_bsr_t c_fast;
      Helper::run(params, a_fast, a_slow, b_fast, b_slow, c_fast);
    } else {
      slow_bsr_t c_slow;
      Helper::run(params, a_fast, a_slow, b_fast, b_slow, c_slow);
    }
  } else {
    using Helper =
        Impl::SpGEMMRunHelper<myExecSpace, hbm_mem_space, sbm_mem_space,
                              fast_crstmat_t, slow_crstmat_t>;
    if (params.c_mem_space == 1) {
      Helper::run(params, a_fast_crsmat, a_slow_crsmat, b_fast_crsmat,
                  b_slow_crsmat, c_fast_crsmat);
    } else {
      Helper::run(params, a_fast_crsmat, a_slow_crsmat, b_fast_crsmat,
                  b_slow_crsmat, c_slow_crsmat);
    }
  }

  if (c_mat_file != NULL) {
    if (params.block_size > 1) {
      std::cerr << "Exporting BSR results is not supported" << std::endl;
    } else if (params.c_mem_space == 1) {
      KokkosSparse::sort_crs_matrix(c_fast_crsmat);

      KokkosSparse::Impl::write_graph_bin(
          (lno_t)(c_fast_crsmat.numRows()),
          (size_type)(c_fast_crsmat.graph.entries.extent(0)),
          c_fast_crsmat.graph.row_map.data(),
          c_fast_crsmat.graph.entries.data(), c_fast_crsmat.values.data(),
          c_mat_file);
    } else {
      KokkosSparse::sort_crs_matrix(c_slow_crsmat);

      KokkosSparse::Impl::write_graph_bin(
          (lno_t)c_slow_crsmat.numRows(),
          (size_type)c_slow_crsmat.graph.entries.extent(0),
          c_slow_crsmat.graph.row_map.data(),
          c_slow_crsmat.graph.entries.data(), c_slow_crsmat.values.data(),
          c_mat_file);
    }
  }
}

}  // namespace Experiment
}  // namespace KokkosKernels
