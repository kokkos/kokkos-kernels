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
// Questions? Contact James Foucar (jgfouca@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <Kokkos_Core.hpp>
#include <KokkosSparse_Preconditioner.hpp>
#include <iostream>
#include <string>

#ifndef _GMRESHANDLE_HPP
#define _GMRESHANDLE_HPP

namespace KokkosSparse {
namespace Experimental {

template <class size_type_, class lno_t_, class scalar_t_, class ExecutionSpace,
          class TemporaryMemorySpace, class PersistentMemorySpace>
class GMRESHandle {
 public:
  using HandleExecSpace             = ExecutionSpace;
  using HandleTempMemorySpace       = TemporaryMemorySpace;
  using HandlePersistentMemorySpace = PersistentMemorySpace;

  using execution_space = ExecutionSpace;
  using memory_space    = HandlePersistentMemorySpace;
  using device_t        = Kokkos::Device<execution_space, memory_space>;

  using size_type       = typename std::remove_const<size_type_>::type;
  using const_size_type = const size_type;

  using nnz_lno_t       = typename std::remove_const<lno_t_>::type;
  using const_nnz_lno_t = const nnz_lno_t;

  using nnz_scalar_t       = typename std::remove_const<scalar_t_>::type;
  using const_nnz_scalar_t = const nnz_scalar_t;

  using float_t = typename Kokkos::ArithTraits<nnz_scalar_t>::mag_type;

  using nnz_row_view_t = typename Kokkos::View<size_type *, device_t>;

  using nnz_lno_view_t = typename Kokkos::View<nnz_lno_t *, device_t>;

  using nnz_value_view_t = typename Kokkos::View<nnz_scalar_t *, device_t>;

  using nnz_value_view2d_t =
      typename Kokkos::View<nnz_scalar_t **, Kokkos::LayoutLeft, device_t>;

  using signed_integral_t = typename std::make_signed<
      typename nnz_row_view_t::non_const_value_type>::type;

  using signed_nnz_lno_view_t =
      Kokkos::View<signed_integral_t *, typename nnz_row_view_t::array_layout,
                   typename nnz_row_view_t::device_type,
                   typename nnz_row_view_t::memory_traits>;

  enum Ortho { CGS2, MGS };
  enum Flag { Conv, NoConv, LOA, NotRun };

 private:
  // Inputs

  size_type nrows;
  size_type m;
  size_type max_restart;

  float_t tol;

  Ortho ortho;

  bool verbose;

  // Internals
  int team_size;
  int vector_size;

  // Outputs
  int num_iters;
  float_t end_rel_res;
  Flag conv_flag_val;

 public:
  // Use set methods to control ortho, and verbose
  GMRESHandle(const size_type nrows_, const size_type m_ = 50,
              const size_type max_restart_ = 50)
      : nrows(nrows_),
        m(m_),
        max_restart(max_restart_),
        tol(1e-8),
        ortho(CGS2),
        verbose(false),
        team_size(-1),
        vector_size(-1),
        num_iters(-1),
        end_rel_res(0),
        conv_flag_val(NotRun) {
    if (m <= 0) {
      throw std::invalid_argument(
          "gmres: Please choose restart size m greater than zero.");
    }
  }

  void reset_handle(const size_type nrows_, const size_type m_ = 50,
                    const size_type max_restart_ = 50) {
    set_nrows(nrows_);
    set_m(m_);
    set_max_restart(max_restart_);
    set_tol(1e-8);
    set_ortho(CGS2);
    set_verbose(false);
    num_iters     = -1;
    end_rel_res   = 0;
    conv_flag_val = NotRun;
  }

  KOKKOS_INLINE_FUNCTION
  ~GMRESHandle() {}

  KOKKOS_INLINE_FUNCTION
  size_type get_nrows() const { return nrows; }

  KOKKOS_INLINE_FUNCTION
  void set_nrows(const size_type nrows_) { this->nrows = nrows_; }

  KOKKOS_INLINE_FUNCTION
  size_type get_m() const { return m; }

  KOKKOS_INLINE_FUNCTION
  void set_m(const size_type m_) { this->m = m_; }

  KOKKOS_INLINE_FUNCTION
  size_type get_max_restart() const { return max_restart; }

  KOKKOS_INLINE_FUNCTION
  void set_max_restart(const size_type max_restart_) {
    this->max_restart = max_restart_;
  }

  KOKKOS_INLINE_FUNCTION
  float_t get_tol() const { return tol; }

  KOKKOS_INLINE_FUNCTION
  void set_tol(const float_t tol_) { this->tol = tol_; }

  KOKKOS_INLINE_FUNCTION
  Ortho get_ortho() const { return ortho; }

  KOKKOS_INLINE_FUNCTION
  void set_ortho(const Ortho ortho_) { this->ortho = ortho_; }

  KOKKOS_INLINE_FUNCTION
  bool get_verbose() const { return verbose; }

  KOKKOS_INLINE_FUNCTION
  void set_verbose(const bool verbose_) { this->verbose = verbose_; }

  void set_team_size(const int ts) { this->team_size = ts; }
  int get_team_size() const { return this->team_size; }

  void set_vector_size(const int vs) { this->vector_size = vs; }
  int get_vector_size() const { return this->vector_size; }

  int get_num_iters() const {
    assert(get_conv_flag_val() != NotRun);
    return num_iters;
  }
  float_t get_end_rel_res() const {
    assert(get_conv_flag_val() != NotRun);
    return end_rel_res;
  }
  Flag get_conv_flag_val() const { return conv_flag_val; }

  void set_stats(int num_iters_, float_t end_rel_res_, Flag conv_flag_val_) {
    assert(conv_flag_val_ != NotRun);
    num_iters     = num_iters_;
    end_rel_res   = end_rel_res_;
    conv_flag_val = conv_flag_val_;
  }
};

}  // namespace Experimental
}  // namespace KokkosSparse

#endif
