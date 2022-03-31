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

#include <Kokkos_Core.hpp>
#include <iostream>
#include <string>

#ifndef __KOKKOSBATCHED_KRYLOV_HANDLE_HPP__
#define __KOKKOSBATCHED_KRYLOV_HANDLE_HPP__
//#define VERBOSE

namespace KokkosBatched {

/// \brief KrylovHandle
///
/// \tparam scalar_type: Scalar type of the linear solver

template <class NormViewType, class IntViewType, class ViewType3D>
class KrylovHandle {
 public:
  using norm_type = typename NormViewType::non_const_value_type;

  typedef ViewType3D ArnoldiViewType;
  typedef Kokkos::View<typename ViewType3D::non_const_value_type **,
                       typename ViewType3D::array_layout,
                       typename ViewType3D::execution_space>
      TemporaryViewType;

 public:
  NormViewType residual_norms;
  IntViewType iteration_numbers;
  typename NormViewType::HostMirror residual_norms_host;
  typename IntViewType::HostMirror iteration_numbers_host;
  IntViewType first_index;
  IntViewType last_index;
  ArnoldiViewType Arnoldi_view;
  TemporaryViewType tmp_view;

 private:
  norm_type tolerance;
  norm_type max_tolerance;
  int max_iteration;
  int batched_size;
  int N_team;
  int ortho_strategy;
  int scratch_pad_level;
  bool compute_last_residual;
  bool monitor_residual;
  bool host_synchronised;

 public:
  KrylovHandle(int _batched_size, int _N_team, int _max_iteration = 200,
               bool _monitor_residual = false)
      : max_iteration(_max_iteration),
        batched_size(_batched_size),
        N_team(_N_team),
        monitor_residual(_monitor_residual) {
    tolerance     = Kokkos::Details::ArithTraits<norm_type>::epsilon();
    max_tolerance = 1e-30;
    if (std::is_same<norm_type, double>::value) max_tolerance = 1e-50;
    if (monitor_residual) {
      residual_norms = NormViewType("", batched_size, max_iteration + 2);
    }
    iteration_numbers = IntViewType("", batched_size);
    Kokkos::deep_copy(iteration_numbers, -1);

    int n_teams = ceil(1. * batched_size / N_team);
    first_index = IntViewType("", n_teams);
    last_index  = IntViewType("", n_teams);

    auto first_index_host = Kokkos::create_mirror_view(first_index);
    auto last_index_host  = Kokkos::create_mirror_view(last_index);

    first_index_host(0) = 0;
    last_index_host(0)  = N_team;
    for (int i = 1; i < n_teams; ++i) {
      first_index_host(i) = last_index_host(i - 1);
      last_index_host(i)  = first_index_host(i) + N_team;
    }
    last_index_host(n_teams - 1) = batched_size;

    Kokkos::deep_copy(first_index, first_index_host);
    Kokkos::deep_copy(last_index, last_index_host);

    // Default Classical GS
    ortho_strategy        = 1;
    scratch_pad_level     = 0;
    compute_last_residual = true;
    host_synchronised     = false;
  }

  /// \brief reset
  ///   Reset the iteration numbers to the default value of -1
  ///   and the residual norms if monitored.
  ///   (Usefull when mulitple consecutive solvers use the same handle)
  ///

  void reset() {
    Kokkos::deep_copy(iteration_numbers, -1);
    if (monitor_residual) {
      Kokkos::deep_copy(residual_norms, 0.);
    }
    host_synchronised = false;
  }

  ///

  void synchronise_host() {
    iteration_numbers_host = Kokkos::create_mirror_view(iteration_numbers);
    Kokkos::deep_copy(iteration_numbers_host, iteration_numbers);
    if (monitor_residual) {
      residual_norms_host = Kokkos::create_mirror_view(residual_norms);
      Kokkos::deep_copy(residual_norms_host, residual_norms);
    }
    host_synchronised = true;
  }

  /// \brief is_converged
  ///   Test if all the systems have converged.
  ///

  KOKKOS_INLINE_FUNCTION
  bool is_converged() const {
    bool all_converged = true;
    for (size_t i = 0; i < batched_size; ++i)
      if (iteration_numbers(i) == -1) {
        all_converged = false;
        break;
      }
    return all_converged;
  }

  /// \brief is_converged_host
  ///   Test if all the systems have converged (host).
  ///

  bool is_converged_host() {
    if (!host_synchronised) this->synchronise_host();
    bool all_converged = true;
    for (int i = 0; i < batched_size; ++i)
      if (iteration_numbers_host(i) == -1) {
        all_converged = false;
        break;
      }
    return all_converged;
  }

  /// \brief is_converged
  ///   Test if one particular system has converged.
  ///
  /// \param batched_id [in]: Global batched ID

  KOKKOS_INLINE_FUNCTION
  bool is_converged(int batched_id) const {
    return (iteration_numbers(batched_id) != -1);
  }

  /// \brief is_converged
  ///   Test if one particular system has converged (host).
  ///
  /// \param batched_id [in]: Global batched ID

  bool is_converged_host(int batched_id) {
    if (!host_synchronised) this->synchronise_host();
    return (iteration_numbers_host(batched_id) != -1);
  }

  /// \brief set_tolerance
  ///   Set the tolerance of the batched Krylov solver
  ///
  /// \param _tolerance [in]: New tolerance

  KOKKOS_INLINE_FUNCTION
  void set_tolerance(norm_type _tolerance) { tolerance = _tolerance; }

  /// \brief get_tolerance
  ///   Get the tolerance of the batched Krylov solver

  KOKKOS_INLINE_FUNCTION
  norm_type get_tolerance() const { return tolerance; }

  /// \brief set_max_tolerance
  ///   Set the maximal tolerance of the batched Krylov solver
  ///
  /// \param _max_tolerance [in]: New tolerance

  KOKKOS_INLINE_FUNCTION
  void set_max_tolerance(norm_type _max_tolerance) {
    max_tolerance = _max_tolerance;
  }

  /// \brief get_max_tolerance
  ///   Get the maximal tolerance of the batched Krylov solver

  KOKKOS_INLINE_FUNCTION
  norm_type get_max_tolerance() const { return max_tolerance; }

  /// \brief set_max_iteration
  ///   Set the maximum number of iterations of the batched Krylov solver
  ///
  /// \param _max_iteration [in]: New maximum number of iterations

  KOKKOS_INLINE_FUNCTION
  void set_max_iteration(int _max_iteration) { max_iteration = _max_iteration; }

  /// \brief get_max_iteration
  ///   Get the maximum number of iterations of the batched Krylov solver

  KOKKOS_INLINE_FUNCTION
  int get_max_iteration() const { return max_iteration; }

  /// \brief set_norm
  ///   Store the norm of one of the system at one of the iteration
  ///
  /// \param batched_id [in]: Global batched ID
  /// \param iteration_id [in]: Iteration ID
  /// \param norm_i [in]: Norm to store

  KOKKOS_INLINE_FUNCTION
  void set_norm(int batched_id, int iteration_id, norm_type norm_i) const {
    if (monitor_residual) residual_norms(batched_id, iteration_id) = norm_i;
  }

  /// \brief set_norm
  ///   Store the norm of one of the system at one of the iteration
  ///
  /// \param batchedteam_id [in]: Team ID
  /// \param batched_id [in]: Local batched ID (local ID within the team)
  /// \param iteration_id [in]: Iteration ID
  /// \param norm_i [in]: Norm to store

  KOKKOS_INLINE_FUNCTION
  void set_norm(int team_id, int batched_id, int iteration_id,
                norm_type norm_i) const {
    if (monitor_residual)
      residual_norms(team_id * N_team + batched_id, iteration_id) = norm_i;
  }

  /// \brief get_norm
  ///   Get the norm of one system at a given iteration
  ///
  /// \param batched_id [in]: Global batched ID
  /// \param iteration_id [in]: Iteration ID

  KOKKOS_INLINE_FUNCTION
  norm_type get_norm(int batched_id, int iteration_id) const {
    if (monitor_residual) {
      return residual_norms(batched_id, iteration_id);
    } else
      return 0;
  }

  /// \brief get_norm_host
  ///   Get the norm of one system at a given iteration (host)
  ///
  /// \param batched_id [in]: Global batched ID
  /// \param iteration_id [in]: Iteration ID

  norm_type get_norm_host(int batched_id, int iteration_id) {
    if (monitor_residual) {
      if (!host_synchronised) this->synchronise_host();
      return residual_norms_host(batched_id, iteration_id);
    } else
      return 0;
  }

  /// \brief set_last_norm
  ///   Store the last norm of one system
  ///
  /// \param batched_id [in]: Global batched ID
  /// \param norm_i [in]: Norm to store

  KOKKOS_INLINE_FUNCTION
  void set_last_norm(int batched_id, norm_type norm_i) const {
    if (monitor_residual)
      residual_norms(batched_id, max_iteration + 1) = norm_i;
  }

  /// \brief set_last_norm
  ///   Store the last norm of one system
  ///
  /// \param batchedteam_id [in]: Team ID
  /// \param batched_id [in]: Local batched ID (local ID within the team)
  /// \param batched_id [in]: Global batched ID
  /// \param norm_i [in]: Norm to store

  KOKKOS_INLINE_FUNCTION
  void set_last_norm(int team_id, int batched_id, norm_type norm_i) const {
    if (monitor_residual)
      residual_norms(team_id * N_team + batched_id, max_iteration + 1) = norm_i;
  }

  /// \brief get_last_norm
  ///   Get the last norm of one system
  ///
  /// \param batched_id [in]: Global batched ID

  KOKKOS_INLINE_FUNCTION
  norm_type get_last_norm(int batched_id) const {
    if (monitor_residual && compute_last_residual) {
      return residual_norms(batched_id, max_iteration + 1);
    } else
      return 0;
  }

  /// \brief get_last_norm_host
  ///   Get the last norm of one system (host)
  ///
  /// \param batched_id [in]: Global batched ID

  norm_type get_last_norm_host(int batched_id) {
    if (monitor_residual && compute_last_residual) {
      if (!host_synchronised) this->synchronise_host();
      return residual_norms_host(batched_id, max_iteration + 1);
    } else
      return 0;
  }

  /// \brief set_iteration
  ///   Store the number of iteration after convergence for one system
  ///
  /// \param batched_id [in]: Global batched ID
  /// \param iteration_id [in]: Iteration ID

  KOKKOS_INLINE_FUNCTION
  void set_iteration(int batched_id, int iteration_id) const {
    iteration_numbers(batched_id) = iteration_id;
  }

  /// \brief set_iteration
  ///   Store the number of iteration after convergence for one system
  ///
  /// \param batchedteam_id [in]: Team ID
  /// \param batched_id [in]: Local batched ID (local ID within the team)
  /// \param iteration_id [in]: Iteration ID

  KOKKOS_INLINE_FUNCTION
  void set_iteration(int team_id, int batched_id, int iteration_id) const {
    iteration_numbers(team_id * N_team + batched_id) = iteration_id;
  }

  /// \brief get_iteration
  ///   Get the number of iteration after convergence for one system
  ///
  /// \param batched_id [in]: Global batched ID

  KOKKOS_INLINE_FUNCTION
  int get_iteration(int batched_id) const {
    return iteration_numbers(batched_id);
  }

  /// \brief get_iteration_host
  ///   Get the number of iteration after convergence for one system (host)
  ///
  /// \param batched_id [in]: Global batched ID

  int get_iteration_host(int batched_id) {
    if (!host_synchronised) this->synchronise_host();
    return iteration_numbers_host(batched_id);
  }

  /// \brief set_ortho_strategy
  ///   Set the used orthogonalization strategy.
  ///   Either classical GS (_ortho_strategy=0) or modified GS
  ///   (_ortho_strategy=1)
  ///
  /// \param _ortho_strategy [in]: used orthogonalization strategy

  KOKKOS_INLINE_FUNCTION
  void set_ortho_strategy(int _ortho_strategy) {
    ortho_strategy = _ortho_strategy;
  }

  /// \brief get_ortho_strategy
  ///   Get the used orthogonalization strategy.
  ///   Either classical GS (_ortho_strategy=0) or modified GS
  ///   (_ortho_strategy=1)

  KOKKOS_INLINE_FUNCTION
  int get_ortho_strategy() const { return ortho_strategy; }

  /// \brief set_scratch_pad_level
  ///   Set the scratch pad level used to store temporary variables.
  ///
  /// \param _scratch_pad_level [in]: used level

  KOKKOS_INLINE_FUNCTION
  void set_scratch_pad_level(int _scratch_pad_level) {
    scratch_pad_level = _scratch_pad_level;
  }

  /// \brief get_scratch_pad_level
  ///   Get the scratch pad level used to store temporary variables.

  KOKKOS_INLINE_FUNCTION
  int get_scratch_pad_level() const { return scratch_pad_level; }

  /// \brief set_compute_last_residual
  ///   Select if the last residual is explicitly computed.
  ///
  /// \param _compute_last_residual [in]: boolean that specifies if we compute
  /// the last residual explicitly

  KOKKOS_INLINE_FUNCTION
  void set_compute_last_residual(bool _compute_last_residual) {
    if (monitor_residual)
      compute_last_residual = _compute_last_residual;
    else
      compute_last_residual = false;
  }

  /// \brief get_compute_last_residual
  ///   Specify if the last residual has to be computed explicitly.

  KOKKOS_INLINE_FUNCTION
  bool get_compute_last_residual() const {
    if (monitor_residual)
      return compute_last_residual;
    else
      return false;
  }
};

}  // namespace KokkosBatched

#endif
