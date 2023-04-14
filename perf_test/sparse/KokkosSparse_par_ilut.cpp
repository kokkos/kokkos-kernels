//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <cstdio>

#include <ctime>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <cmath>
#include <unordered_map>
#include <iomanip>  // std::setprecision

#include <Kokkos_Core.hpp>

#include "KokkosSparse_Utils.hpp"
#include "KokkosSparse_spiluk.hpp"
#include "KokkosSparse_par_ilut.hpp"
#include "KokkosSparse_spmv.hpp"
#include "KokkosBlas1_nrm2.hpp"
#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosKernels_default_types.hpp"
#include <KokkosKernels_IOUtils.hpp>
#include <KokkosSparse_IOUtils.hpp>

#ifdef USE_GINKGO
#include <ginkgo/ginkgo.hpp>
#endif

namespace {

using KokkosSparse::Experimental::par_ilut_symbolic;
using KokkosSparse::Experimental::par_ilut_numeric;

using KokkosSparse::Experimental::spiluk_symbolic;
using KokkosSparse::Experimental::spiluk_numeric;
using KokkosSparse::Experimental::SPILUKAlgorithm;

// Build up useful types
using scalar_t  = default_scalar;
using lno_t     = default_lno_t;
using size_type = default_size_type;
using exe_space = Kokkos::DefaultExecutionSpace;
using mem_space = typename exe_space::memory_space;
using device    = Kokkos::Device<exe_space, mem_space>;

using RowMapType  = Kokkos::View<size_type*, device>;
using EntriesType = Kokkos::View<lno_t*, device>;
using ValuesType  = Kokkos::View<scalar_t*, device>;

using sp_matrix_type =
  KokkosSparse::CrsMatrix<scalar_t, lno_t, device, void, size_type>;
using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<
  size_type, lno_t, scalar_t, exe_space, mem_space, mem_space>;
using float_t = typename Kokkos::ArithTraits<scalar_t>::mag_type;

///////////////////////////////////////////////////////////////////////////////
int run_par_ilut_test(KernelHandle& kh, const sp_matrix_type& A,
                      const int rows, const int team_size, const int loop)
///////////////////////////////////////////////////////////////////////////////
{
  auto par_ilut_handle = kh.get_par_ilut_handle();

  // Pull out views from CRS
  auto A_row_map = A.graph.row_map;
  auto A_entries = A.graph.entries;
  auto A_values  = A.values;

  // Allocate L and U CRS views as outputs
  RowMapType L_row_map("L_row_map", rows + 1);
  RowMapType U_row_map("U_row_map", rows + 1);

  // Initial L/U approximations for A
  EntriesType L_entries("L_entries", 0);
  ValuesType L_values("L_values", 0);
  EntriesType U_entries("U_entries", 0);
  ValuesType U_values("U_values", 0);

  Kokkos::Timer timer;
  double min_time = std::numeric_limits<double>::infinity();
  double max_time = 0.0;
  double ave_time = 0.0;
  int num_iters = -1;
  size_type nnzL = 0;
  size_type nnzU = 0;
  for (int i = 0; i < loop; ++i) {
    // Run par_ilut
    timer.reset();
    par_ilut_symbolic(&kh, A_row_map, A_entries, L_row_map, U_row_map);

    nnzL = par_ilut_handle->get_nnzL();
    nnzU = par_ilut_handle->get_nnzU();

    Kokkos::resize(L_entries, nnzL);
    Kokkos::resize(U_entries, nnzU);
    Kokkos::resize(L_values, nnzL);
    Kokkos::resize(U_values, nnzU);
    Kokkos::deep_copy(L_entries, 0);
    Kokkos::deep_copy(U_entries, 0);
    Kokkos::deep_copy(L_values, 0);
    Kokkos::deep_copy(U_values, 0);

    par_ilut_numeric(&kh, A_row_map, A_entries, A_values, L_row_map, L_entries,
                     L_values, U_row_map, U_entries, U_values);
    Kokkos::fence();

    // Check worked
    num_iters = par_ilut_handle->get_num_iters();
    KK_REQUIRE_MSG(num_iters < par_ilut_handle->get_max_iter(), "par_ilut hit max iters");

    // Measure time
    double time = timer.seconds();
    ave_time += time;
    if (time > max_time) max_time = time;
    if (time < min_time) min_time = time;

    // Reset inputs
    Kokkos::deep_copy(L_row_map, 0);
    Kokkos::deep_copy(U_row_map, 0);

    // Report run so user knows something is happening
    std::cout << "PAR_ILUT Finished a run in:  " << time << " seconds and " << num_iters << " iters" << std::endl;
  }

  std::cout << "PAR_ILUT LOOP_AVG_TIME:  " << ave_time / loop << std::endl;
  std::cout << "PAR_ILUT LOOP_MAX_TIME:  " << max_time << std::endl;
  std::cout << "PAR_ILUT LOOP_MIN_TIME:  " << min_time << std::endl;

  return num_iters;
}

#ifdef USE_GINKGO
///////////////////////////////////////////////////////////////////////////////
void run_par_ilut_test_ginkgo(KernelHandle& kh, const sp_matrix_type& A,
                              const int rows, const int team_size, const int loop, const int num_iters)
///////////////////////////////////////////////////////////////////////////////
{
  auto par_ilut_handle = kh.get_par_ilut_handle();

  // Pull out views from CRS
  auto A_row_map = A.graph.row_map;
  auto A_entries = A.graph.entries;
  auto A_values  = A.values;

  using mtx = gko::matrix::Csr<scalar_t, lno_t>;
  auto exec = gko::OmpExecutor::create();

  // ginkgo does not differentiate between index type and size type. We need
  // to convert A_row_map to lno_t.
  EntriesType A_row_map_cp("A_row_map_cp", rows+1);
  for (size_type i = 0; i < A_row_map.extent(0); ++i) {
    A_row_map_cp(i) = A_row_map(i);
  }

  // Populate mtx
  auto a_mtx_uniq = mtx::create_const(
    exec, gko::dim<2>(rows, rows),
    gko::array<scalar_t>::const_view(exec, A_values.extent(0), A_values.data()),
    gko::array<lno_t>::const_view(exec, A_entries.extent(0), A_entries.data()),
    gko::array<lno_t>::const_view(exec, A_row_map_cp.extent(0), A_row_map_cp.data()));

  std::shared_ptr<const mtx> a_mtx = std::move(a_mtx_uniq);

  Kokkos::Timer timer;
  double min_time = std::numeric_limits<double>::infinity();
  double max_time = 0.0;
  double ave_time = 0.0;
  for (int i = 0; i < loop; ++i) {
    timer.reset();
    auto fact = gko::factorization::ParIlut<scalar_t, lno_t>::build()
      .with_fill_in_limit(par_ilut_handle->get_fill_in_limit())
      .with_approximate_select(false)
      .with_iterations(num_iters)
      .on(exec)->generate(a_mtx);

    double time = timer.seconds();
    ave_time += time;
    if (time > max_time) max_time = time;
    if (time < min_time) min_time = time;

    // Report run so user knows something is happening
    std::cout << "GINKGO Finished a run in:  " << time << " seconds" << std::endl;
  }

  std::cout << "GINKGO LOOP_AVG_TIME:  " << ave_time / loop << std::endl;
  std::cout << "GINKGO LOOP_MAX_TIME:  " << max_time << std::endl;
  std::cout << "GINKGO LOOP_MIN_TIME:  " << min_time << std::endl;
}
#endif

///////////////////////////////////////////////////////////////////////////////
void run_spiluk_test(KernelHandle& kh, const sp_matrix_type& A,
                     const int rows, const int team_size, const int loop)
///////////////////////////////////////////////////////////////////////////////
{
  constexpr int EXPAND_FACT = 10;
  const lno_t fill_lev = 2;
  const size_type handle_nnz = EXPAND_FACT * A.nnz() * (fill_lev + 1);
  kh.create_spiluk_handle(SPILUKAlgorithm::SEQLVLSCHD_TP1, rows, handle_nnz, handle_nnz);
  auto spiluk_handle = kh.get_spiluk_handle();
  spiluk_handle->set_team_size(team_size);

  // Pull out views from CRS
  auto A_row_map = A.graph.row_map;
  auto A_entries = A.graph.entries;
  auto A_values  = A.values;

  // Allocate L and U CRS views as outputs
  RowMapType L_row_map("L_row_map", rows + 1);
  RowMapType U_row_map("U_row_map", rows + 1);

  // Initial L/U approximations for A
  EntriesType L_entries("L_entries", handle_nnz);
  ValuesType L_values("L_values", handle_nnz);
  EntriesType U_entries("U_entries", handle_nnz);
  ValuesType U_values("U_values", handle_nnz);

  Kokkos::Timer timer;
  double min_time = std::numeric_limits<double>::infinity();
  double max_time = 0.0;
  double ave_time = 0.0;
  for (int i = 0; i < loop; ++i) {
    // Run par_ilut
    timer.reset();
    spiluk_symbolic(&kh, fill_lev, A_row_map, A_entries, L_row_map, L_entries, U_row_map, U_entries);

    const size_type nnzL = spiluk_handle->get_nnzL();
    const size_type nnzU = spiluk_handle->get_nnzU();

    Kokkos::resize(L_entries, nnzL);
    Kokkos::resize(U_entries, nnzU);
    Kokkos::resize(L_values, nnzL);
    Kokkos::resize(U_values, nnzU);

    spiluk_numeric(&kh, fill_lev, A_row_map, A_entries, A_values, L_row_map, L_entries,
                   L_values, U_row_map, U_entries, U_values);
    Kokkos::fence();

    // Measure time
    double time = timer.seconds();
    ave_time += time;
    if (time > max_time) max_time = time;
    if (time < min_time) min_time = time;

    // Reset inputs
    Kokkos::deep_copy(L_row_map, 0);
    Kokkos::deep_copy(U_row_map, 0);
    Kokkos::deep_copy(L_entries, 0);
    Kokkos::deep_copy(U_entries, 0);
    Kokkos::deep_copy(L_values, 0);
    Kokkos::deep_copy(U_values, 0);
    Kokkos::resize(L_entries, handle_nnz);
    Kokkos::resize(U_entries, handle_nnz);

    spiluk_handle->reset_handle(rows, handle_nnz, handle_nnz);

    // Report run so user knows something is happening
    std::cout << "SPILUK Finished a run in:  " << time << " seconds" << std::endl;
  }

  std::cout << "SPILUK LOOP_AVG_TIME:  " << ave_time / loop << std::endl;
  std::cout << "SPILUK LOOP_MAX_TIME:  " << max_time << std::endl;
  std::cout << "SPILUK LOOP_MIN_TIME:  " << min_time << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
int test_par_ilut_perf(const int rows, const int nnz_per_row, const int bandwidth_per_nnz,
                       const int team_size, const int loop)
///////////////////////////////////////////////////////////////////////////////
{
  KernelHandle kh;
  kh.create_par_ilut_handle();

  // Report test config to user
  auto par_ilut_handle = kh.get_par_ilut_handle();
  par_ilut_handle->set_team_size(team_size);
  par_ilut_handle->set_nrows(rows);

  const auto default_policy = par_ilut_handle->get_default_team_policy();

  // Generate A
  size_type nnz   = rows * nnz_per_row;
  const lno_t bandwidth = nnz_per_row * bandwidth_per_nnz;
  const lno_t row_size_variance = 0;
  const scalar_t diag_dominance = 1;
  auto A = KokkosSparse::Impl::kk_generate_diagonally_dominant_sparse_matrix<
    sp_matrix_type>(rows, rows, nnz, row_size_variance, bandwidth, diag_dominance);

  KokkosSparse::sort_crs_matrix(A);

  // Report test config
  std::cout << "Testing par_ilut with rows=" << rows
            << "\n  nnz_per_row=" << nnz_per_row
            << "\n  bandwidth_per_nnz=" << bandwidth_per_nnz
            << "\n  total nnz=" << A.nnz()
            << "\n  league_size=" << default_policy.league_size()
            << "\n  team_size=" << default_policy.team_size()
            << "\n  concurrent teams=" << exe_space::concurrency() / default_policy.team_size()
            << "\n  loop=" << loop
            << std::endl;

  const auto num_iters = run_par_ilut_test(kh, A, rows, team_size, loop);

#ifdef USE_GINKGO
  run_par_ilut_test_ginkgo(kh, A, rows, team_size, loop, num_iters);
#endif

  run_spiluk_test(kh, A, rows, team_size, loop);

  return 0;
}

}

///////////////////////////////////////////////////////////////////////////////
void print_help_par_ilut()
///////////////////////////////////////////////////////////////////////////////
{
  printf("Options:\n");
  // printf(
  //     "  -f [file]       : Read in Matrix Market formatted text file. Not yet supported "
  //     "'file'.\n");
  printf("  -n [N]  : generate a semi-random banded NxN matrix. Default 10000.\n");
  printf("  -z [Z]  : number nnz per row. Default is 1% of N.\n");
  printf("  -b [B]  : bandwidth nnz multiplier. Default is 5.\n");
  printf("  -ts [T] : Number of threads per team. Default is 1 on OpenMP, nnz_per_row on CUDA\n");
  //printf("  -vl [V] : Vector-length (i.e. how many Cuda threads are a Kokkos 'thread').\n");
  printf("  -l [L]  : How many runs to aggregate average time. Default is 10\n\n");
}

///////////////////////////////////////////////////////////////////////////////
void handle_int_arg(int argc, char** argv, int& i, std::map<std::string, int*> option_map)
///////////////////////////////////////////////////////////////////////////////
{
  std::string arg = argv[i];
  auto it = option_map.find(arg);
  if (it == option_map.end()) {
    throw std::runtime_error(std::string("Unknown option: ") + arg);
  }
  if (i+1 == argc) {
    throw std::runtime_error(std::string("Missing option value for option: ") + arg);
  }
  *(it->second) = atoi(argv[++i]);
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
///////////////////////////////////////////////////////////////////////////////
{
  int rows          = 10000;
  int nnz_per_row   = -1; // depends on other options, so don't set to default yet
  int band_per_nnz  = 5;
  int team_size     = -1;
  int loop          = 10;

  std::map<std::string, int*> option_map = {
    {"-n" , &rows},
    {"-z" , &nnz_per_row},
    {"-b" , &band_per_nnz},
    {"-ts", &team_size},
    {"-l" , &loop}
  };

  if (argc == 1) {
    print_help_par_ilut();
    return 0;
  }

  // Handle user options
  for (int i = 1; i < argc; i++) {
    if ((strcmp(argv[i], "--help") == 0) || (strcmp(argv[i], "-h") == 0)) {
      print_help_par_ilut();
      return 0;
    }
    else {
      handle_int_arg(argc, argv, i, option_map);
    }
  }

  if (rows < 100) {
    throw std::runtime_error("Need to have at least 100 rows");
  }

  // Set dependent defaults
  if (nnz_per_row == -1) {
    nnz_per_row = rows / 100;
  }
  if (team_size == -1) {
    team_size = KokkosKernels::Impl::kk_is_gpu_exec_space<exe_space>() ? nnz_per_row : 1;
  }

  Kokkos::initialize(argc, argv);
  {
    test_par_ilut_perf(rows, nnz_per_row, band_per_nnz, team_size, loop);
  }
  Kokkos::finalize();
  return 0;
}
