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

struct RunPerfTest {

  // Build up useful types
  using scalar_t  = default_scalar;
  using lno_t     = default_lno_t;
  using size_type = default_size_type;
  using exe_space = Kokkos::DefaultExecutionSpace;
  using mem_space = typename exe_space::memory_space;

  using RowMapType  = Kokkos::View<size_type*, device>;
  using EntriesType = Kokkos::View<lno_t*, device>;
  using ValuesType  = Kokkos::View<scalar_t*, device>;

  using device = Kokkos::Device<exe_space, mem_space>;

  using sp_matrix_type =
      KokkosSparse::CrsMatrix<scalar_t, lno_t, device, void, size_type>;
  using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<
      size_type, lno_t, scalar_t, exe_space, mem_space, mem_space>;
  using float_t = typename Kokkos::ArithTraits<scalar_t>::mag_type;

  /////////////////////////////////////////////////////////////////////////////
  void run_par_ilut_test(KernelHandle& kh, const sp_matrix_type& A, int rows, int team_size, int loop)
  /////////////////////////////////////////////////////////////////////////////
  {
    auto par_ilut_handle = kh.get_par_ilut_handle();

    // Pull out views from CRS
    auto A_row_map = A.graph.row_map;
    auto A_entries = A.graph.entries;
    auto A_values  = A.values;

    // Allocate L and U CRS views as outputs
    RowMapType L_row_map("L_row_map", numRows + 1);
    RowMapType U_row_map("U_row_map", numRows + 1);
    RowMapType L_row_map_orig("L_row_map", numRows + 1);
    RowMapType U_row_map_orig("U_row_map", numRows + 1);

    // Initial L/U approximations for A
    par_ilut_symbolic(&kh, row_map, entries, L_row_map, U_row_map);

    const size_type nnzL = par_ilut_handle->get_nnzL();
    const size_type nnzU = par_ilut_handle->get_nnzU();

    EntriesType L_entries("L_entries", nnzL);
    ValuesType L_values("L_values", nnzL);
    EntriesType U_entries("U_entries", nnzU);
    ValuesType U_values("U_values", nnzU);

    Kokkos::Timer timer;
    double min_time = std::numeric_limits<double>::infinity();
    double max_time = 0.0;
    double ave_time = 0.0;
    for (int i = 0; i < loop; ++i) {
      timer.reset();
      par_ilut_numeric(&kh, A_row_map, A_entries, A_values, L_row_map, L_entries,
                       L_values, U_row_map, U_entries, U_values);
      Kokkos::fence();
      double time = timer.seconds();
      ave_time += time;
      if (time > max_time) max_time = time;
      if (time < min_time) min_time = time;
    }

    std::cout << "PAR_ILUT LOOP_AVG_TIME:  " << ave_time / loop << std::endl;
    std::cout << "PAR_ILUT LOOP_MAX_TIME:  " << max_time << std::endl;
    std::cout << "PAR_ILUT LOOP_MIN_TIME:  " << min_time << std::endl;
  }

#ifdef USE_GINKGO
  /////////////////////////////////////////////////////////////////////////////
  void run_par_ilut_test_ginkgo(const sp_matrix_type& A, int rows, int team_size, int loop)
  /////////////////////////////////////////////////////////////////////////////
  {
    // Use ginko!

    using mtx = gko::matrix::Csr<scalar_t, lno_t>;
    auto exec = gko::OmpExecutor::create();

    EntriesType A_row_map_cp("A_row_map_cp", numRows+1);
    for (size_type i = 0; i < row_map.extent(0); ++i) {
      A_row_map_cp(i) = row_map(i);
    }

    // Populate mtx
    auto a_mtx_uniq = mtx::create_const(exec, gko::dim<2>(n, n),
                                        gko::array<scalar_t>::const_view(exec, values.extent(0), values.data()),
                                        gko::array<lno_t>::const_view(exec, entries.extent(0), entries.data()),
                                        gko::array<lno_t>::const_view(exec, A_row_map_cp.extent(0), A_row_map_cp.data()));

    std::shared_ptr<const mtx> a_mtx = std::move(a_mtx_uniq);

    auto fact = gko::factorization::ParIlut<scalar_t, lno_t>::build()
      .with_fill_in_limit(par_ilut_handle->get_fill_in_limit())
      .with_approximate_select(false)
      .on(exec)->generate(a_mtx);
  }
#endif

  /////////////////////////////////////////////////////////////////////////////
  int test_par_ilut_perf(int rows, int nnz_per_row, float bandwidth_per_nnz,
                         int team_size, int loop)
  /////////////////////////////////////////////////////////////////////////////
  {
    // Generate A
    const size_type nnz   = rows * nnz_per_row;
    const lno_t bandwidth = nnz_per_row * bandwidth_per_nnz;
    const lno_t row_size_variance = 0;
    const scalar_t diag_dominance = 1;
    auto A = KokkosSparse::Impl::kk_generate_diagonally_dominant_sparse_matrix<
      sp_matrix_type>(rows, rows, nnz, row_size_variance, bandwidth, diag_dominance);

    KokkosSparse::sort_crs_matrix(A);

    KernelHandle kh;
    kh.create_par_ilut_handle();

    run_par_ilut_test(kh, A, rows, team_size, loop);

#ifdef USE_GINKGO
    run_par_ilut_test_ginkgo(A, rows, team_size, loop);
#endif

    return 0;
  }

};

void print_help_par_ilut() {
  printf("Options:\n");
  printf("  --test [OPTION] : Use different kernel implementations\n");
  printf("                    Options:\n");
  printf("                      lvlrp, lvltp1, lvltp2\n\n");
  printf(
      "  -f [file]       : Read in Matrix Market formatted text file "
      "'file'.\n");
  //  printf("  -s [N]          : generate a semi-random banded (band size
  //  0.01xN) NxN matrix\n"); printf("                    with average of 10
  //  entries per row.\n"); printf("  --schedule [SCH]: Set schedule for kk
  //  variant (static,dynamic,auto [ default ]).\n"); printf("  -afb [file] :
  //  Read in binary Matrix files 'file'.\n"); printf("  --write-binary  : In
  //  combination with -f, generate binary files.\n"); printf("  --offset [O] :
  //  Subtract O from every index.\n"); printf("                    Useful in
  //  case the matrix market file is not 0 based.\n\n");
  printf("  -k [K]          : Fill level (default: 0)\n");
  printf("  -ts [T]         : Number of threads per team.\n");
  printf(
      "  -vl [V]         : Vector-length (i.e. how many Cuda threads are a "
      "Kokkos 'thread').\n");
  printf(
      "  --loop [LOOP]   : How many runs to aggregate average time. "
      "\n");
}

int main(int argc, char **argv) {
  std::vector<int> tests;

  std::string afilename;

  int kin           = 0;
  int vector_length = -1;
  int team_size     = -1;
  // int idx_offset = 0;
  int loop = 1;
  // int schedule=AUTO;

  if (argc == 1) {
    print_help_par_ilut();
    return 0;
  }

  for (int i = 0; i < argc; i++) {
    if ((strcmp(argv[i], "--test") == 0)) {
      i++;
      if ((strcmp(argv[i], "lvlrp") == 0)) {
        tests.push_back(LVLSCHED_RP);
      }
      if ((strcmp(argv[i], "lvltp1") == 0)) {
        tests.push_back(LVLSCHED_TP1);
      }
      /*
            if((strcmp(argv[i],"lvltp2")==0)) {
              tests.push_back( LVLSCHED_TP2 );
            }
      */
      continue;
    }
    if ((strcmp(argv[i], "-f") == 0)) {
      afilename = argv[++i];
      continue;
    }
    if ((strcmp(argv[i], "-k") == 0)) {
      kin = atoi(argv[++i]);
      continue;
    }
    if ((strcmp(argv[i], "-ts") == 0)) {
      team_size = atoi(argv[++i]);
      continue;
    }
    if ((strcmp(argv[i], "-vl") == 0)) {
      vector_length = atoi(argv[++i]);
      continue;
    }
    // if((strcmp(argv[i],"--offset")==0)) {idx_offset = atoi(argv[++i]);
    // continue;}
    if ((strcmp(argv[i], "--loop") == 0)) {
      loop = atoi(argv[++i]);
      continue;
    }
    /*
        if((strcmp(argv[i],"-afb")==0)) {afilename = argv[++i]; binaryfile =
       true; continue;} if((strcmp(argv[i],"--schedule")==0)) { i++;
          if((strcmp(argv[i],"auto")==0))
            schedule = AUTO;
          if((strcmp(argv[i],"dynamic")==0))
            schedule = DYNAMIC;
          if((strcmp(argv[i],"static")==0))
            schedule = STATIC;
          continue;
        }
    */
    if ((strcmp(argv[i], "--help") == 0) || (strcmp(argv[i], "-h") == 0)) {
      print_help_par_ilut();
      return 0;
    }
  }

  if (tests.size() == 0) {
    tests.push_back(DEFAULT);
  }
  for (size_t i = 0; i < tests.size(); ++i) {
    std::cout << "tests[" << i << "] = " << tests[i] << std::endl;
  }

  Kokkos::initialize(argc, argv);
  {
    int total_errors = test_par_perf(tests, afilename, kin, team_size,
                                        vector_length, /*idx_offset,*/ loop);

    if (total_errors == 0)
      printf("Kokkos::PAR_ILUT Test: Passed\n");
    else
      printf("Kokkos::PAR_ILUT Test: Failed\n");
  }
  Kokkos::finalize();
  return 0;
}
