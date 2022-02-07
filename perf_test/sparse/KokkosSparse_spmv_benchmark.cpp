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

#include <vector>
#include <string>
#include <filesystem>  // For use with C++ 17
#include <fstream>
// Needed for `strftime` in timestamping function
#include <time.h>

#include <Kokkos_Core.hpp>

#include <KokkosSparse_spmv_test.hpp>

// Function to set up SPMV test
//
SPMVTestData setup_test(spmv_additional_data* data, SPMVTestData::matrix_type A,
                        Ordinal rows_per_thread, int team_size,
                        int vector_length, int schedule, int) {
  SPMVTestData test_data;
  using mv_type         = SPMVTestData::mv_type;
  using h_graph_type    = SPMVTestData::h_graph_type;
  using h_values_type   = SPMVTestData::h_values_type;
  test_data.A           = A;
  test_data.numRows     = A.numRows();
  test_data.numCols     = A.numCols();
  test_data.num_errors  = 0;
  test_data.total_error = 0;
  test_data.nnz         = A.nnz();
  mv_type x("X", test_data.numCols);
  mv_type y("Y", test_data.numRows);
  test_data.h_x         = Kokkos::create_mirror_view(x);
  test_data.h_y         = Kokkos::create_mirror_view(y);
  test_data.h_y_compare = Kokkos::create_mirror(y);

  h_graph_type h_graph   = Kokkos::create_mirror(test_data.A.graph);
  h_values_type h_values = Kokkos::create_mirror_view(test_data.A.values);

  for (int i = 0; i < test_data.numCols; i++) {
    test_data.h_x(i) = (Scalar)(1.0 * (rand() % 40) - 20.);
  }
  for (int i = 0; i < test_data.numRows; i++) {
    test_data.h_y(i) = (Scalar)(1.0 * (rand() % 40) - 20.);
  }

  test_data.generate_gold_standard(h_graph, h_values);

  Kokkos::deep_copy(x, test_data.h_x);
  Kokkos::deep_copy(y, test_data.h_y);
  Kokkos::deep_copy(test_data.A.graph.entries, h_graph.entries);
  Kokkos::deep_copy(test_data.A.values, h_values);
  test_data.x1 = mv_type("X1", test_data.numCols);
  Kokkos::deep_copy(test_data.x1, test_data.h_x);
  test_data.y1 = mv_type("Y1", test_data.numRows);

  // int nnz_per_row = A.nnz()/A.numRows(); // TODO: relocate
  matvec(A, test_data.x1, test_data.y1, rows_per_thread, team_size,
         vector_length, data, schedule);

  // Error Check
  Kokkos::deep_copy(test_data.h_y, test_data.y1);

  test_data.check_errors();
  test_data.min_time        = 1.0e32;
  test_data.ave_time        = 0.0;
  test_data.max_time        = 0.0;
  test_data.rows_per_thread = rows_per_thread;
  test_data.team_size       = team_size;
  test_data.vector_length   = vector_length;
  test_data.data            = data;
  test_data.schedule        = schedule;

  return test_data;
}

// Nota bene: `test` controls the number of test replicates

struct SPMVConfiguration {
  int test;
  // Launch parameters in Kokkos Kernels
  Ordinal rows_per_thread;
  Ordinal team_size;
  Ordinal vector_length;
  int schedule;
  int loop;
};

// CSV report generated for every run

inline void write_results_to_csv(std::string outfile_name, double seconds,
                                 std::string const& input_matrix) {
  std::fstream file;

  file.open(outfile_name, std::ios_base::app | std::ios_base::in);
  if (file.is_open()) file << input_matrix << "," << seconds << std::endl;
  std::cout << "Calling CSV writer for these input matrices:  " << input_matrix
            << std::endl;
}

// Uniqify csv reports with timestamp

inline std::string timestamp_now() {
  std::chrono::system_clock::time_point tp{std::chrono::system_clock::now()};
  std::time_t timeT{std::chrono::system_clock::to_time_t(tp)};

  struct tm tm;
  localtime_r(&timeT, &tm);

  // Allocation for timestamp format listed below
  std::string formatted(sizeof "2022-01-26T13:30:26", '\0');
  strftime(formatted.data(), formatted.size(), "%FT%T", &tm);
  formatted.pop_back();
  return formatted;
}

void benchmark_spmv_kernel(std::string matrix_file_name,
                           std::string output_filename) {
  spmv_additional_data data(KOKKOS);
  using matrix_type = typename SPMVTestData::matrix_type;

  matrix_type A = KokkosKernels::Impl::read_kokkos_crst_matrix<matrix_type>(
      matrix_file_name.c_str());

  //
  // Instantiate struct with configuration data
  //
  SPMVConfiguration config;

  config.test            = 100;
  config.rows_per_thread = 1;
  config.team_size       = 1;
  // SPMV uses 3 levels of parallelism;
  // Vectorization you'll get at the innermost level
  config.vector_length = 100;
  config.schedule      = 1;
  config.loop          = 10;

  auto test_data =
      setup_test(&data, A, config.rows_per_thread, config.team_size,
                 config.vector_length, config.schedule, config.loop);
  // From run_benchmark
  Kokkos::Timer timer;

  Kokkos::fence();
  double time = timer.seconds();
  // Test time will be an input for Apollo autotuning
  std::cout << "Benchmark time: " << time << std::endl;

  // Write results to file
  write_results_to_csv(output_filename, time, matrix_file_name);
}

int main(int argc, char** argv) {
  Kokkos::initialize();

  std::string timestamp = timestamp_now();

  std::string output_filename = "spmv_benchmark_" + timestamp + ".csv";

  // Input data directory / repo; matrices in the `SuiteSparseMatrix` are from
  // https://sparse.tamu.edu/
  std::string path = "/ascldap/users/ajpowel/SuiteSparseMatrix/";

  const std::string my_vect_mtx = ".mtx";
  std::vector<std::string> matrices_vect;

  // Recurse directories for matrix inputs
  for (const std::filesystem::directory_entry& dir_entry :
       std::filesystem::recursive_directory_iterator(path)) {
    if (dir_entry.path().extension().string() == my_vect_mtx) {
      std::cout << "Sparse matrices to be benchmarked: "
                << dir_entry.path().string() << std::endl;
      matrices_vect.push_back(dir_entry.path().string());
    }
  }

  // Call benchmarking function
  for (auto item : matrices_vect) {
    benchmark_spmv_kernel(item, output_filename);
  }

  Kokkos::finalize();
  return 0;
}
