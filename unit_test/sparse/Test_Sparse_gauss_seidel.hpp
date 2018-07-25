/*
//@HEADER
// ************************************************************************
//
//               KokkosKernels 0.9: Linear Algebra and Graph Kernels
//                 Copyright 2017 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
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


#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include "KokkosKernels_Handle.hpp"
#include "KokkosKernels_IOUtils.hpp"
//#include <Kokkos_Sparse_CrsMatrix.hpp>
#include <KokkosSparse_spmv.hpp>
#include <KokkosBlas1_dot.hpp>
#include <KokkosBlas1_axpby.hpp>
#include <KokkosBlas1_nrm2.hpp>
#include <cstdlib>
#include <iostream>
#include <complex>
#include "KokkosSparse_gauss_seidel.hpp"
#include "KokkosSparse_rcm_impl.hpp"

#ifndef kokkos_complex_double
#define kokkos_complex_double Kokkos::complex<double>
#define kokkos_complex_float Kokkos::complex<float>
#endif

using namespace KokkosKernels;
using namespace KokkosKernels::Experimental;
using namespace KokkosSparse;
using namespace KokkosSparse::Experimental;
namespace Test {

template <typename crsMat_t, typename device>
int run_gauss_seidel_1(
    crsMat_t input_mat,
    KokkosSparse::GSAlgorithm gs_algorithm,
    typename crsMat_t::values_type::non_const_type x_vector,
    typename crsMat_t::values_type::const_type y_vector,
    bool is_symmetric_graph,
    int apply_type = 0, // 0 for symmetric, 1 for forward, 2 for backward.
    bool skip_symbolic = false,
    bool skip_numeric = false,
    typename crsMat_t::value_type omega = Kokkos::Details::ArithTraits<typename crsMat_t::value_type>::one()
    ){
  typedef typename crsMat_t::StaticCrsGraphType graph_t;
  typedef typename graph_t::row_map_type lno_view_t;
  typedef typename graph_t::entries_type lno_nnz_view_t;
  typedef typename crsMat_t::values_type::non_const_type scalar_view_t;

  typedef typename lno_view_t::value_type size_type;
  typedef typename lno_nnz_view_t::value_type lno_t;
  typedef typename scalar_view_t::value_type scalar_t;

  typedef KokkosKernelsHandle
      <size_type,lno_t, scalar_t,
      typename device::execution_space, typename device::memory_space,typename device::memory_space > KernelHandle;



  KernelHandle kh;
  kh.set_team_work_size(16);
  kh.set_dynamic_scheduling(true);
  kh.create_gs_handle(gs_algorithm);


  const size_t num_rows_1 = input_mat.numRows();
  const size_t num_cols_1 = input_mat.numCols();
  const int apply_count = 100;

  if (!skip_symbolic){
    gauss_seidel_symbolic
      (&kh, num_rows_1, num_cols_1, input_mat.graph.row_map, input_mat.graph.entries, is_symmetric_graph);
  }

  if (!skip_numeric){
    gauss_seidel_numeric
    (&kh, num_rows_1, num_cols_1, input_mat.graph.row_map, input_mat.graph.entries, input_mat.values, is_symmetric_graph);
  }

  switch (apply_type){
  case 0:
    symmetric_gauss_seidel_apply
      (&kh, num_rows_1, num_cols_1, input_mat.graph.row_map, input_mat.graph.entries, input_mat.values, x_vector, y_vector,false, true, omega, apply_count);
    break;
  case 1:
    forward_sweep_gauss_seidel_apply
    (&kh, num_rows_1, num_cols_1, input_mat.graph.row_map, input_mat.graph.entries, input_mat.values, x_vector, y_vector,false, true, omega, apply_count);
    break;
  case 2:
    backward_sweep_gauss_seidel_apply
    (&kh, num_rows_1, num_cols_1, input_mat.graph.row_map, input_mat.graph.entries, input_mat.values, x_vector, y_vector,false, true, omega, apply_count);
    break;
  default:
    symmetric_gauss_seidel_apply
    (&kh, num_rows_1, num_cols_1, input_mat.graph.row_map, input_mat.graph.entries, input_mat.values, x_vector, y_vector,false, true, omega, apply_count);
    break;
  }


  kh.destroy_gs_handle();
  return 0;
}

template<typename scalar_view_t>
scalar_view_t create_x_vector(size_t nv, double max_value = 10.0){
  scalar_view_t kok_x ("X", nv);


  typename scalar_view_t::HostMirror h_x =  Kokkos::create_mirror_view (kok_x);


  for (size_t i = 0; i < nv; ++i){
    typename scalar_view_t::value_type r =
        static_cast <typename scalar_view_t::value_type> (rand()) /
        static_cast <typename scalar_view_t::value_type> (RAND_MAX / max_value);
    h_x(i) = r;
    //h_x(i) = 1;
  }
  Kokkos::deep_copy (kok_x, h_x);


  return kok_x;
}
template <typename crsMat_t, typename vector_t>
vector_t create_y_vector(crsMat_t crsMat, vector_t x_vector){
  vector_t y_vector ("Y VECTOR", crsMat.numRows());
  KokkosSparse::spmv("N", 1, crsMat, x_vector, 1, y_vector);
  return y_vector;
}

}

template <typename scalar_t, typename lno_t, typename size_type, typename device>
void test_gauss_seidel(lno_t numRows, size_type nnz, lno_t bandwidth, lno_t row_size_variance) {

  using namespace Test;
  srand(245);
  typedef typename KokkosSparse::CrsMatrix<scalar_t, lno_t, device, void, size_type> crsMat_t;
  //typedef typename crsMat_t::StaticCrsGraphType graph_t;
  //typedef typename graph_t::row_map_type lno_view_t;
  //typedef typename graph_t::entries_type lno_nnz_view_t;
  //typedef typename graph_t::entries_type::non_const_type   color_view_t;
  typedef typename crsMat_t::values_type::non_const_type scalar_view_t;

  lno_t numCols = numRows;
  crsMat_t input_mat = KokkosKernels::Impl::kk_generate_diagonally_dominant_sparse_matrix<crsMat_t>(numRows,numCols,nnz,row_size_variance, bandwidth);

  lno_t nv = input_mat.numRows();

  //KokkosKernels::Impl::print_1Dview(input_mat.graph.row_map);
  //KokkosKernels::Impl::print_1Dview(input_mat.graph.entries);
  //KokkosKernels::Impl::print_1Dview(input_mat.values);

  //scalar_view_t solution_x ("sol", nv);
  //Kokkos::Random_XorShift64_Pool<ExecutionSpace> g(1931);
  //Kokkos::fill_random(solution_x,g,Kokkos::Random_XorShift64_Pool<ExecutionSpace>::generator_type::MAX_URAND);

  const scalar_view_t solution_x = create_x_vector<scalar_view_t>(nv);
  scalar_view_t y_vector = create_y_vector(input_mat, solution_x);
#ifdef gauss_seidel_testmore
  GSAlgorithm gs_algorithms[] ={GS_DEFAULT, GS_TEAM, GS_PERMUTED};
  int apply_count = 3;
  for (int ii = 0; ii < 3; ++ii){
#else
  int apply_count = 1;
  GSAlgorithm gs_algorithms[] ={GS_DEFAULT};
  for (int ii = 0; ii < 1; ++ii){
#endif
    GSAlgorithm gs_algorithm = gs_algorithms[ii];
    scalar_view_t x_vector ("x vector", nv);
    const scalar_t alpha = 1.0;
    KokkosBlas::axpby(alpha, solution_x, -alpha, x_vector);
    scalar_t dot_product = KokkosBlas::dot( x_vector , x_vector );
    typedef typename Kokkos::Details::ArithTraits<scalar_t>::mag_type mag_t;
    mag_t initial_norm_res = Kokkos::Details::ArithTraits<scalar_t>::abs (dot_product);
    initial_norm_res  = Kokkos::Details::ArithTraits<mag_t>::sqrt( initial_norm_res );
    Kokkos::deep_copy (x_vector , 0);

    //bool is_symmetric_graph = false;
    //int apply_type = 0;
    //bool skip_symbolic = false;
    //bool skip_numeric = false;
    scalar_t omega = 0.9;

    for (int is_symmetric_graph = 0; is_symmetric_graph < 2; ++is_symmetric_graph){

      for (int apply_type = 0; apply_type < apply_count; ++apply_type){
        for (int skip_symbolic = 0; skip_symbolic < 2; ++skip_symbolic){
          for (int skip_numeric = 0; skip_numeric < 2; ++skip_numeric){

            Kokkos::Impl::Timer timer1;
            //int res =
            run_gauss_seidel_1<crsMat_t, device>(input_mat, gs_algorithm, x_vector, y_vector, is_symmetric_graph, apply_type, skip_symbolic, skip_numeric, omega);
            //double gs = timer1.seconds();

            //KokkosKernels::Impl::print_1Dview(x_vector);
            KokkosBlas::axpby(alpha, solution_x, -alpha, x_vector);
            //KokkosKernels::Impl::print_1Dview(x_vector);
            scalar_t result_dot_product = KokkosBlas::dot( x_vector , x_vector );
            mag_t result_norm_res  = Kokkos::Details::ArithTraits<scalar_t>::abs( result_dot_product );
            result_norm_res = Kokkos::Details::ArithTraits<mag_t>::sqrt(result_norm_res);
            //std::cout << "result_norm_res:" << result_norm_res << " initial_norm_res:" << initial_norm_res << std::endl;
            EXPECT_TRUE( (result_norm_res < initial_norm_res));
          }
        }
      }
    }
  }
  //device::execution_space::finalize();
}

template <typename scalar_t, typename lno_t, typename size_type, typename device>
void test_cluster_sgs(lno_t numRows, size_type nnz, lno_t bandwidth, lno_t row_size_variance)
{
  using namespace Test;
  typedef typename KokkosSparse::CrsMatrix<scalar_t, lno_t, device, void, size_type> crsMat_t;
  typedef typename crsMat_t::StaticCrsGraphType graph_t;
  typedef typename crsMat_t::values_type::non_const_type scalar_view_t;
  typedef typename graph_t::row_map_type lno_view_t;
  typedef typename graph_t::entries_type::non_const_type lno_nnz_view_t;
  typedef KokkosKernelsHandle
      <size_type, lno_t, scalar_t,
      typename device::execution_space, typename device::memory_space,typename device::memory_space> KernelHandle;
  crsMat_t A = KokkosKernels::Impl::kk_generate_diagonally_dominant_sparse_matrix<crsMat_t>(numRows,numRows,nnz,row_size_variance, bandwidth);
  //create a randomized RHS vector (b)
  scalar_view_t b("b", numRows);
  {
    Kokkos::View<scalar_t*, Kokkos::HostSpace> bHost("b (hosts)", numRows);
    for(lno_t i = 0; i < numRows; i++)
    {
      bHost(i) = (double) rand() / RAND_MAX;
    }
    Kokkos::deep_copy(b, bHost);
  }
  //create solution vector (x), zero initial guess
  //try a bunch of powers of 2 for cluster sizes, up to about the
  //point where there are very few clusters (8)
  std::vector<lno_t> clusterSizes;
  for(lno_t i = 1; i <= numRows / 8; i <<= 1)
    clusterSizes.push_back(i);
  const int niters = 5;
  for(size_t test = 0; test < clusterSizes.size(); test++)
  {
    //starting solution is zero vector
    scalar_view_t x("x", numRows);
    auto clusterSize = clusterSizes[test];
#ifdef CLUSTER_VERBOSE
    output << "Testing cluster size = " << clusterSize << '\n';
#endif
    KernelHandle kh;
    kh.create_gs_handle(clusterSize);
    //only need to do G-S setup (symbolic/numeric) once
    Kokkos::Impl::Timer timer;
    KokkosSparse::Experimental::gauss_seidel_symbolic<KernelHandle, lno_view_t, lno_nnz_view_t>
      (&kh, numRows, numRows, A.graph.row_map, A.graph.entries, false);
    KokkosSparse::Experimental::gauss_seidel_numeric<KernelHandle, lno_view_t, lno_nnz_view_t, scalar_view_t>
      (&kh, numRows, numRows, A.graph.row_map, A.graph.entries, A.values, false);
#ifdef CLUSTER_VERBOSE
    output << "Cluster size " << clusterSize << " setup time: " << timer.seconds() << " s\n";
#endif
    timer.reset();
    typedef Kokkos::Details::ArithTraits<scalar_t> KAT;
    scalar_t alpha = KAT::one();
    scalar_t beta = -KAT::one();
    KokkosSparse::Experimental::symmetric_gauss_seidel_apply
      <KernelHandle, lno_view_t, lno_nnz_view_t, scalar_view_t, scalar_view_t, scalar_view_t>
      (&kh, numRows, numRows, A.graph.row_map, A.graph.entries, A.values, x, b, false, true, 0.6, niters);
    scalar_view_t res("Ax-b", numRows);
    Kokkos::deep_copy(res, b);
    KokkosSparse::spmv<scalar_t, crsMat_t, scalar_view_t, scalar_t, scalar_view_t>
      ("N", alpha, A, x, beta, res, KokkosSparse::RANK_ONE());
#ifdef CLUSTER_VERBOSE
    double norm = KokkosBlas::nrm2(res);
    double bnorm = KokkosBlas::nrm2(b);
    output << "Cluster size " << clusterSize << " apply time: " << timer.seconds() << " s\n";
    output << "Cluster size " << clusterSize << " norm after " << niters << " sweeps: " << norm << '\n';
    output << "Cluster size " << clusterSize << " proportion of residual eliminated: " << 1.0 - (norm / bnorm) << std::endl;
#endif
  }
}

template <typename scalar_t, typename lno_t, typename size_type, typename device>
void test_rcm(lno_t numRows, size_type nnzPerRow, lno_t bandwidth)
{
  using namespace Test;
  typedef typename KokkosSparse::CrsMatrix<scalar_t, lno_t, device, void, size_type> crsMat_t;
  typedef typename crsMat_t::StaticCrsGraphType graph_t;
  typedef typename graph_t::entries_type::non_const_type lno_nnz_view_t;
  typedef KokkosKernelsHandle
      <size_type, lno_t, scalar_t,
      typename device::execution_space, typename device::memory_space,typename device::memory_space> KernelHandle;
  srand(245);
  size_type nnzTotal = nnzPerRow * numRows;
  lno_t nnzVariance = nnzPerRow / 4;
  crsMat_t A = KokkosKernels::Impl::kk_generate_sparse_matrix<crsMat_t>(numRows, numRows, nnzTotal, nnzVariance, bandwidth);
  typedef KokkosSparse::Impl::RCM<KernelHandle, typename graph_t::row_map_type, typename graph_t::entries_type> rcm_t;
  typename rcm_t::const_lno_row_view_t rowmap = A.graph.row_map;
  rcm_t rcm(numRows, A.graph.row_map, A.graph.entries);
  lno_nnz_view_t rcmOrder = rcm.rcm();
  //perm(i) = the node with timestamp i
  //make sure that perm is in fact a permutation matrix (contains each row exactly once)
  Kokkos::View<lno_t*, Kokkos::HostSpace> rcmHost("RCM row ordering", numRows);
  Kokkos::deep_copy(rcmHost, rcmOrder);
  std::set<lno_t> rowSet;
  for(lno_t i = 0; i < numRows; i++)
    rowSet.insert(rcmHost(i));
  if((lno_t) rowSet.size() != numRows)
  {
    std::cerr << "Only got back " << rowSet.size() << " unique row IDs!\n";
    return;
  }
  //make a new CRS graph based on permuting the rows and columns of mat
}

#define EXECUTE_TEST(SCALAR, ORDINAL, OFFSET, DEVICE) \
TEST_F( TestCategory, sparse ## _ ## gauss_seidel ## _ ## SCALAR ## _ ## ORDINAL ## _ ## OFFSET ## _ ## DEVICE ) { \
  test_gauss_seidel<SCALAR,ORDINAL,OFFSET,DEVICE>(10000, 10000 * 30, 200, 10); \
} \
TEST_F( TestCategory, sparse ## _ ## rcm ## _ ## SCALAR ## _ ## ORDINAL ## _ ## OFFSET ## _ ## DEVICE ) { \
  test_rcm<SCALAR,ORDINAL,OFFSET,DEVICE>(10000, 50, 2000); \
} \
TEST_F( TestCategory, sparse ## _ ## cluster_sgs ## _ ## SCALAR ## _ ## ORDINAL ## _ ## OFFSET ## _ ## DEVICE ) { \
  test_cluster_sgs<SCALAR,ORDINAL,OFFSET,DEVICE>(10000, 10000 * 30, 200, 10); \
}

#if (defined (KOKKOSKERNELS_INST_DOUBLE) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT) \
 && defined (KOKKOSKERNELS_INST_OFFSET_INT) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(double, int, int, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_DOUBLE) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT64_T) \
 && defined (KOKKOSKERNELS_INST_OFFSET_INT) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(double, int64_t, int, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_DOUBLE) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT) \
 && defined (KOKKOSKERNELS_INST_OFFSET_SIZE_T) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(double, int, size_t, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_DOUBLE) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT64_T) \
 && defined (KOKKOSKERNELS_INST_OFFSET_SIZE_T) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(double, int64_t, size_t, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_FLOAT) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT) \
 && defined (KOKKOSKERNELS_INST_OFFSET_INT) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(float, int, int, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_FLOAT) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT64_T) \
 && defined (KOKKOSKERNELS_INST_OFFSET_INT) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(float, int64_t, int, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_FLOAT) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT) \
 && defined (KOKKOSKERNELS_INST_OFFSET_SIZE_T) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(float, int, size_t, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_FLOAT) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT64_T) \
 && defined (KOKKOSKERNELS_INST_OFFSET_SIZE_T) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(float, int64_t, size_t, TestExecSpace)
#endif


#if (defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT) \
 && defined (KOKKOSKERNELS_INST_OFFSET_INT) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(kokkos_complex_double, int, int, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT64_T) \
 && defined (KOKKOSKERNELS_INST_OFFSET_INT) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(kokkos_complex_double, int64_t, int, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT) \
 && defined (KOKKOSKERNELS_INST_OFFSET_SIZE_T) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(kokkos_complex_double, int, size_t, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT64_T) \
 && defined (KOKKOSKERNELS_INST_OFFSET_SIZE_T) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(kokkos_complex_double, int64_t, size_t, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT) \
 && defined (KOKKOSKERNELS_INST_OFFSET_INT) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(kokkos_complex_float, int, int, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT64_T) \
 && defined (KOKKOSKERNELS_INST_OFFSET_INT) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(kokkos_complex_float, int64_t, int, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT) \
 && defined (KOKKOSKERNELS_INST_OFFSET_SIZE_T) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(kokkos_complex_float, int, size_t, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT64_T) \
 && defined (KOKKOSKERNELS_INST_OFFSET_SIZE_T) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(kokkos_complex_float, int64_t, size_t, TestExecSpace)
#endif




