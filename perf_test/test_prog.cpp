#include <iostream>
#include "KokkosBlas1_dot.hpp"
#include "KokkosKernels_Utils.hpp"
#include <chrono>

// in test_common
//#include "KokkosKernels_TestUtils.hpp"

/*


case Kokkos_Lambda : {

                          // open Kokkosfence
                          Kokkos::fence();
                          startTimer();

                          for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
                                   // Declare and initialize dot
                                   // dot will contain the reduction value,
                                   // i.e., the dot product
                                   //
                                   // Reductions combine contributions from
                                   // loop iterations
                                   Real_type dot = m_dot_init;

                                   parallel_reduce("DOT-Kokkos Kokkos_Lambda",
                                                  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
                                                  KOKKOS_LAMBDA(const int64_t i, Real_type& dot_res){

                                                  // DOT BODY definition from header:
                                                  //   dot += a[i] * b[i] ;
                                                  //dot_res += a_view[i]*b_view[i];
                                                  ///////////////////////////////
                                                  //Int_type vec_i = vec_view[i];
                                                  dot_res += a_view[i]*b_view[i];
                                                  //dot_res = vec_i;
                                                  }, dot);
                                  m_dot += static_cast<Real_type>(dot);
                          }
*/

// The BLAS dot replaces the parallel_reduce
//
// https://github.com/kokkos/kokkos-kernels/wiki/BLAS-1%3A%3Adot
// x & y must be Kokkos Views with the same number of lengths / dimensions
// 1) create views of vector inputs
// Passing vec -- these were moved to main()
//Test::create_random_x_vector(vec);
// Test::create_random_x_vector(vec_2D);


// 1D dot product
// x and y are vectors as views; x and y are always single column in the
// example below, i.e., only rank 1 vectors can be passed in
// result = KokkosBlas::dot(x,y); 
// 2D dot product; this case would be double **
// KokkosBlas::dot(r,x,y);

// FUNCTION THAT IS PART OF KK for generating test matrices
  //create_random_x_vector and create_random_y_vector can be used together to generate a random 
  //linear system Ax = y.
  template<typename vec_t>
  vec_t create_random_x_vector(vec_t& kok_x, double max_value = 10.0) {
    typedef typename vec_t::value_type scalar_t;
    auto h_x = Kokkos::create_mirror_view (kok_x);
    for (size_t j = 0; j < h_x.extent(1); ++j){
      for (size_t i = 0; i < h_x.extent(0); ++i){
        scalar_t r =
            static_cast <scalar_t> (rand()) /
            static_cast <scalar_t> (RAND_MAX / max_value);
        h_x.access(i, j) = r;
      }
    }
    Kokkos::deep_copy (kok_x, h_x);
    return kok_x;
  }


int main() {

Kokkos::initialize();

// Always scope Kokkos allocations with curly braces!

{
// Allocating 1D vec 
Kokkos::View<double*> vec("My 1D vector", 1000);

// Allocating 2D vec
Kokkos::View<double**> vec_2D("My 2D vector", 1000, 100);
// Allocating a view to contain 2D results
Kokkos::View<double*> results_2D("My 2D vector results", 100);

// Passing vec
///Test::create_random_x_vector(vec);
create_random_x_vector(vec);

//Test::create_random_x_vector(vec_2D);
//create_random_x_vector(vec_2D);

Kokkos::fence();
// create an instance of Kokkos Timer
Kokkos::Timer timer;
// Get starting time:
std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
timer.reset();

// Dot product of 1D vectors!
double result_1D = KokkosBlas::dot(vec, vec);

double elapsed = timer.seconds();

Kokkos::fence();
// Get end time:
std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();



// Results are stored in a view , i.e., the r argument;
// r = make 1D view called results;
//KokkosBlas::dot(100,vec_2D,vec_2D);
// KokkosBlas::dot(results_2D,vec_2D,vec_2D);
// Need to get results that are in a view 
//
auto result_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), results_2D);

std::cout << " 1D dot product:  " << result_1D << std::endl;

std::cout << "Test elapsed time with chrono = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

std::cout << " Test elapsed time with Kokkos Timer =  " << elapsed << std::endl; 


// Won't work b/c it's a View!
// std::cout << " 2D dot product:  " << result_host << std::endl;

//std::cout << " 2D dot product:  " << std::endl;
//KokkosKernels::Impl::print_1Dview(std::cout, results_2D, true);
//KokkosKernels::Impl::print_1Dview(std::cout, results_2D);
}

Kokkos::finalize();

return  0;

}
