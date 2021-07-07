

// GET HEADING
//
#include <Kokkos_Core.hpp>
#include <KokkosBlas1_dot.hpp>
#include <Kokkos_Random.hpp>
// For RPS implementation
#include "KokkosBlas_dot_perf_test.hpp"

// Recall -- testData is a tempated class, 
// setup_test is a templated function
template<class ExecSpace, class Layout>
testData<ExecSpace, Layout> setup_test(int m,
                    int repeat
                    )
{
        // use constructor to generate test data
        testData<ExecSpace, Layout> testData_obj(m);

        // set a field in the struct
        testData_obj.m = m;
        testData_obj.repeat = repeat;

        return testData_obj;
}


test_list construct_dot_kernel_base(const rajaperf::RunParams& run_params)

{
        // instantiate test_list as kernel_base_vector
        test_list kernel_base_vector;


kernel_base_vector.push_back(rajaperf::make_kernel_base(
        "BLAS_DOT ",
        run_params,
        [=](const int repeat, const int m) {
          // returns a tuple of testData_obj
          return std::make_tuple(
                          setup_test<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::array_layout>(m, repeat));
          },
        [&](const int iteration, const int runsize, auto& data) {
        KokkosBlas::dot(data.x, data.y);
        }));


        // return a vector of kernel base objects
        // of type test_list
        return kernel_base_vector;
}



