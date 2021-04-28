//
// Created by Poliakoff, David Zoeller on 4/27/21.
//

#ifndef KOKKOSKERNELS_PERFTESTUTILITIES_HPP
#define KOKKOSKERNELS_PERFTESTUTILITIES_HPP
#include "KokkosKernels_default_types.hpp"
#include <common/KernelBase.hpp>
#include <common/QuickKernelBase.hpp>
namespace readers {

template <class Scalar, class Ordinal, class Offset>
using matrix_type =
    KokkosSparse::CrsMatrix<Scalar, Ordinal, Kokkos::DefaultExecutionSpace,
                            void, Offset>;

template <class>
struct test_reader;

template <class Scalar, class Ordinal, class Offset>
struct test_reader<matrix_type<Scalar, Ordinal, Offset>> {
  static matrix_type<Scalar, Ordinal, Offset> read(
      const std::string& filename) {
    return KokkosKernels::Impl::read_kokkos_crst_matrix<
        matrix_type<Scalar, Ordinal, Offset>>(filename.c_str());
  }
};

};  // namespace readers
template <class... SubComponents>
struct data_retriever {
  std::string root_path =
      "/Users/dzpolia/src/kokkos-kernels/perf_test/sparse/data/";
  std::string sub_path;
  struct test_case {
    std::string filename;
    std::tuple<SubComponents...> test_data;
  };
  std::vector<test_case> test_cases;
  std::string make_full_path_to_data_file(std::string repo,
                                          std::string path_to_data,
                                          std::string dataset,
                                          std::string filename) {
    return root_path + repo + path_to_data + dataset + filename;
  }
  template <class... Locations>
  data_retriever(std::string path_to_data, Locations... locations)
      : sub_path(path_to_data) {
    // TODO: way to list the directories in the root path
    std::vector<std::string> data_repos{"uur/"};
    // TODO: list directories in subpaths
    std::vector<std::string> datasets{"dataset_0/", "dataset_1/"};
    for (auto repo : data_repos) {
      for (auto dataset : datasets) {
        test_cases.push_back(
            test_case{repo + dataset,
                      std::make_tuple(readers::test_reader<SubComponents>::read(
                          make_full_path_to_data_file(
                              repo, path_to_data, dataset, locations))...)});
      }
    }
  }
};
using test_list = std::vector<KernelBase*>;

#endif  // KOKKOSKERNELS_PERFTESTUTILITIES_HPP
