kokkoskernels_include_directories(${CMAKE_CURRENT_BINARY_DIR})
kokkoskernels_include_directories(${CMAKE_CURRENT_SOURCE_DIR})

kokkoskernels_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../../test_common)

kokkoskernels_add_executable_and_test(wiki_lapack_potr SOURCES KokkosLapack_wiki_potr.cpp)
