# Build google test
if (NOT TARGET gtest)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  add_subdirectory("${PROJECT_SOURCE_DIR}/extern/googletest" "extern/googletest"
    EXCLUDE_FROM_ALL)

  mark_as_advanced(
    BUILD_GMOCK BUILD_GTEST BUILD_SHARED_LIBS
    gmock_build_tests gtest_build_samples gtest_build_tests
    gtest_disable_pthreads gtest_force_shared_crt gtest_hide_internal_symbols
    )
  set_target_properties(gtest PROPERTIES FOLDER extern)
  set_target_properties(gtest_main PROPERTIES FOLDER extern)
  set_target_properties(gmock PROPERTIES FOLDER extern)
  set_target_properties(gmock_main PROPERTIES FOLDER extern)
endif (NOT TARGET gtest)

if(ENABLE_CUDA)
  # Work around -pthread argument to nvcc propagating from thread
  # detection in gtest
  set(gtest_disable_pthreads On)
endif()

function(camp_add_gtest TESTNAME)
  add_executable(${TESTNAME} ${ARGN})
  target_link_libraries(${TESTNAME} camp gtest gmock gtest_main)
  add_test(${TESTNAME} ${TESTNAME})
  set_target_properties(${TESTNAME} PROPERTIES FOLDER test)
endfunction()

function(camp_add_test TESTNAME)
  cmake_parse_arguments(ABT "GTEST;RUN" "" "BUILD;TEST" ${ARGN})


  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${TESTNAME}.cpp")
    list(APPEND ABT_BUILD "${CMAKE_CURRENT_SOURCE_DIR}/${TESTNAME}.cpp")
    list(REMOVE_DUPLICATES ABT_BUILD)
  endif()


  # target that is run by the test, build and run
  if(ABT_GTEST OR ABT_RUN)
    # Actual build setup, do not build by default
    add_executable(${TESTNAME} EXCLUDE_FROM_ALL ${ABT_BUILD})
    add_custom_target(${TESTNAME}.runner
      COMMAND ${CMAKE_COMMAND} --build . --target ${TESTNAME} --config ${CMAKE_BUILD_TYPE}
      COMMAND "$<TARGET_FILE:${TESTNAME}>"
      WORKING_DIRECTORY ${CURRENT_BINARY_DIR}
      )
  else()
    # Actual build setup, do not build by default, use a lib so no main required
    add_library(${TESTNAME} EXCLUDE_FROM_ALL ${ABT_BUILD})
    add_custom_target(${TESTNAME}.runner
      COMMAND ${CMAKE_COMMAND} --build . --target ${TESTNAME} --config ${CMAKE_BUILD_TYPE}
      WORKING_DIRECTORY ${CURRENT_BINARY_DIR}
      )
  endif()

  # Add interface library for includes
  target_link_libraries(${TESTNAME} camp)

  if(ENABLE_CUDA)
    set(source_files ${ABT_BUILD})
    list(FILTER source_files INCLUDE REGEX ".*\.cpp$")
    if(NOT ENABLE_CLANG_CUDA)
      set_source_files_properties( ${source_files} PROPERTIES LANGUAGE CUDA )
      set_target_properties( ${TESTNAME} PROPERTIES LINKER_LANGUAGE CUDA)
    else()
      if(BLT_LOADED)
        target_link_libraries(${TESTNAME} cuda)
      else()
        target_link_libraries(${TESTNAME} camp_cuda)
      endif()
    endif()
  endif()


  if(ABT_GTEST)
    target_link_libraries(${TESTNAME} gtest gtest_main)
  endif()

  # The test itself, invoke the runner
  add_test(NAME ${TESTNAME}
    COMMAND ${CMAKE_COMMAND} --build . --target ${TESTNAME}.runner
    WORKING_DIRECTORY ${CURRENT_BINARY_DIR}
    ${TEST})
  set_target_properties(${TESTNAME} PROPERTIES FOLDER test)

  list(GET ABT_BUILD 0 source_file)
  if(NOT source_file)
    return()
  endif()
  set(test_prop_regex  ".*(WILL_FAIL|PASS_REGEX|FAIL_REGEX):?[ ]*(.*)[ ]*")
  file(STRINGS ${source_file} test_props REGEX "${test_prop_regex}")
  foreach(line ${test_props})

    if(NOT line MATCHES "${test_prop_regex}")
      continue()
    endif()
    if(CMAKE_MATCH_1 STREQUAL "WILL_FAIL")
      set_tests_properties(${TESTNAME} PROPERTIES WILL_FAIL)
    elseif(CMAKE_MATCH_1 STREQUAL "PASS_REGEX")
      set_tests_properties(${TESTNAME} PROPERTIES
        PASS_REGULAR_EXPRESSION "${CMAKE_MATCH_2}")
    elseif(CMAKE_MATCH_1 STREQUAL "FAIL_REGEX")
      set_tests_properties(${TESTNAME} PROPERTIES
        FAIL_REGULAR_EXPRESSION "${CMAKE_MATCH_2}")
    endif()
  endforeach()
endfunction()

camp_add_test(resource GTEST)
camp_add_test(tuple GTEST)

camp_add_test(tuple_out_of_range RUN)

camp_add_test(accumulate)
camp_add_test(at_key)
camp_add_test(filter)
camp_add_test(find_if)
camp_add_test(flatten)
camp_add_test(index_of)
camp_add_test(lambda)
camp_add_test(number)
camp_add_test(size)
camp_add_test(transform)
