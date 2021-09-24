# Source: https://devblogs.microsoft.com/cppblog/clear-functional-c-documentation-with-sphinx-breathe-doxygen-cmake/
# Author: Evan Harvey <eharvey@sandia.gov>
find_package(Doxygen REQUIRED)

# Find all the public headers
get_target_property(${PROJECT_NAME}_PUBLIC_HEADER_DIRS Kokkos::kokkoskernels INTERFACE_INCLUDE_DIRECTORIES)
foreach(dir ${${PROJECT_NAME}_PUBLIC_HEADER_DIRS})
    if (${dir} MATCHES "BUILD_INTERFACE")
        string(REPLACE "$<BUILD_INTERFACE:" "" new_dir "${dir}")
        string(REPLACE ">" "" dir "${new_dir}")
        file(GLOB_RECURSE ${PROJECT_NAME}_PUBLIC_HEADERS ${${PROJECT_NAME}_PUBLIC_HEADERS} ${dir}/*.hpp)
    endif()
endforeach()

#string(REPLACE ";" " " NEW_${PROJECT_NAME}_PUBLIC_HEADERS ${${PROJECT_NAME}_PUBLIC_HEADERS})

message(STATUS "headers = ${NEW_${PROJECT_NAME}_PUBLIC_HEADERS}")

set(DOXYGEN_INPUT_DIR ${PROJECT_SOURCE_DIR}/src)
set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/doxygen/html)
set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/index.html)
set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

#Replace variables inside @@ with the current values
configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT})

file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR}) #Doxygen won't create this for us

add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
        DEPENDS ${NEW_${PROJECT_NAME}_PUBLIC_HEADERS}
        COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        MAIN_DEPENDENCY Doxyfile
        COMMENT "Generating docs")

add_custom_target(Doxygen ALL DEPENDS ${DOXYGEN_INDEX_FILE})