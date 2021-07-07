# Fortran unit test framework
#
# http://sourceforge.net/projects/fortranxunit
# http://www.software.ac.uk/blog/2014-07-22-look-fortran-unit-test-frameworks
# 
# Modules will be installed into CMAKE_Fortran_MODULE_DIRECTORY
#  defined in CMakeBasics.cmake
 
blt_add_library(
    NAME fruit
    SOURCES fruit.f90
)
blt_register_library( NAME fruit
                      FORTRAN_MODULES ${CMAKE_Fortran_MODULE_DIRECTORY}
                      INCLUDES ${CMAKE_Fortran_MODULE_DIRECTORY}
                      LIBRARIES fruit 
)
