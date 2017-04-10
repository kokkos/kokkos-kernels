#ifndef KOKKOSKERNELS_CONFIG_H
#define KOKKOSKERNELS_CONFIG_H

/* Define this macro if the quadmath TPL is enabled */
/* #undef HAVE_KOKKOSKERNELS_QUADMATH */

/* Define this macro if the MKL TPL is enabled.  This is different
   than just linking against the MKL to get the BLAS and LAPACK; it
   requires (a) header file(s) as well, and may use functions other
   than just BLAS and LAPACK functions.  */
//#define HAVE_KOKKOSKERNELS_MKL

/* Define this macro if experimental features of Kokkoskernels are enabled */
#define HAVE_KOKKOSKERNELS_EXPERIMENTAL

/* Define this macro to disallow instantiations of kernels which are not covered by ETI */
/* #undef KOKKOSKERNELS_ETI_ONLY */

/* Whether to build kernels for execution space Kokkos::Cuda */
/* #undef KOKKOSKERNELS_INST_EXECSPACE_CUDA */
/* #undef KOKKOSKERNELS_INST_MEMSPACE_CUDASPACE */
/* #undef KOKKOSKERNELS_INST_MEMSPACE_CUDAUVMSPACE */
/* Whether to build kernels for execution space Kokkos::OpenMP */
#define KOKKOSKERNELS_INST_EXECSPACE_OPENMP
/* Whether to build kernels for execution space Kokkos::Threads */
/* #undef KOKKOSKERNELS_INST_EXECSPACE_PTHREAD */
/* Whether to build kernels for execution space Kokkos::Serial */
#define KOKKOSKERNELS_INST_EXECSPACE_SERIAL

/* Whether to build kernels for memory space Kokkos::HostSpace */
#define KOKKOSKERNELS_INST_MEMSPACE_HOSTSPACE


/* Whether to build kernels for scalar type double */
#define KOKKOSKERNELS_INST_DOUBLE
/* Whether to build kernels for scalar type float */
/* #undef KOKKOSKERNELS_INST_FLOAT */
/* Whether to build kernels for scalar type complex<double> */
/* #undef KOKKOSKERNELS_INST_COMPLEX_DOUBLE */
/* Whether to build kernels for scalar type complex<float> */
/* #undef KOKKOSKERNELS_INST_COMPLEX_FLOAT */
#if defined KOKKOSKERNELS_INST_COMPLEX_DOUBLE
#define KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_
#endif
#if defined KOKKOSKERNELS_INST_COMPLEX_FLOAT
#define KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_
#endif

/* Whether to build kernels for multivectors of LayoutLeft */
#define KOKKOSKERNELS_INST_LAYOUTLEFT
/* Whether to build kernels for multivectors of LayoutRight */
/* #undef KOKKOSKERNELS_INST_LAYOUTRIGHT */
/*
 * "Optimization level" for computational kernels in this subpackage.
 * The higher the level, the more code variants get generated, and
 * thus the longer the compile times.  However, more code variants
 * mean both better performance overall, and more uniform performance
 * for corner cases.
 */
#define KOKKOSLINALG_OPT_LEVEL 1

#endif // KOKKOSKERNELS_CONFIG_H
