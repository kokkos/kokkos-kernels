/*--------------------------------------------------------------------*/
/*    Copyright 2002 - 2008, 2010, 2011 National Technology &         */
/*    Engineering Solutions of Sandia, LLC (NTESS). Under the terms   */
/*    of Contract DE-NA0003525 with NTESS, there is a                 */
/*    non-exclusive license for use of this work by or on behalf      */
/*    of the U.S. Government.  Export of this program may require     */
/*    a license from the United States Government.                    */
/*--------------------------------------------------------------------*/
#ifndef TFTK_TFTK_UTIL_TFTK_UTIL_TFTK_KOKKOSTYPES_H_
#define TFTK_TFTK_UTIL_TFTK_UTIL_TFTK_KOKKOSTYPES_H_

#include <Kokkos_Core.hpp>
//#include <tftk_util/tftk_Macros.h>
#include <tftk_Macros.h>

namespace KokkosBatched {

using HostSpace = Kokkos::HostSpace;

#ifdef TFTK_ENABLE_CUDA
using DeviceSpace     = Kokkos::CudaSpace;
using DeviceExecSpace = Kokkos::Cuda;
#else
using DeviceSpace     = Kokkos::DefaultExecutionSpace::memory_space;
using DeviceExecSpace = Kokkos::DefaultExecutionSpace;
#endif

#ifdef TFTK_ENABLE_CUDA
using UVMSpace = Kokkos::CudaUVMSpace;
#else
using UVMSpace        = Kokkos::HostSpace;
#endif

using UVMSpaceHostExec = Kokkos::Device<HostSpace::execution_space, UVMSpace>;
using AnonSpace        = Kokkos::AnonymousSpace;

#ifdef TFTK_ENABLE_CUDA
// MTP: As of July 1, 2021, using Kokkos::OpenMP with
// Kokkos::Experimental::UniqueToken<Kokkos::OpenMP, Instance> yields a nasty
// compiler error error #2985: identifier "Kokkos::Impl::t_openmp_instance" is
// undefined in device code
// using HostOnlyKokkosDevice = Kokkos::Device<Kokkos::Serial,
// Kokkos::HostSpace>;
using HostOnlyKokkosDevice =
    Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace>;
#else
using HostOnlyKokkosDevice =
    Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace>;
#endif

using DefaultKokkosDevice = Kokkos::Device<DeviceExecSpace, DeviceSpace>;

// GenericView should be copy/move constructible from a View with any layout in
// any memory space. It cannot be directly constructed or have deep_copy called
// on it since it does not have a true memory space, and there is a slight
// performance penalty from using LayoutStride, but it can be very useful for
// defining function arguments for functions that want to accept View's of
// various memory spaces and layouts where avoiding the compile time cost that
// templating on those parameters would require is of more value than the modest
// runtime cost of using a strided layout view.
template <typename T>
using GenericView =
    Kokkos::View<T, Kokkos::LayoutStride, AnonSpace, Kokkos::MemoryUnmanaged>;

template <class T>
using Array = Kokkos::View<T *, Kokkos::LayoutRight, UVMSpace>;

template <class T>
using Array2d = Kokkos::View<T **, Kokkos::LayoutRight, UVMSpace>;

template <class T>
using Array3d = Kokkos::View<T ***, Kokkos::LayoutRight, UVMSpace>;
}  // namespace KokkosBatched

#endif /* TFTK_TFTK_UTIL_TFTK_UTIL_TFTK_KOKKOSTYPES_H_ */
