//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Tyoes and methods for managing Suite kernels and variants.
///

#ifndef RAJAPerfSuite_HPP
#define RAJAPerfSuite_HPP

#include "common/RPTypes.hpp"
#ifndef RAJAPERF_INFRASTRUCTURE_ONLY
#include "RAJA/config.hpp"
#include "common/PerfsuiteKernelDefinitions.hpp"
#if defined(RUN_KOKKOS)
#include "Kokkos_Core.hpp"
#endif
#endif

#include <string>
namespace rajaperf
{

class Executor; // forward declaration
class KernelBase;

void free_register_group(Executor*, std::string); // forward declaration
void free_register_kernel(Executor*, std::string, KernelBase*); // forward declaration
#if defined(RUN_KOKKOS)

// Kokkos Design Spirit:
// WE NEED:
// 1) Use KokkosViews --> a wrapper around pointers for host and device memory
// management
// 2) Use default execution space
//
//
//
//
// NEW FUNCTION WILL:
// 1) Take in a raw pointer (e.g., float*, int*, etc.)
// 2) From this pointer, return a Kokkos::View
//
// Return type :  Kokkos::View
// Kokkos::View takes tempalted arguments
// To write "generically" implies templated arguments
// https://eli.thegreenplace.net/2014/variadic-templates-in-c/
//
template <class PointedAt, size_t NumBoundaries>

// This is a TEMPLATED STRUCT.  This struct will contain the type of a pointer
// of n dimensions This struct is templated on the template<class PointedAt,
// size_t NumBoundaries> that immediately precedes the struct declaration.
struct PointerOfNdimensions;

// This template block declares a specialization, which means that you say the
// template arguments that you're NOT specializing
template <class PointedAt>

// Here, we are specialising a template according to the type of argument that
// is passed.  In this example, we've specialized the PointedAt template
// argument for the case that the number of dimensions is 0.  All we will do in
// this struct is to define a type.

// This struct is a specialization of :
// template<class PointedAt, size_t NumBoundaries>
struct PointerOfNdimensions<PointedAt, 0> {
  // "using" is a type alias
  // if you derefernce a pointer, you're just left with an object, the value
  // of that pointer
  using type = PointedAt;
};

// NO SPECIALIZATION, i.e., we fix no templated arguments
template <class PointedAt, size_t NumBoundaries>

struct PointerOfNdimensions {
  // PointerOfNdimensions is a type
  // My type is a pointer to the type of myself, decremented
  using type =
      typename PointerOfNdimensions<PointedAt, NumBoundaries - 1>::type *;
};

template <class PointedAt, class... Boundaries>

// FUNCTION THAT GETS A VIEW FROM A POINTER WITH RETURN TYPE KOKKOS::VIEW
//


auto getViewFromPointer(PointedAt *kokkos_ptr, Boundaries... boundaries)
    // Recall:  PointerOfNdimensions is struct that exists solely to hold a
    // type
    // -> connotes "return type after the arrow"
    -> typename Kokkos::View<
        typename PointerOfNdimensions<PointedAt, sizeof...(Boundaries)>::type,
        // typename Kokkos::DefaultHostExecutionSpace::memory_space>
        // This more generic expression allow moving the
        // View-wrapped pointer b/w
        // Host and GPU
        typename Kokkos::DefaultExecutionSpace::memory_space>

{
  // This says construct the pointer_holder variable from arguments passed to
  // the template block
  // Declaration of a type alias, host_view_type

  using host_view_type = typename Kokkos::View<
        // in the line below , you are using the type alias that is the memeber
        // of a struct

      typename PointerOfNdimensions<PointedAt, sizeof...(Boundaries)>::type,
      typename Kokkos::DefaultHostExecutionSpace::memory_space>;

  // FYI - Device can be GPU, OpenMPTarget, HIP (for targeting an AMD GPU), SYCL
  // (library in Intel Compiler)
  //
  using device_view_type = typename Kokkos::View<
      typename PointerOfNdimensions<PointedAt, sizeof...(Boundaries)>::type,
      typename Kokkos::DefaultExecutionSpace::memory_space>;

  // When copying data, we can either change the Layout or the memory_space
  // (host or device), but we cannot change both!
  // Here, we are mirroring data on the host to the device, i.e., Layout is
  // as if on the device, but the data is actually on the host.  The host
  // mirror will be Layout Left (optimal for the device), but data are
  // actually on the HOST!

  // Here, "using" is type alias; in this example,its our gpu Layout on cpu
  using mirror_view_type = typename device_view_type::HostMirror;

  // Assignment statement; we are constructing a host_view_type with the name
  // pointer_holder.  The value of kokkos_ptr is the pointer we're wrapping on
  // the Host, and the Boundaries parameter pack values, boundaries, will also
  // be part of this this host_view_type object.

  host_view_type pointer_holder(kokkos_ptr, boundaries...);

  // boundaries will contain the array dimenions; an allocation is implicitly
  // made here
  device_view_type device_data_copy("StringName", boundaries...);

  mirror_view_type cpu_to_gpu_mirror =
      Kokkos::create_mirror_view(device_data_copy);

  // We need to deep_copy our existing data, the contents of
  // pointer_holder, into the mirror_view;
  // Copying from Host to Device has two steps:  1) Change the layout, 2)
  // change the memory_space (host or device).  Step 1 is to change the
  // layout to enable sending data from CPU to GPU. Step 2 is actually
  // sending the optimal data layout to the GPU

  // This step changes the Layout to be optimal for the gpu
  Kokkos::deep_copy(cpu_to_gpu_mirror, pointer_holder);

  // The mirror view data layout on the HOST is like the layout for the GPU.
  // GPU-optimized layouts are LayoutLeft, i.e., column-major This deep_copy
  // copy GPU-layout data on the HOST to the Device

  // Actual copying of the data from the host to the gpu
  Kokkos::deep_copy(device_data_copy, cpu_to_gpu_mirror);

  // Kokkos::View return type

  return device_data_copy;
}

///////////////////////////////////////////////////////////////////////////////
// THIS FUNCTION WILL MOVE DATA IN A KOKKOS::VIEW BACK TO HOST FROM DEVICE, AND
// STORE IN AN EXISTING POINTER
///////////////////////////////////////////////////////////////////////////////

template <class PointedAt, class ExistingView, class... Boundaries>

// DEFINING FUNCTION THAT GETS A VIEW FROM A POINTER WITH RETURN TYPE
// KOKKOS::VIEW
//"my_view" parameter is equivalent to device_data_copy
//
void moveDataToHostFromKokkosView(PointedAt *kokkos_ptr, ExistingView my_view,
                                  Boundaries... boundaries)

{
  // This says construct the pointer_holder variable from arguments passed to
  // the template block
  //
  using host_view_type = typename Kokkos::View<
      typename PointerOfNdimensions<PointedAt, sizeof...(Boundaries)>::type,
      typename Kokkos::DefaultHostExecutionSpace::memory_space>;

  // FYI - Device can be GPU, OpenMPTarget, HIP (for targeting an AMD GPU), SYCL
  // (library in Intel Compiler)
  //
  using device_view_type = typename Kokkos::View<
      typename PointerOfNdimensions<PointedAt, sizeof...(Boundaries)>::type,
      typename Kokkos::DefaultExecutionSpace::memory_space>;

  // When copying data, we can either change the Layout or the memory_space
  // (host or device), but we cannot change both!
  // Here, we are mirroring data on the host to the device, i.e., Layout is
  // as if on the device, but the data is actually on the host.  The host
  // mirror will be Layout Left (optimal for the device), but data are
  // actually on the HOST!

  // Here, "using" is type alias; in this example,its our gpu Layout on cpu
  using mirror_view_type = typename device_view_type::HostMirror;

  // Assignment statement; we are constructing a host_view_type with the name
  // pointer_holder.  The value of kokkos_ptr is the pointer we're wrapping on
  // the Host, and the Boundaries parameter pack values, boundaries, will also
  // be part of this this host_view_type object.

  host_view_type pointer_holder(kokkos_ptr, boundaries...);

  // Layout is optimal for gpu, but located on CPU
  mirror_view_type cpu_to_gpu_mirror = Kokkos::create_mirror_view(my_view);
  //auto mirror_view_type cpu_to_gpu_mirror = Kokkos::create_mirror_view(my_view);

  // We need to deep_copy our existing data, the contents of
  // pointer_holder, into the mirror_view;
  // Copying from Host to Device has two steps:  1) Change the layout, 2)
  // change the memory_space (host or device).  Step 1 is to change the
  // layout to enable sending data from CPU to GPU. Step 2 is actually
  // sending the optimal data layout to the GPU

  // This step changes the Layout to be optimal for the gpu

  // The mirror view data layout on the HOST is like the layout for the GPU.
  // GPU-optimized layouts are LayoutLeft, i.e., column-major This deep_copy
  // copy GPU-layout data on the HOST to the Device

  // Actual copying of the data from the gpu to the cpu
  Kokkos::deep_copy(cpu_to_gpu_mirror, my_view);

  // This copies from the mirror on the cpu
  Kokkos::deep_copy(pointer_holder, cpu_to_gpu_mirror);
}


#endif // RUN_KOKKOS

class KernelBase;
class RunParams;

/*!
 *******************************************************************************
 *
 * \brief Enumeration defining size specification for the polybench kernels
 *
 * Polybench comes with a spec file to setup the iteration space for 
 * various sizes: Mini, Small, Medium, Large, Extralarge
 *
 * We adapt those entries within this perfsuite.
 *
 * The default size is Medium, which can be overridden at run-time.
 *
 * An example partial entry from that file showing the MINI and SMALL spec 
 * for the kernel 3mm is:
 *
 * kernel	category	datatype	params	MINI	SMALL	MEDIUM	LARGE	EXTRALARGE
 * 3mm	linear-algebra/kernels	double	NI NJ NK NL NM	16 18 20 22 24	40 50 60 70 80 .... 
 * *
 *******************************************************************************
 */
enum SizeSpec {
  
  Mini = 0,
  Small,
  Medium,
  Large,
  Extralarge,
  Specundefined

};


//
/*!
 *******************************************************************************
 *
 * \brief Enumeration defining unique id for each VARIANT in suite.
 *
 * IMPORTANT: This is only modified when a variant is added or removed.
 *
 *            ENUM VALUES MUST BE KEPT CONSISTENT (CORRESPONDING ONE-TO-ONE) 
 *            WITH ARRAY OF VARIANT NAMES IN IMPLEMENTATION FILE!!!
 *
 *******************************************************************************
 */
enum VariantID {

  Base_Seq = 0,
  Lambda_Seq,
  RAJA_Seq,

  Base_OpenMP,
  Lambda_OpenMP,
  RAJA_OpenMP,

  Base_OpenMPTarget,
  RAJA_OpenMPTarget,

  Base_CUDA,
  Lambda_CUDA,
  RAJA_CUDA,
  RAJA_WORKGROUP_CUDA,

  Base_HIP,
  Lambda_HIP,
  RAJA_HIP,
  RAJA_WORKGROUP_HIP,

  Kokkos_Lambda,
  Kokkos_Functor,

  NumVariants // Keep this one last and NEVER comment out (!!)

};


/*!
 *******************************************************************************
 *
 * \brief Return variant name associated with VariantID enum value.
 *
 *******************************************************************************
 */
const std::string& getVariantName(VariantID vid);

/*!
 *******************************************************************************
 *
 * \brief Return true if variant associated with VariantID enum value is 
 *        available * to run; else false.
 *
 *******************************************************************************
 */
bool isVariantAvailable(VariantID vid);

}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
