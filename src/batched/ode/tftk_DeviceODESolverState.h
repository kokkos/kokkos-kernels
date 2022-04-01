/*--------------------------------------------------------------------*/
/*    Copyright 2002 - 2008, 2010, 2011 National Technology &         */
/*    Engineering Solutions of Sandia, LLC (NTESS). Under the terms   */
/*    of Contract DE-NA0003525 with NTESS, there is a                 */
/*    non-exclusive license for use of this work by or on behalf      */
/*    of the U.S. Government.  Export of this program may require     */
/*    a license from the United States Government.                    */
/*--------------------------------------------------------------------*/
#ifndef SIERRA_tftk_DeviceODESolverState_h
#define SIERRA_tftk_DeviceODESolverState_h

#include "Kokkos_View.hpp"
//#include "tftk_util/tftk_KokkosTypes.h"
#include "tftk_KokkosTypes.h"

namespace tftk {
namespace ode {

struct StackAllocationTag;
struct ScratchAllocationTag;
struct DynamicAllocationTag;

struct EmptyRkStack {};

template <int NDOFS, int NSTAGES>
struct RkStack {
  using type = StackAllocationTag;

  using V1 =
      Kokkos::View<double[NDOFS], tftk::AnonSpace, Kokkos::MemoryUnmanaged>;
  using V2   = Kokkos::View<double[NSTAGES][NDOFS], Kokkos::LayoutRight,
                          tftk::AnonSpace, Kokkos::MemoryUnmanaged>;
  using Arr1 = Kokkos::Array<double, NDOFS>;
  using Arr2 = Kokkos::Array<double, NSTAGES * NDOFS>;
  Arr1 y;
  Arr1 y0;
  Arr1 dydt;
  Arr1 ytemp;
  Arr2 k;
};

template <typename MemorySpace>
struct RkDynamicAllocation {
  using type  = DynamicAllocationTag;
  using View1 = Kokkos::View<double **, MemorySpace>;
  using View2 = Kokkos::View<double ***, MemorySpace>;

  RkDynamicAllocation(View1 y_, View1 y0_, View1 dydt_, View1 ytemp_, View2 k_)
      : y(y_), y0(y0_), dydt(dydt_), ytemp(ytemp_), k(k_) {}

  RkDynamicAllocation(const int n, const int ndofs, const int nstages)
      : y(Kokkos::ViewAllocateWithoutInitializing("y"), n, ndofs),
        y0(Kokkos::ViewAllocateWithoutInitializing("y0"), n, ndofs),
        dydt(Kokkos::ViewAllocateWithoutInitializing("dydt"), n, ndofs),
        ytemp(Kokkos::ViewAllocateWithoutInitializing("ytemp"), n, ndofs),
        k(Kokkos::ViewAllocateWithoutInitializing("k"), n, nstages, ndofs) {}

  View1 y;
  View1 y0;
  View1 dydt;
  View1 ytemp;
  View2 k;
};

template <typename MemorySpace>
struct RkSharedAllocation {
  using type = ScratchAllocationTag;
  using ScratchSpace =
      Kokkos::ScratchMemorySpace<typename MemorySpace::execution_space>;
  using V1 = Kokkos::View<double *, ScratchSpace, Kokkos::MemoryUnmanaged>;
  using V2 = Kokkos::View<double **, ScratchSpace, Kokkos::MemoryUnmanaged>;
};

template <typename Allocation>
struct RkSolverState {
  using Type = typename Allocation::type;
  using StackType =
      std::conditional_t<std::is_same<Type, StackAllocationTag>::value,
                         Allocation, EmptyRkStack>;
  using Layout =
      std::conditional_t<std::is_same<Type, StackAllocationTag>::value,
                         Kokkos::LayoutRight, Kokkos::LayoutStride>;
  using View1 =
      Kokkos::View<double *, Layout, tftk::AnonSpace, Kokkos::MemoryUnmanaged>;
  using View2 =
      Kokkos::View<double **, Layout, tftk::AnonSpace, Kokkos::MemoryUnmanaged>;

  KOKKOS_FORCEINLINE_FUNCTION int ndofs() const {
    return static_cast<int>(y.extent(0));
  };

  // wrap stack
  // Its unclear why cuda compiler decides to put the stack in local memory /
  // unified cache when we make the stack a member variable... so we're forced
  // to pass it in to give the compiler a chance of putting it in the threads
  // registers
  template <typename... Ignored>
  KOKKOS_FORCEINLINE_FUNCTION void set_views(Allocation &stack,
                                             Ignored... ignored) {
    using V1 = typename Allocation::V1;
    using V2 = typename Allocation::V2;
    y        = V1(stack.y.data());
    y0       = V1(stack.y0.data());
    dydt     = V1(stack.dydt.data());
    ytemp    = V1(stack.ytemp.data());
    k        = V2(stack.k.data());
  };

  // wrap host / device dynamically allocated memory
  KOKKOS_FORCEINLINE_FUNCTION void set_views(EmptyRkStack &stack,
                                             const Allocation &dynamic,
                                             int tid) {
    y     = Kokkos::subview(dynamic.y, tid, Kokkos::ALL);
    y0    = Kokkos::subview(dynamic.y0, tid, Kokkos::ALL);
    dydt  = Kokkos::subview(dynamic.dydt, tid, Kokkos::ALL);
    ytemp = Kokkos::subview(dynamic.ytemp, tid, Kokkos::ALL);
    k     = Kokkos::subview(dynamic.k, tid, Kokkos::ALL, Kokkos::ALL);
  };

  KOKKOS_FORCEINLINE_FUNCTION void set_views(const Allocation &dynamic,
                                             int tid) {
    EmptyRkStack stack{};
    set_views(stack, dynamic, tid);
  };

  // wrap scratch pad memory
  template <typename ScratchHandle>
  KOKKOS_FORCEINLINE_FUNCTION void set_views(EmptyRkStack &stack,
                                             ScratchHandle &handle,
                                             const int ndofs,
                                             const int nstages) {
    using V1 = typename Allocation::V1;
    using V2 = typename Allocation::V2;
    y        = V1(handle, ndofs);
    y0       = V1(handle, ndofs);
    dydt     = V1(handle, ndofs);
    ytemp    = V1(handle, ndofs);
    k        = V2(handle, nstages, ndofs);
  };

  template <typename ScratchHandle>
  KOKKOS_FORCEINLINE_FUNCTION void set_views(ScratchHandle &handle,
                                             const int ndofs,
                                             const int nstages) {
    EmptyRkStack stack{};
    set_views(stack, handle, ndofs, nstages);
  };

  View1 y;
  View1 y0;
  View1 dydt;
  View1 ytemp;
  View2 k;  // NSTAGES x NDOFS
};

}  // namespace ode
}  // namespace tftk

#endif
