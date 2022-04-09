/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Jennifer Loe (jloe@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef __KOKKOSBATCHED_ODE_ALLOCATIONSTATE_HPP__
#define __KOKKOSBATCHED_ODE_ALLOCATIONSTATE_HPP__

#include "Kokkos_View.hpp"

namespace KokkosBatched {
namespace Experimental {
namespace ODE {

struct StackAllocationTag;
struct ScratchAllocationTag;
struct DynamicAllocationTag;

struct EmptyRkStack {};

template <int NDOFS, int NSTAGES>
struct RkStack {
  using type = StackAllocationTag;

  using V1   = Kokkos::View<double[NDOFS], Kokkos::AnonymousSpace,
                          Kokkos::MemoryUnmanaged>;
  using V2   = Kokkos::View<double[NSTAGES][NDOFS], Kokkos::LayoutRight,
                          Kokkos::AnonymousSpace, Kokkos::MemoryUnmanaged>;
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
  using type   = DynamicAllocationTag;
  using View2D = Kokkos::View<double **, MemorySpace>;
  using View3D = Kokkos::View<double ***, MemorySpace>;

  RkDynamicAllocation(View2D y_, View2D y0_, View2D dydt_, View2D ytemp_,
                      View3D k_)
      : y(y_), y0(y0_), dydt(dydt_), ytemp(ytemp_), k(k_) {}

  RkDynamicAllocation(const int n, const int ndofs, const int nstages)
      : y(Kokkos::ViewAllocateWithoutInitializing("y"), n, ndofs),
        y0(Kokkos::ViewAllocateWithoutInitializing("y0"), n, ndofs),
        dydt(Kokkos::ViewAllocateWithoutInitializing("dydt"), n, ndofs),
        ytemp(Kokkos::ViewAllocateWithoutInitializing("ytemp"), n, ndofs),
        k(Kokkos::ViewAllocateWithoutInitializing("k"), n, nstages, ndofs) {}

  View2D y;
  View2D y0;
  View2D dydt;
  View2D ytemp;
  View3D k;
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
  using View1D = Kokkos::View<double *, Layout, Kokkos::AnonymousSpace,
                              Kokkos::MemoryUnmanaged>;
  using View2D = Kokkos::View<double **, Layout, Kokkos::AnonymousSpace,
                              Kokkos::MemoryUnmanaged>;
  // Internal variables:
  View1D y;
  View1D y0;
  View1D dydt;
  View1D ytemp;
  View2D k;  // NSTAGES x NDOFS

  KOKKOS_FORCEINLINE_FUNCTION int ndofs() const {
    return static_cast<int>(y.extent(0));
  }

  // wrap stack
  // Its unclear why cuda compiler decides to put the stack in local memory /
  // unified cache when we make the stack a member variable... so we're forced
  // to pass it in to give the compiler a chance of putting it in the threads
  // registers
  template <typename... Ignored>
  KOKKOS_FORCEINLINE_FUNCTION void set_views(Allocation &stack, Ignored...) {
    using V1 = typename Allocation::V1;
    using V2 = typename Allocation::V2;
    y        = V1(stack.y.data());
    y0       = V1(stack.y0.data());
    dydt     = V1(stack.dydt.data());
    ytemp    = V1(stack.ytemp.data());
    k        = V2(stack.k.data());
  }

  // wrap host / device dynamically allocated memory
  KOKKOS_FORCEINLINE_FUNCTION void set_views(EmptyRkStack /*&stack*/,
                                             const Allocation &dynamic,
                                             int tid) {
    y     = Kokkos::subview(dynamic.y, tid, Kokkos::ALL);
    y0    = Kokkos::subview(dynamic.y0, tid, Kokkos::ALL);
    dydt  = Kokkos::subview(dynamic.dydt, tid, Kokkos::ALL);
    ytemp = Kokkos::subview(dynamic.ytemp, tid, Kokkos::ALL);
    k     = Kokkos::subview(dynamic.k, tid, Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_FORCEINLINE_FUNCTION void set_views(const Allocation &dynamic,
                                             int tid) {
    EmptyRkStack stack{};
    set_views(stack, dynamic, tid);
  }

  // wrap scratch pad memory
  template <typename ScratchHandle>
  KOKKOS_FORCEINLINE_FUNCTION void set_views(EmptyRkStack /*&stack*/,
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
  }

  template <typename ScratchHandle>
  KOKKOS_FORCEINLINE_FUNCTION void set_views(ScratchHandle &handle,
                                             const int ndofs,
                                             const int nstages) {
    EmptyRkStack stack{};
    set_views(stack, handle, ndofs, nstages);
  }
};

}  // namespace ODE
}  // namespace Experimental
}  // namespace KokkosBatched

#endif
