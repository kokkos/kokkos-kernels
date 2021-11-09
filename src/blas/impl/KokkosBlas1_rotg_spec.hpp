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
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/
#ifndef KOKKOSBLAS1_ROTG_SPEC_HPP_
#define KOKKOSBLAS1_ROTG_SPEC_HPP_

#include <KokkosKernels_config.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_ArithTraits.hpp>

// Include the actual functors
#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
#include <KokkosBlas1_rotg_impl.hpp>
#endif

namespace KokkosBlas {
namespace Impl {
// Specialization struct which defines whether a specialization exists
template <class Scalar, class execution_space, class memory_space>
struct rotg_eti_spec_avail {
  enum : bool { value = false };
};
}  // namespace Impl
}  // namespace KokkosBlas

//
// Macro for declaration of full specialization availability
// KokkosBlas::Impl::Iamax for rank == 1.  This is NOT for users!!!  All
// the declarations of full specializations go in this header file.
// We may spread out definitions (see _INST macro below) across one or
// more .cpp files.
//
#define KOKKOSBLAS1_ROTG_ETI_SPEC_AVAIL(SCALAR, EXECSPACE, MEMSPACE)	\
  template <>								\
  struct rotg_eti_spec_avail<SCALAR, EXECSPACE, MEMSPACE> {		\
    enum : bool { value = true };					\
  };


// Include the actual specialization declarations
#include <KokkosBlas1_rotg_tpl_spec_avail.hpp>
#include <generated_specializations_hpp/KokkosBlas1_rotg_eti_spec_avail.hpp>

namespace KokkosBlas {
namespace Impl {

// Unification layer
template <class Scalar, class execution_space, class memory_space,
          bool tpl_spec_avail = rotg_tpl_spec_avail<Scalar, execution_space, memory_space>::value,
          bool eti_spec_avail = rotg_eti_spec_avail<Scalar, execution_space, memory_space>::value>
struct Rotg {
  static void rotg(execution_space space, Scalar& a, Scalar& b,
		   typename Kokkos::ArithTraits<Scalar>::mag_type& c,
		   Scalar& s);
};

#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
//! Full specialization of Rotg
template <class Scalar, class execution_space, class memory_space>
struct Rotg<Scalar, execution_space, memory_space, false, KOKKOSKERNELS_IMPL_COMPILE_LIBRARY> {

  static void rotg(execution_space space,
		   Scalar& a, Scalar& b,
		   typename Kokkos::ArithTraits<Scalar>::mag_type& c,
		   Scalar& s) {
    Kokkos::Profiling::pushRegion(KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
				  ? "KokkosBlas::rotg[ETI]"
				  : "KokkosBlas::rotg[noETI]");
#ifdef KOKKOSKERNELS_ENABLE_CHECK_SPECIALIZATION
    if (KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
      printf("KokkosBlas1::rotg<> ETI specialization for %s\n",
             typeid(Scalar).name());
    else {
      printf("KokkosBlas1::rotg<> non-ETI specialization for %s\n",
             typeid(Scalar).name());
    }
#endif

    Rotg_Invoke<Scalar, execution_space, memory_space>(space, a, b, c, s);
    Kokkos::Profiling::popRegion();
  }
};
#endif

}  // namespace Impl
}  // namespace KokkosBlas

//
// Macro for declaration of full specialization of
// KokkosBlas::Impl::Rotg.  This is NOT for users!!!  All
// the declarations of full specializations go in this header file.
// We may spread out definitions (see _DEF macro below) across one or
// more .cpp files.
//
#define KOKKOSBLAS1_ROTG_ETI_SPEC_DECL(SCALAR, EXECSPACE, MEMSPACE)	\
  extern template struct Rotg<SCALAR, EXECSPACE, MEMSPACE, false, true>;

//
// Macro for definition of full specialization of
// KokkosBlas::Impl::Rotg.  This is NOT for users!!!  We
// use this macro in one or more .cpp files in this directory.
//
#define KOKKOSBLAS1_ROTG_ETI_SPEC_INST(SCALAR, EXECSPACE, MEMSPACE)	\
  template struct Rotg<SCALAR, EXECSPACE, MEMSPACE, false, true>;

#include <KokkosBlas1_rotg_tpl_spec_decl.hpp>
#include <generated_specializations_hpp/KokkosBlas1_rotg_eti_spec_decl.hpp>

#endif  // KOKKOSBLAS1_ROTG_SPEC_HPP_
