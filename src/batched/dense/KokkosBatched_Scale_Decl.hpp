#ifndef __KOKKOSBATCHED_SCALE_DECL_HPP__
#define __KOKKOSBATCHED_SCALE_DECL_HPP__

/// \author Kyungjoo Kim (kyukim@sandia.gov)

namespace KokkosBatched {

///
/// Serial Scale
///

struct SerialScale {
  template <typename ScalarType, typename AViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const ScalarType alpha,
                                           const AViewType &A) {
    assert(false && "Deprecated: use KokkosBlas::SerialScale");
    return 0;
  }
};

///
/// Team Scale
///

template <typename MemberType>
struct TeamScale {
  template <typename ScalarType, typename AViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member,
                                           const ScalarType alpha,
                                           const AViewType &A) {
    assert(false && "Deprecated: use KokkosBlas::TeamScale");
    return 0;
  }
};

///
/// TeamVector Scale
///

template <typename MemberType>
struct TeamVectorScale {
  template <typename ScalarType, typename AViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member,
                                           const ScalarType alpha,
                                           const AViewType &A) {
    // static_assert(false);
    assert(false && "Deprecated: use KokkosBlas::TeamVectorScale");
    return 0;
  }
};

}  // namespace KokkosBatched

#endif
