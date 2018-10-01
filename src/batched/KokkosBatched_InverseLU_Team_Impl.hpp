#ifndef __KOKKOSBATCHED_INVERSELU_TEAM_IMPL_HPP__
#define __KOKKOSBATCHED_INVERSELU_TEAM_IMPL_HPP__


/// \author Vinh Dang (vqdang@sandia.gov)

#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_Trsm_Decl.hpp"
#include "KokkosBatched_Trsm_Team_Impl.hpp"


namespace KokkosBatched {
  namespace Experimental {
    ///
    /// Team Impl
    /// =========

    ///
    /// InverseLU no piv
    ///
    
    template<typename MemberType>
    struct TeamInverseLU<MemberType,Algo::InverseLU::Unblocked> {
      template<typename AViewType,
               typename WViewType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const MemberType &member, const AViewType &A, const WViewType &W,
             const typename MagnitudeScalarType<typename AViewType::non_const_value_type>::type tiny = 0) {
        static_assert(AViewType::rank == 2, "A should have two dimensions");
        static_assert(std::is_same<typename AViewType::memory_space, typename WViewType::memory_space>::value, "A and W should be on the same memory space");
        assert(A.span()*sizeof(typename AViewType::value_type) <= W.span()*sizeof(typename WViewType::value_type));

        typedef typename AViewType::value_type ScalarType;

        int B_stride_0, B_stride_1;
        if (WViewType::rank == 1) {
            if (std::is_same<typename WViewType::array_layout, Kokkos::LayoutRight>::value){
                B_stride_0 = A.extent(1);
                B_stride_1 = 1;
            } 
            else {
                B_stride_0 = A.stride_0();
                B_stride_1 = A.stride_0()*A.extent(0);
            }
        }
        else { //WViewType::rank = 2
            B_stride_0 = A.stride_0();
            B_stride_1 = A.stride_1();
        }
        auto B = Kokkos::View<ScalarType**, Kokkos::LayoutStride, typename WViewType::memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> >(W.data(), A.extent(0), B_stride_0, A.extent(1), B_stride_1);

        ScalarType one(1.0);

        Kokkos::parallel_for(Kokkos::TeamThreadRange(member,A.extent(1)),[&](const int &i) {
            B(i,i) = one;
        });

        //First, compute L inverse by solving the system L*Linv = I for Linv
        TeamTrsm<MemberType,Side::Left,Uplo::Lower,Trans::NoTranspose,Diag::Unit,Algo::Trsm::Unblocked>::invoke(member, one, A, B);
        //Second, compute A inverse by solving the system U*Ainv = Linv for Ainv
        TeamTrsm<MemberType,Side::Left,Uplo::Upper,Trans::NoTranspose,Diag::NonUnit,Algo::Trsm::Unblocked>::invoke(member, one, A, B);

        Kokkos::parallel_for(Kokkos::TeamThreadRange(member,A.extent(0)*A.extent(1)),[&](const int &tid) {
            int i = tid/A.extent(1);
            int j = tid%A.extent(1);
            A(i,j) = B(i,j);
        });

        return 0;
      }
    };
    
    template<typename MemberType>
    struct TeamInverseLU<MemberType,Algo::InverseLU::Blocked> {
      template<typename AViewType,
               typename WViewType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const MemberType &member, const AViewType &A, const WViewType &W,
             const typename MagnitudeScalarType<typename AViewType::non_const_value_type>::type tiny = 0) {
        static_assert(AViewType::rank == 2, "A should have two dimensions");
        static_assert(std::is_same<typename AViewType::memory_space, typename WViewType::memory_space>::value, "A and W should be on the same memory space");
        assert(A.span()*sizeof(typename AViewType::value_type) <= W.span()*sizeof(typename WViewType::value_type));

        typedef typename AViewType::value_type ScalarType;

        int B_stride_0, B_stride_1;
        if (WViewType::rank == 1) {
            if (std::is_same<typename WViewType::array_layout, Kokkos::LayoutRight>::value){
                B_stride_0 = A.extent(1);
                B_stride_1 = 1;
            } 
            else {
                B_stride_0 = A.stride_0();
                B_stride_1 = A.stride_0()*A.extent(0);
            }
        }
        else { //WViewType::rank = 2
            B_stride_0 = A.stride_0();
            B_stride_1 = A.stride_1();
        }
        auto B = Kokkos::View<ScalarType**, Kokkos::LayoutStride, typename WViewType::memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> >(W.data(), A.extent(0), B_stride_0, A.extent(1), B_stride_1);

        ScalarType one(1.0);

        Kokkos::parallel_for(Kokkos::TeamThreadRange(member,A.extent(1)),[&](const int &i) {
            B(i,i) = one;
        });

        //First, compute L inverse by solving the system L*Linv = I for Linv
        TeamTrsm<MemberType,Side::Left,Uplo::Lower,Trans::NoTranspose,Diag::Unit,Algo::Trsm::Blocked>::invoke(member, one, A, B);
        //Second, compute A inverse by solving the system U*Ainv = Linv for Ainv
        TeamTrsm<MemberType,Side::Left,Uplo::Upper,Trans::NoTranspose,Diag::NonUnit,Algo::Trsm::Blocked>::invoke(member, one, A, B);

        Kokkos::parallel_for(Kokkos::TeamThreadRange(member,A.extent(0)*A.extent(1)),[&](const int &tid) {
            int i = tid/A.extent(1);
            int j = tid%A.extent(1);
            A(i,j) = B(i,j);
        });

        return 0;
      }
    };

  }
}

#endif
