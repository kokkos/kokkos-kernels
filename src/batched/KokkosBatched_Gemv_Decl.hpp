#ifndef __KOKKOSBATCHED_GEMV_DECL_HPP__
#define __KOKKOSBATCHED_GEMV_DECL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)


namespace KokkosBatched {
  namespace Experimental {

    ///
    /// Serial Gemv 
    ///

    namespace Serial {
      template<typename ArgTrans,
               typename ArgAlgo>
      struct Gemv {
        template<typename ScalarType,
                 typename AViewType,
                 typename xViewType,
                 typename yViewType>
        KOKKOS_INLINE_FUNCTION
        static int
        invoke(const ScalarType alpha,
               const AViewType &A,
               const xViewType &x,
               const ScalarType beta,
               const yViewType &y);
      };
    }
  
    ///
    /// Team Gemv 
    ///

    namespace Team {
      template<typename MemberType,
               typename ArgTrans,
               typename ArgAlgo>
      struct Gemv {
        template<typename ScalarType,
                 typename AViewType,
                 typename xViewType,
                 typename yViewType>
        KOKKOS_INLINE_FUNCTION
        static int
        invoke(const MemberType &member, 
               const ScalarType alpha,
               const AViewType &A,
               const xViewType &x,
               const ScalarType beta,
               const yViewType &y);
      };
    }
  
  }
}

#endif
