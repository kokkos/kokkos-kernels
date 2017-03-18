#ifndef __KOKKOSKERNELS_GEMV_SERIAL_IMPL_HPP__
#define __KOKKOSKERNELS_GEMV_SERIAL_IMPL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosKernels_Util.hpp"

namespace KokkosKernels {

  ///
  /// Serial Impl
  /// ===========
  namespace Serial {

    template<>
    template<typename ScalarType,
             typename AViewType,
             typename xViewType,
             typename yViewType>
    KOKKOS_INLINE_FUNCTION
    int
    Gemv<Trans::NoTranspose,Algo::Gemv::CompactMKL>::
    invoke(const ScalarType alpha,
           const AViewType &A,
           const xViewType &x,
           const ScalarType beta,
           const yViewType &y) {
      typedef typename yViewType::value_type vector_type;
      typedef typename vector_type::value_type value_type;

      const int
        m = A.dimension(0),
        n = 1,
        k = A.dimension(1),
        vl = vector_type::vector_length;

      // no error check
      cblas_dgemm_compact(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                          m, n, k, 
                          alpha, 
                          (const double*)A.data(), A.stride_0(), 
                          (const double*)x.data(), x.stride_0(), 
                          beta,
                          (double*)y.data(), y.stride_0(),
                          (MKL_INT)vl, (MKL_INT)1);
    }
    
    template<>
    template<typename ScalarType,
             typename AViewType,
             typename xViewType,
             typename yViewType>
    KOKKOS_INLINE_FUNCTION
    int
    Gemv<Trans::NoTranspose,Algo::Gemv::Unblocked>::
    invoke(const ScalarType alpha,
           const AViewType &A,
           const xViewType &x,
           const ScalarType beta,
           const yViewType &y) {
      // y = beta y + alpha A x
      // y (m), A(m x n), B(n)
      
      typedef typename yViewType::value_type value_type;
      
      if      (beta == 0) Util::set  (y, value_type(0)   );
      else if (beta != 1) Util::scale(y, value_type(beta));
      
      if (alpha != 0) {
        const int
          m = A.dimension_0(),
          n = A.dimension_1();

        if (!(m && n)) return 0;
        
        const int
          as0 = A.stride_0(),
          as1 = A.stride_1(),
          xs0 = x.stride_0(),
          ys0 = y.stride_0();
        
        value_type
          *__restrict__ pY = &y(0);
        const value_type
          *__restrict__ pA = &A(0,0),
          *__restrict__ pX = &x(0);

        for (int i=0;i<m;++i) {
          value_type t(0);
          const value_type 
            *__restrict__ tA = (pA + i*as0);
          for (int j=0;j<n;++j)
            t += tA[j*as1]*pX[j*xs0];
          pY[i*ys0] += alpha*t;
        }
      }
      return 0;
    }

    // template<int mb>
    // template<typename ScalarType,
    //          typename ValueType>
    // KOKKOS_INLINE_FUNCTION
    // int 
    // InnerMultipleDotProduct<mb>::
    // invoke(const ScalarType alpha,
    //        const ValueType *__restrict__ A,
    //        const ValueType *__restrict__ x,
    //        const int m, const int n, 
    //        /**/  ValueType *__restrict__ y) {
    //   if (!(m>0 && n>0)) return 0;

    //   for (int i=0;i<m;++i) {
    //     ValueType t(0);
    //     const ValueType
    //       *__restrict__ tA = A + i*_as0;
    //     for (int j=0;j<n;++j)
    //       t += tA[j*_as1]*x[j*_xs0];
    //     y[i*_ys0] += alpha*t;
    //   }
    // }

    template<>
    template<typename ScalarType,
             typename ValueType>
    KOKKOS_INLINE_FUNCTION
    int 
    InnerMultipleDotProduct<4>::
    invoke(const ScalarType alpha,
           const ValueType *__restrict__ A,
           const ValueType *__restrict__ x,
           const int m, const int n, 
           /**/  ValueType *__restrict__ y) {
      if (!(m>0 && n>0)) return 0;

      const int 
        i0 = 0*_as0, i1 = 1*_as0, i2 = 2*_as0, i3 = 3*_as0;

      // unroll by rows

      ValueType
        y_0 = 0, y_1 = 0, y_2 = 0, y_3 = 0;
            
      for (int j=0;j<n;++j) {
        const int jj = j*_as1;
        const ValueType x_j = x[j*_xs0];

        switch (m) {
        case 4: y_3 += A[i3+jj]*x_j;
        case 3: y_2 += A[i2+jj]*x_j;
        case 2: y_1 += A[i1+jj]*x_j;
        case 1: y_0 += A[i0+jj]*x_j;
        }
      }
      
      switch (m) {
      case 4: y[3*_ys0] += alpha*y_3;
      case 3: y[2*_ys0] += alpha*y_2;
      case 2: y[1*_ys0] += alpha*y_1;
      case 1: y[0*_ys0] += alpha*y_0;
      }

      return 0;
    }

    template<>
    template<typename ScalarType,
             typename ValueType>
    KOKKOS_INLINE_FUNCTION
    int 
    InnerMultipleDotProduct<8>::
    invoke(const ScalarType alpha,
           const ValueType *__restrict__ A,
           const ValueType *__restrict__ x,
           const int m, const int n, 
           /**/  ValueType *__restrict__ y) {
      if (!(m>0 && n>0)) return 0;

      const int 
        i0 = 0*_as0, i1 = 1*_as0, i2 = 2*_as0, i3 = 3*_as0,
        i4 = 4*_as0, i5 = 5*_as0, i6 = 6*_as0, i7 = 7*_as0;

      // unroll by rows

      ValueType
        y_0 = 0, y_1 = 0, y_2 = 0, y_3 = 0,
        y_4 = 0, y_5 = 0, y_6 = 0, y_7 = 0;
            
      for (int j=0;j<n;++j) {
        const int jj = j*_as1;
        const ValueType x_j = x[j*_xs0];

        switch (m) {
        case 8: y_7 += A[i7+jj]*x_j;
        case 7: y_6 += A[i6+jj]*x_j;
        case 6: y_5 += A[i5+jj]*x_j;
        case 5: y_4 += A[i4+jj]*x_j;
        case 4: y_3 += A[i3+jj]*x_j;
        case 3: y_2 += A[i2+jj]*x_j;
        case 2: y_1 += A[i1+jj]*x_j;
        case 1: y_0 += A[i0+jj]*x_j;
        }
      }
      
      switch (m) {
      case 8: y[7*_ys0] += alpha*y_7;
      case 7: y[6*_ys0] += alpha*y_6;
      case 6: y[5*_ys0] += alpha*y_5;
      case 5: y[4*_ys0] += alpha*y_4;
      case 4: y[3*_ys0] += alpha*y_3;
      case 3: y[2*_ys0] += alpha*y_2;
      case 2: y[1*_ys0] += alpha*y_1;
      case 1: y[0*_ys0] += alpha*y_0;
      }

      return 0;
    }

    template<>
    template<typename ScalarType,
             typename AViewType,
             typename xViewType,
             typename yViewType>
    KOKKOS_INLINE_FUNCTION
    int
    Gemv<Trans::NoTranspose,Algo::Gemv::Blocked>::
    invoke(const ScalarType alpha,
           const AViewType &A,
           const xViewType &x,
           const ScalarType beta,
           const yViewType &y) {
      // y = beta y + alpha A x
      // y (m), A(m x n), B(n)
      
      typedef typename yViewType::value_type value_type;
      
      if      (beta == 0) Util::set  (y, value_type(0)   );
      else if (beta != 1) Util::scale(y, value_type(beta));
      
      if (alpha != 0) {
        const int
          m = A.dimension_0(),
          n = A.dimension_1();

        if (!(m && n)) return 0;
        
        const int
          as0 = A.stride_0(),
          as1 = A.stride_1(),
          xs0 = x.stride_0(),
          ys0 = y.stride_0();

        enum : int {
          mb = Algo::Gemm::Blocked::mb };

        InnerMultipleDotProduct<mb> inner(as0, as1,
                                          xs0, 
                                          ys0);

        const value_type
          *__restrict__ pX = &x(0);
        
        for (int i=0;i<m;i+=mb) {
          const int mi = (m - i), mp = (mi > mb ? mb : mi);
          inner.invoke(alpha, &A(i,0), pX, mp, n, &y(i));
        }
      }
      return 0;
    }
  }
} // end namespace KokkosKernels

#endif
