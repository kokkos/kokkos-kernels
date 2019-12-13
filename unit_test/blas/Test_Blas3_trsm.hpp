#include<gtest/gtest.h>
#include<Kokkos_Core.hpp>
#include<Kokkos_Random.hpp>
#include<KokkosBlas3_trsm.hpp>
#include<KokkosKernels_TestUtils.hpp>

namespace Test {

  template<class ViewTypeA, class ExecutionSpace>
  struct UnitDiagTRSM {
    ViewTypeA A_;
    using ScalarA = typename ViewTypeA::value_type;

    UnitDiagTRSM (const ViewTypeA& A) : A_(A) {}

    KOKKOS_INLINE_FUNCTION
    void operator() (const int& i) const {
      A_(i,i) = ScalarA(1);
    }
  };
  template<class ViewTypeA, class ExecutionSpace>
  struct NonUnitDiagTRSM {
    ViewTypeA A_;
    using ScalarA = typename ViewTypeA::value_type;

    NonUnitDiagTRSM (const ViewTypeA& A) : A_(A) {}

    KOKKOS_INLINE_FUNCTION
    void operator() (const int& i) const {
      A_(i,i) = A_(i,i)*1000;
    }
  };

  //For convenient testing purpose, wrappers of BLAS trmm and  
  //cuBLAS trmm are used
  //float
  template<class ViewTypeA, class ViewTypeB, class ExecSpace, class MemSpace>
  void trmm_wrapper (const char side[],
                     const char uplo[],
                     const char trans[],
                     const char diag[],
                     float& alpha,
                     const ViewTypeA& A,
                     const ViewTypeB& B)
  {
    const int M = static_cast<int> (B.extent(0));
    const int N = static_cast<int> (B.extent(1));

    bool A_is_ll = std::is_same<Kokkos::LayoutLeft,typename ViewTypeA::array_layout>::value;
    bool B_is_ll = std::is_same<Kokkos::LayoutLeft,typename ViewTypeB::array_layout>::value;

    const int AST = A_is_ll?A.stride(1):A.stride(0), LDA = (AST == 0) ? 1 : AST;
    const int BST = B_is_ll?B.stride(1):B.stride(0), LDB = (BST == 0) ? 1 : BST;

#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS
    if( std::is_same< ExecSpace, Kokkos::Cuda >::value ) {
      cublasHandle_t handle;
      cublasStatus_t stat = cublasCreate(&handle);
      if (stat != CUBLAS_STATUS_SUCCESS)
        Kokkos::abort("CUBLAS initialization failed\n");
      
      cublasSideMode_t  side_;
      cublasFillMode_t  uplo_;
      cublasOperation_t trans_;
      cublasDiagType_t  diag_;
      
      if(A_is_ll) {
        if ((side[0]=='L')||(side[0]=='l')) side_ = CUBLAS_SIDE_LEFT;
        else side_ = CUBLAS_SIDE_RIGHT;
        if ((uplo[0]=='L')||(uplo[0]=='l')) uplo_ = CUBLAS_FILL_MODE_LOWER;
        else uplo_ = CUBLAS_FILL_MODE_UPPER;
      } else {
        if ((side[0]=='L')||(side[0]=='l')) side_ = CUBLAS_SIDE_RIGHT;
        else side_ = CUBLAS_SIDE_LEFT;
        if ((uplo[0]=='L')||(uplo[0]=='l')) uplo_ = CUBLAS_FILL_MODE_UPPER;
        else uplo_ = CUBLAS_FILL_MODE_LOWER;
      }
      if ((trans[0]=='N')||(trans[0]=='n')) trans_ = CUBLAS_OP_N;
      else if ((trans[0]=='T')||(trans[0]=='t')) trans_ = CUBLAS_OP_T;
      else trans_ = CUBLAS_OP_C;
      if ((diag[0]=='U')||(diag[0]=='u')) diag_ = CUBLAS_DIAG_UNIT;
      else diag_ = CUBLAS_DIAG_NON_UNIT;
      
      if(A_is_ll)
        cublasStrmm(handle, side_, uplo_, trans_, diag_, M, N, &alpha, A.data(), LDA, B.data(), LDB, B.data(), LDB);
      else
        cublasStrmm(handle, side_, uplo_, trans_, diag_, N, M, &alpha, A.data(), LDA, B.data(), LDB, B.data(), LDB);
      
      cublasDestroy(handle);
    }
#endif
#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS
    if( std::is_same< MemSpace, Kokkos::HostSpace >::value ) {
      char  side_;
      char  uplo_;
      
      if(A_is_ll) {
        if ((side[0]=='L')||(side[0]=='l')) side_ = 'L';
        else side_ = 'R';
        if ((uplo[0]=='L')||(uplo[0]=='l')) uplo_ = 'L';
        else uplo_ = 'U';
      } else {
        if ((side[0]=='L')||(side[0]=='l')) side_ = 'R';
        else side_ = 'L';
        if ((uplo[0]=='L')||(uplo[0]=='l')) uplo_ = 'U';
        else uplo_ = 'L';
      }
      
      if(A_is_ll)
        KokkosBlas::Impl::HostBlas<float>::trmm(side_, uplo_, trans[0], diag[0], M, N, alpha, A.data(), LDA, B.data(), LDB);
      else
        KokkosBlas::Impl::HostBlas<float>::trmm(side_, uplo_, trans[0], diag[0], N, M, alpha, A.data(), LDA, B.data(), LDB);
    }
#endif
  }
  //double
  template<class ViewTypeA, class ViewTypeB, class ExecSpace, class MemSpace>
  void trmm_wrapper (const char side[],
                     const char uplo[],
                     const char trans[],
                     const char diag[],
                     double& alpha,
                     const ViewTypeA& A,
                     const ViewTypeB& B)
  {
    const int M = static_cast<int> (B.extent(0));
    const int N = static_cast<int> (B.extent(1));

    bool A_is_ll = std::is_same<Kokkos::LayoutLeft,typename ViewTypeA::array_layout>::value;
    bool B_is_ll = std::is_same<Kokkos::LayoutLeft,typename ViewTypeB::array_layout>::value;

    const int AST = A_is_ll?A.stride(1):A.stride(0), LDA = (AST == 0) ? 1 : AST;
    const int BST = B_is_ll?B.stride(1):B.stride(0), LDB = (BST == 0) ? 1 : BST;

#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS
    if( std::is_same< ExecSpace, Kokkos::Cuda >::value ) {
      cublasHandle_t handle;
      cublasStatus_t stat = cublasCreate(&handle);
      if (stat != CUBLAS_STATUS_SUCCESS)
        Kokkos::abort("CUBLAS initialization failed\n");
      
      cublasSideMode_t  side_;
      cublasFillMode_t  uplo_;
      cublasOperation_t trans_;
      cublasDiagType_t  diag_;
      
      if(A_is_ll) {
        if ((side[0]=='L')||(side[0]=='l')) side_ = CUBLAS_SIDE_LEFT;
        else side_ = CUBLAS_SIDE_RIGHT;
        if ((uplo[0]=='L')||(uplo[0]=='l')) uplo_ = CUBLAS_FILL_MODE_LOWER;
        else uplo_ = CUBLAS_FILL_MODE_UPPER;
      } else {
        if ((side[0]=='L')||(side[0]=='l')) side_ = CUBLAS_SIDE_RIGHT;
        else side_ = CUBLAS_SIDE_LEFT;
        if ((uplo[0]=='L')||(uplo[0]=='l')) uplo_ = CUBLAS_FILL_MODE_UPPER;
        else uplo_ = CUBLAS_FILL_MODE_LOWER;
      }
      if ((trans[0]=='N')||(trans[0]=='n')) trans_ = CUBLAS_OP_N;
      else if ((trans[0]=='T')||(trans[0]=='t')) trans_ = CUBLAS_OP_T;
      else trans_ = CUBLAS_OP_C;
      if ((diag[0]=='U')||(diag[0]=='u')) diag_ = CUBLAS_DIAG_UNIT;
      else diag_ = CUBLAS_DIAG_NON_UNIT;
      
      if(A_is_ll)
        cublasDtrmm(handle, side_, uplo_, trans_, diag_, M, N, &alpha, A.data(), LDA, B.data(), LDB, B.data(), LDB);
      else
        cublasDtrmm(handle, side_, uplo_, trans_, diag_, N, M, &alpha, A.data(), LDA, B.data(), LDB, B.data(), LDB);
      
      cublasDestroy(handle);
    }
#endif
#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS
    if( std::is_same< MemSpace, Kokkos::HostSpace >::value ) {
      char  side_;
      char  uplo_;
      
      if(A_is_ll) {
        if ((side[0]=='L')||(side[0]=='l')) side_ = 'L';
        else side_ = 'R';
        if ((uplo[0]=='L')||(uplo[0]=='l')) uplo_ = 'L';
        else uplo_ = 'U';
      } else {
        if ((side[0]=='L')||(side[0]=='l')) side_ = 'R';
        else side_ = 'L';
        if ((uplo[0]=='L')||(uplo[0]=='l')) uplo_ = 'U';
        else uplo_ = 'L';
      }
      
      if(A_is_ll)
        KokkosBlas::Impl::HostBlas<double>::trmm(side_, uplo_, trans[0], diag[0], M, N, alpha, A.data(), LDA, B.data(), LDB);
      else
        KokkosBlas::Impl::HostBlas<double>::trmm(side_, uplo_, trans[0], diag[0], N, M, alpha, A.data(), LDA, B.data(), LDB);
    }
#endif
  }
  //Kokkos::complex<float>
  template<class ViewTypeA, class ViewTypeB, class ExecSpace, class MemSpace>
  void trmm_wrapper (const char side[],
                     const char uplo[],
                     const char trans[],
                     const char diag[],
                     Kokkos::complex<float>& alpha,
                     const ViewTypeA& A,
                     const ViewTypeB& B)
  {
    const int M = static_cast<int> (B.extent(0));
    const int N = static_cast<int> (B.extent(1));

    bool A_is_ll = std::is_same<Kokkos::LayoutLeft,typename ViewTypeA::array_layout>::value;
    bool B_is_ll = std::is_same<Kokkos::LayoutLeft,typename ViewTypeB::array_layout>::value;

    const int AST = A_is_ll?A.stride(1):A.stride(0), LDA = (AST == 0) ? 1 : AST;
    const int BST = B_is_ll?B.stride(1):B.stride(0), LDB = (BST == 0) ? 1 : BST;

#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS
    if( std::is_same< ExecSpace, Kokkos::Cuda >::value ) {
      cublasHandle_t handle;
      cublasStatus_t stat = cublasCreate(&handle);
      if (stat != CUBLAS_STATUS_SUCCESS)
        Kokkos::abort("CUBLAS initialization failed\n");
      
      cublasSideMode_t  side_;
      cublasFillMode_t  uplo_;
      cublasOperation_t trans_;
      cublasDiagType_t  diag_;
      
      if(A_is_ll) {
        if ((side[0]=='L')||(side[0]=='l')) side_ = CUBLAS_SIDE_LEFT;
        else side_ = CUBLAS_SIDE_RIGHT;
        if ((uplo[0]=='L')||(uplo[0]=='l')) uplo_ = CUBLAS_FILL_MODE_LOWER;
        else uplo_ = CUBLAS_FILL_MODE_UPPER;
      } else {
        if ((side[0]=='L')||(side[0]=='l')) side_ = CUBLAS_SIDE_RIGHT;
        else side_ = CUBLAS_SIDE_LEFT;
        if ((uplo[0]=='L')||(uplo[0]=='l')) uplo_ = CUBLAS_FILL_MODE_UPPER;
        else uplo_ = CUBLAS_FILL_MODE_LOWER;
      }
      if ((trans[0]=='N')||(trans[0]=='n')) trans_ = CUBLAS_OP_N;
      else if ((trans[0]=='T')||(trans[0]=='t')) trans_ = CUBLAS_OP_T;
      else trans_ = CUBLAS_OP_C;
      if ((diag[0]=='U')||(diag[0]=='u')) diag_ = CUBLAS_DIAG_UNIT;
      else diag_ = CUBLAS_DIAG_NON_UNIT;
      
      if(A_is_ll)
        cublasCtrmm(handle, side_, uplo_, trans_, diag_, M, N, 
                    reinterpret_cast<const cuComplex*>(&alpha), 
                    reinterpret_cast<const cuComplex*>(A.data()), LDA, 
                    reinterpret_cast<const cuComplex*>(B.data()), LDB, 
                    reinterpret_cast<      cuComplex*>(B.data()), LDB);
      else
        cublasCtrmm(handle, side_, uplo_, trans_, diag_, N, M,
                    reinterpret_cast<const cuComplex*>(&alpha),
                    reinterpret_cast<const cuComplex*>(A.data()), LDA,
                    reinterpret_cast<const cuComplex*>(B.data()), LDB,
                    reinterpret_cast<      cuComplex*>(B.data()), LDB);
      
      cublasDestroy(handle);
    }
#endif
#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS
    if( std::is_same< MemSpace, Kokkos::HostSpace >::value ) {
      char  side_;
      char  uplo_;
      
      if(A_is_ll) {
        if ((side[0]=='L')||(side[0]=='l')) side_ = 'L';
        else side_ = 'R';
        if ((uplo[0]=='L')||(uplo[0]=='l')) uplo_ = 'L';
        else uplo_ = 'U';
      } else {
        if ((side[0]=='L')||(side[0]=='l')) side_ = 'R';
        else side_ = 'L';
        if ((uplo[0]=='L')||(uplo[0]=='l')) uplo_ = 'U';
        else uplo_ = 'L';
      }
      
      const std::complex<float> alpha_val = alpha;
      if(A_is_ll)
        KokkosBlas::Impl::HostBlas<std::complex<float> >::trmm(side_, uplo_, trans[0], diag[0], M, N, alpha_val, reinterpret_cast<const std::complex<float>*>(A.data()), LDA, reinterpret_cast<std::complex<float>*>(B.data()), LDB);
      else
        KokkosBlas::Impl::HostBlas<std::complex<float> >::trmm(side_, uplo_, trans[0], diag[0], N, M, alpha_val, reinterpret_cast<const std::complex<float>*>(A.data()), LDA, reinterpret_cast<std::complex<float>*>(B.data()), LDB);
   }
#endif
  }
  //Kokkos::complex<double>
  template<class ViewTypeA, class ViewTypeB, class ExecSpace, class MemSpace>
  void trmm_wrapper (const char side[],
                     const char uplo[],
                     const char trans[],
                     const char diag[],
                     Kokkos::complex<double>& alpha,
                     const ViewTypeA& A,
                     const ViewTypeB& B)
  {
    const int M = static_cast<int> (B.extent(0));
    const int N = static_cast<int> (B.extent(1));

    bool A_is_ll = std::is_same<Kokkos::LayoutLeft,typename ViewTypeA::array_layout>::value;
    bool B_is_ll = std::is_same<Kokkos::LayoutLeft,typename ViewTypeB::array_layout>::value;

    const int AST = A_is_ll?A.stride(1):A.stride(0), LDA = (AST == 0) ? 1 : AST;
    const int BST = B_is_ll?B.stride(1):B.stride(0), LDB = (BST == 0) ? 1 : BST;

#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS
    if( std::is_same< ExecSpace, Kokkos::Cuda >::value ) {
      cublasHandle_t handle;
      cublasStatus_t stat = cublasCreate(&handle);
      if (stat != CUBLAS_STATUS_SUCCESS)
        Kokkos::abort("CUBLAS initialization failed\n");
      
      cublasSideMode_t  side_;
      cublasFillMode_t  uplo_;
      cublasOperation_t trans_;
      cublasDiagType_t  diag_;
      
      if(A_is_ll) {
        if ((side[0]=='L')||(side[0]=='l')) side_ = CUBLAS_SIDE_LEFT;
        else side_ = CUBLAS_SIDE_RIGHT;
        if ((uplo[0]=='L')||(uplo[0]=='l')) uplo_ = CUBLAS_FILL_MODE_LOWER;
        else uplo_ = CUBLAS_FILL_MODE_UPPER;
      } else {
        if ((side[0]=='L')||(side[0]=='l')) side_ = CUBLAS_SIDE_RIGHT;
        else side_ = CUBLAS_SIDE_LEFT;
        if ((uplo[0]=='L')||(uplo[0]=='l')) uplo_ = CUBLAS_FILL_MODE_UPPER;
        else uplo_ = CUBLAS_FILL_MODE_LOWER;
      }
      if ((trans[0]=='N')||(trans[0]=='n')) trans_ = CUBLAS_OP_N;
      else if ((trans[0]=='T')||(trans[0]=='t')) trans_ = CUBLAS_OP_T;
      else trans_ = CUBLAS_OP_C;
      if ((diag[0]=='U')||(diag[0]=='u')) diag_ = CUBLAS_DIAG_UNIT;
      else diag_ = CUBLAS_DIAG_NON_UNIT;
      
      if(A_is_ll)
        cublasZtrmm(handle, side_, uplo_, trans_, diag_, M, N, 
                    reinterpret_cast<const cuDoubleComplex*>(&alpha), 
                    reinterpret_cast<const cuDoubleComplex*>(A.data()), LDA, 
                    reinterpret_cast<const cuDoubleComplex*>(B.data()), LDB, 
                    reinterpret_cast<      cuDoubleComplex*>(B.data()), LDB);
      else
        cublasZtrmm(handle, side_, uplo_, trans_, diag_, N, M,
                    reinterpret_cast<const cuDoubleComplex*>(&alpha),
                    reinterpret_cast<const cuDoubleComplex*>(A.data()), LDA,
                    reinterpret_cast<const cuDoubleComplex*>(B.data()), LDB,
                    reinterpret_cast<      cuDoubleComplex*>(B.data()), LDB);
      
      cublasDestroy(handle);
    }
#endif
#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS
    if( std::is_same< MemSpace, Kokkos::HostSpace >::value ) {
      char  side_;
      char  uplo_;
      
      if(A_is_ll) {
        if ((side[0]=='L')||(side[0]=='l')) side_ = 'L';
        else side_ = 'R';
        if ((uplo[0]=='L')||(uplo[0]=='l')) uplo_ = 'L';
        else uplo_ = 'U';
      } else {
        if ((side[0]=='L')||(side[0]=='l')) side_ = 'R';
        else side_ = 'L';
        if ((uplo[0]=='L')||(uplo[0]=='l')) uplo_ = 'U';
        else uplo_ = 'L';
      }
      
      const std::complex<double> alpha_val = alpha;
      if(A_is_ll)
        KokkosBlas::Impl::HostBlas<std::complex<double> >::trmm(side_, uplo_, trans[0], diag[0], M, N, alpha_val, reinterpret_cast<const std::complex<double>*>(A.data()), LDA, reinterpret_cast<std::complex<double>*>(B.data()), LDB);
      else
        KokkosBlas::Impl::HostBlas<std::complex<double> >::trmm(side_, uplo_, trans[0], diag[0], N, M, alpha_val, reinterpret_cast<const std::complex<double>*>(A.data()), LDA, reinterpret_cast<std::complex<double>*>(B.data()), LDB);
    }
#endif
  }

  //
  //
  //

  template<class ViewTypeA, class ViewTypeB, class Device>
  void impl_test_trsm(const char* side, const char* uplo, const char* trans, const char* diag, 
                      int M, int N, typename ViewTypeA::value_type alpha) {

    using execution_space = typename ViewTypeA::device_type::execution_space;
    using memory_space    = typename ViewTypeA::device_type::memory_space;
    using ScalarA         = typename ViewTypeA::value_type;
    using ScalarB         = typename ViewTypeB::value_type;
    using APT             = Kokkos::Details::ArithTraits<ScalarA>;
    using mag_type        = typename APT::mag_type;
    
    double machine_eps = APT::epsilon();
    bool A_l = (side[0]=='L') || (side[0]=='l');
    int K = A_l?M:N;

    //printf("KokkosBlas::trsm test for alpha %lf, %c %c %c %c, M %d, N %d, eps %.12lf, ViewType: %s\n", double(APT::abs(alpha)),side[0],uplo[0],trans[0],diag[0],M,N,1.0e10 * machine_eps,typeid(ViewTypeA).name());

    ViewTypeA A  ("A", K,K);
    ViewTypeB B  ("B", M,N);
    ViewTypeB X0 ("X0",M,N);

    typename ViewTypeB::HostMirror h_B  = Kokkos::create_mirror_view(B);
    typename ViewTypeB::HostMirror h_X0 = Kokkos::create_mirror_view(X0);

    uint64_t seed = Kokkos::Impl::clock_tic();
    Kokkos::Random_XorShift64_Pool<execution_space> rand_pool(seed);

    Kokkos::fill_random(A, rand_pool,ScalarA(0.01));
    if((diag[0]=='U')||(diag[0]=='u')) {
      using functor_type = UnitDiagTRSM<ViewTypeA,execution_space>;
      functor_type udtrsm(A);
      Kokkos::parallel_for("KokkosBlas::Test::UnitDiagTRSM", Kokkos::RangePolicy<execution_space>(0,K), udtrsm);
    } else {//(diag[0]=='N')||(diag[0]=='n')
      using functor_type = NonUnitDiagTRSM<ViewTypeA,execution_space>;
      functor_type nudtrsm(A);
      Kokkos::parallel_for("KokkosBlas::Test::NonUnitDiagTRSM", Kokkos::RangePolicy<execution_space>(0,K), nudtrsm);
    }
    Kokkos::deep_copy(X0, ScalarA(1));

    Kokkos::deep_copy(B, X0);

    ScalarA alpha_trmm = 1.0/alpha;

    Kokkos::fence();
 
    trmm_wrapper<ViewTypeA, ViewTypeB, execution_space, memory_space>(side, uplo, trans, diag, alpha_trmm, A, B);

    KokkosBlas::trsm(side, uplo, trans, diag, alpha, A, B);

    Kokkos::fence();

    Kokkos::deep_copy(h_B,  B);
    Kokkos::deep_copy(h_X0, X0);

    // Checking vs ref on CPU, this eps is about 10^-6
    const mag_type eps = 1.0e10 * machine_eps;
    bool test_flag = true;
    for (int i=0; i<M; i++) {
      for (int j=0; j<N; j++) {
        if ( APT::abs(h_B(i,j) - h_X0(i,j)) > eps ) {
          test_flag = false;
          //printf( "   Error: abs_result( %.15lf ) != abs_solution( %.15lf ) at (i %ld, j %ld)\n", APT::abs(h_B(i,j)), APT::abs(h_X0(i,j)), i, j );
          break;
        }
      }
      if (!test_flag) break;
    }
    ASSERT_EQ( test_flag, true );
  }
}

template<class ScalarA, class ScalarB, class Device>
int test_trsm(const char* mode, ScalarA alpha) {

#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  using view_type_a_ll = Kokkos::View<ScalarA**, Kokkos::LayoutLeft, Device>;
  using view_type_b_ll = Kokkos::View<ScalarB**, Kokkos::LayoutLeft, Device>;
  Test::impl_test_trsm<view_type_a_ll, view_type_b_ll, Device>(&mode[0],&mode[1],&mode[2],&mode[3],0,0,alpha);
  Test::impl_test_trsm<view_type_a_ll, view_type_b_ll, Device>(&mode[0],&mode[1],&mode[2],&mode[3],101,1,alpha);
  Test::impl_test_trsm<view_type_a_ll, view_type_b_ll, Device>(&mode[0],&mode[1],&mode[2],&mode[3],1,101,alpha);
  Test::impl_test_trsm<view_type_a_ll, view_type_b_ll, Device>(&mode[0],&mode[1],&mode[2],&mode[3],101,19,alpha);
  Test::impl_test_trsm<view_type_a_ll, view_type_b_ll, Device>(&mode[0],&mode[1],&mode[2],&mode[3],19,101,alpha);
  Test::impl_test_trsm<view_type_a_ll, view_type_b_ll, Device>(&mode[0],&mode[1],&mode[2],&mode[3],3031,91,alpha);
#endif

#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  using view_type_a_lr = Kokkos::View<ScalarA**, Kokkos::LayoutRight, Device>;
  using view_type_b_lr = Kokkos::View<ScalarB**, Kokkos::LayoutRight, Device>;
  Test::impl_test_trsm<view_type_a_lr, view_type_b_lr, Device>(&mode[0],&mode[1],&mode[2],&mode[3],0,0,alpha);
  Test::impl_test_trsm<view_type_a_lr, view_type_b_lr, Device>(&mode[0],&mode[1],&mode[2],&mode[3],101,1,alpha);
  Test::impl_test_trsm<view_type_a_lr, view_type_b_lr, Device>(&mode[0],&mode[1],&mode[2],&mode[3],1,101,alpha);
  Test::impl_test_trsm<view_type_a_lr, view_type_b_lr, Device>(&mode[0],&mode[1],&mode[2],&mode[3],101,19,alpha);
  Test::impl_test_trsm<view_type_a_lr, view_type_b_lr, Device>(&mode[0],&mode[1],&mode[2],&mode[3],19,101,alpha);
  Test::impl_test_trsm<view_type_a_lr, view_type_b_lr, Device>(&mode[0],&mode[1],&mode[2],&mode[3],3031,91,alpha);
#endif

  return 1;
}

#if defined( KOKKOSKERNELS_ENABLE_TPL_CUBLAS ) || defined (KOKKOSKERNELS_ENABLE_TPL_BLAS)

#if defined(KOKKOSKERNELS_INST_FLOAT) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F( TestCategory, trsm_float ) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::trsm_float");
    float alpha = 1.0f;
    test_trsm<float,float,TestExecSpace> ("LLNN",alpha);
    test_trsm<float,float,TestExecSpace> ("LLNU",alpha);
    test_trsm<float,float,TestExecSpace> ("LLTN",alpha);
    test_trsm<float,float,TestExecSpace> ("LLTU",alpha);
    test_trsm<float,float,TestExecSpace> ("LUNN",alpha);
    test_trsm<float,float,TestExecSpace> ("LUNU",alpha);
    test_trsm<float,float,TestExecSpace> ("LUTN",alpha);
    test_trsm<float,float,TestExecSpace> ("LUTU",alpha);

    test_trsm<float,float,TestExecSpace> ("RLNN",alpha);
    test_trsm<float,float,TestExecSpace> ("RLNU",alpha);
    test_trsm<float,float,TestExecSpace> ("RLTN",alpha);
    test_trsm<float,float,TestExecSpace> ("RLTU",alpha);
    test_trsm<float,float,TestExecSpace> ("RUNN",alpha);
    test_trsm<float,float,TestExecSpace> ("RUNU",alpha);
    test_trsm<float,float,TestExecSpace> ("RUTN",alpha);
    test_trsm<float,float,TestExecSpace> ("RUTU",alpha);

    alpha = 4.5f;
    test_trsm<float,float,TestExecSpace> ("LLNN",alpha);
    test_trsm<float,float,TestExecSpace> ("LLNU",alpha);
    test_trsm<float,float,TestExecSpace> ("LLTN",alpha);
    test_trsm<float,float,TestExecSpace> ("LLTU",alpha);
    test_trsm<float,float,TestExecSpace> ("LUNN",alpha);
    test_trsm<float,float,TestExecSpace> ("LUNU",alpha);
    test_trsm<float,float,TestExecSpace> ("LUTN",alpha);
    test_trsm<float,float,TestExecSpace> ("LUTU",alpha);

    test_trsm<float,float,TestExecSpace> ("RLNN",alpha);
    test_trsm<float,float,TestExecSpace> ("RLNU",alpha);
    test_trsm<float,float,TestExecSpace> ("RLTN",alpha);
    test_trsm<float,float,TestExecSpace> ("RLTU",alpha);
    test_trsm<float,float,TestExecSpace> ("RUNN",alpha);
    test_trsm<float,float,TestExecSpace> ("RUNU",alpha);
    test_trsm<float,float,TestExecSpace> ("RUTN",alpha);
    test_trsm<float,float,TestExecSpace> ("RUTU",alpha);
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F( TestCategory, trsm_double ) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::trsm_double");
    double alpha = 1.0;
    test_trsm<double,double,TestExecSpace> ("LLNN",alpha);
    test_trsm<double,double,TestExecSpace> ("LLNU",alpha);
    test_trsm<double,double,TestExecSpace> ("LLTN",alpha);
    test_trsm<double,double,TestExecSpace> ("LLTU",alpha);
    test_trsm<double,double,TestExecSpace> ("LUNN",alpha);
    test_trsm<double,double,TestExecSpace> ("LUNU",alpha);
    test_trsm<double,double,TestExecSpace> ("LUTN",alpha);
    test_trsm<double,double,TestExecSpace> ("LUTU",alpha);

    test_trsm<double,double,TestExecSpace> ("RLNN",alpha);
    test_trsm<double,double,TestExecSpace> ("RLNU",alpha);
    test_trsm<double,double,TestExecSpace> ("RLTN",alpha);
    test_trsm<double,double,TestExecSpace> ("RLTU",alpha);
    test_trsm<double,double,TestExecSpace> ("RUNN",alpha);
    test_trsm<double,double,TestExecSpace> ("RUNU",alpha);
    test_trsm<double,double,TestExecSpace> ("RUTN",alpha);
    test_trsm<double,double,TestExecSpace> ("RUTU",alpha);

    alpha = 4.5;
    test_trsm<double,double,TestExecSpace> ("LLNN",alpha);
    test_trsm<double,double,TestExecSpace> ("LLNU",alpha);
    test_trsm<double,double,TestExecSpace> ("LLTN",alpha);
    test_trsm<double,double,TestExecSpace> ("LLTU",alpha);
    test_trsm<double,double,TestExecSpace> ("LUNN",alpha);
    test_trsm<double,double,TestExecSpace> ("LUNU",alpha);
    test_trsm<double,double,TestExecSpace> ("LUTN",alpha);
    test_trsm<double,double,TestExecSpace> ("LUTU",alpha);

    test_trsm<double,double,TestExecSpace> ("RLNN",alpha);
    test_trsm<double,double,TestExecSpace> ("RLNU",alpha);
    test_trsm<double,double,TestExecSpace> ("RLTN",alpha);
    test_trsm<double,double,TestExecSpace> ("RLTU",alpha);
    test_trsm<double,double,TestExecSpace> ("RUNN",alpha);
    test_trsm<double,double,TestExecSpace> ("RUNU",alpha);
    test_trsm<double,double,TestExecSpace> ("RUTN",alpha);
    test_trsm<double,double,TestExecSpace> ("RUTU",alpha);
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F( TestCategory, trsm_complex_double ) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::trsm_complex_double");
    Kokkos::complex<double> alpha = 1.0;
    test_trsm<Kokkos::complex<double>,Kokkos::complex<double>,TestExecSpace> ("LLNN",alpha);
    test_trsm<Kokkos::complex<double>,Kokkos::complex<double>,TestExecSpace> ("LLNU",alpha);
    test_trsm<Kokkos::complex<double>,Kokkos::complex<double>,TestExecSpace> ("LLCN",alpha);
    test_trsm<Kokkos::complex<double>,Kokkos::complex<double>,TestExecSpace> ("LLCU",alpha);
    test_trsm<Kokkos::complex<double>,Kokkos::complex<double>,TestExecSpace> ("LUNN",alpha);
    test_trsm<Kokkos::complex<double>,Kokkos::complex<double>,TestExecSpace> ("LUNU",alpha);
    test_trsm<Kokkos::complex<double>,Kokkos::complex<double>,TestExecSpace> ("LUCN",alpha);
    test_trsm<Kokkos::complex<double>,Kokkos::complex<double>,TestExecSpace> ("LUCU",alpha);

    test_trsm<Kokkos::complex<double>,Kokkos::complex<double>,TestExecSpace> ("RLNN",alpha);
    test_trsm<Kokkos::complex<double>,Kokkos::complex<double>,TestExecSpace> ("RLNU",alpha);
    test_trsm<Kokkos::complex<double>,Kokkos::complex<double>,TestExecSpace> ("RLCN",alpha);
    test_trsm<Kokkos::complex<double>,Kokkos::complex<double>,TestExecSpace> ("RLCU",alpha);
    test_trsm<Kokkos::complex<double>,Kokkos::complex<double>,TestExecSpace> ("RUNN",alpha);
    test_trsm<Kokkos::complex<double>,Kokkos::complex<double>,TestExecSpace> ("RUNU",alpha);
    test_trsm<Kokkos::complex<double>,Kokkos::complex<double>,TestExecSpace> ("RUCN",alpha);
    test_trsm<Kokkos::complex<double>,Kokkos::complex<double>,TestExecSpace> ("RUCU",alpha);

    alpha = Kokkos::complex<double>(4.5,0.0);
    test_trsm<Kokkos::complex<double>,Kokkos::complex<double>,TestExecSpace> ("LLNN",alpha);
    test_trsm<Kokkos::complex<double>,Kokkos::complex<double>,TestExecSpace> ("LLNU",alpha);
    test_trsm<Kokkos::complex<double>,Kokkos::complex<double>,TestExecSpace> ("LLCN",alpha);
    test_trsm<Kokkos::complex<double>,Kokkos::complex<double>,TestExecSpace> ("LLCU",alpha);
    test_trsm<Kokkos::complex<double>,Kokkos::complex<double>,TestExecSpace> ("LUNN",alpha);
    test_trsm<Kokkos::complex<double>,Kokkos::complex<double>,TestExecSpace> ("LUNU",alpha);
    test_trsm<Kokkos::complex<double>,Kokkos::complex<double>,TestExecSpace> ("LUCN",alpha);
    test_trsm<Kokkos::complex<double>,Kokkos::complex<double>,TestExecSpace> ("LUCU",alpha);

    test_trsm<Kokkos::complex<double>,Kokkos::complex<double>,TestExecSpace> ("RLNN",alpha);
    test_trsm<Kokkos::complex<double>,Kokkos::complex<double>,TestExecSpace> ("RLNU",alpha);
    test_trsm<Kokkos::complex<double>,Kokkos::complex<double>,TestExecSpace> ("RLCN",alpha);
    test_trsm<Kokkos::complex<double>,Kokkos::complex<double>,TestExecSpace> ("RLCU",alpha);
    test_trsm<Kokkos::complex<double>,Kokkos::complex<double>,TestExecSpace> ("RUNN",alpha);
    test_trsm<Kokkos::complex<double>,Kokkos::complex<double>,TestExecSpace> ("RUNU",alpha);
    test_trsm<Kokkos::complex<double>,Kokkos::complex<double>,TestExecSpace> ("RUCN",alpha);
    test_trsm<Kokkos::complex<double>,Kokkos::complex<double>,TestExecSpace> ("RUCU",alpha);
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_FLOAT) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F( TestCategory, trsm_complex_float ) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::trsm_complex_float");
    Kokkos::complex<float> alpha = 5.0f;
    test_trsm<Kokkos::complex<float>,Kokkos::complex<float>,TestExecSpace> ("LLNN",alpha);
    test_trsm<Kokkos::complex<float>,Kokkos::complex<float>,TestExecSpace> ("LLNU",alpha);
    test_trsm<Kokkos::complex<float>,Kokkos::complex<float>,TestExecSpace> ("LLCN",alpha);
    test_trsm<Kokkos::complex<float>,Kokkos::complex<float>,TestExecSpace> ("LLCU",alpha);
    test_trsm<Kokkos::complex<float>,Kokkos::complex<float>,TestExecSpace> ("LUNN",alpha);
    test_trsm<Kokkos::complex<float>,Kokkos::complex<float>,TestExecSpace> ("LUNU",alpha);
    test_trsm<Kokkos::complex<float>,Kokkos::complex<float>,TestExecSpace> ("LUCN",alpha);
    test_trsm<Kokkos::complex<float>,Kokkos::complex<float>,TestExecSpace> ("LUCU",alpha);

    test_trsm<Kokkos::complex<float>,Kokkos::complex<float>,TestExecSpace> ("RLNN",alpha);
    test_trsm<Kokkos::complex<float>,Kokkos::complex<float>,TestExecSpace> ("RLNU",alpha);
    test_trsm<Kokkos::complex<float>,Kokkos::complex<float>,TestExecSpace> ("RLCN",alpha);
    test_trsm<Kokkos::complex<float>,Kokkos::complex<float>,TestExecSpace> ("RLCU",alpha);
    test_trsm<Kokkos::complex<float>,Kokkos::complex<float>,TestExecSpace> ("RUNN",alpha);
    test_trsm<Kokkos::complex<float>,Kokkos::complex<float>,TestExecSpace> ("RUNU",alpha);
    test_trsm<Kokkos::complex<float>,Kokkos::complex<float>,TestExecSpace> ("RUCN",alpha);
    test_trsm<Kokkos::complex<float>,Kokkos::complex<float>,TestExecSpace> ("RUCU",alpha);

    alpha = Kokkos::complex<float>(4.5f,0.0f);
    test_trsm<Kokkos::complex<float>,Kokkos::complex<float>,TestExecSpace> ("LLNN",alpha);
    test_trsm<Kokkos::complex<float>,Kokkos::complex<float>,TestExecSpace> ("LLNU",alpha);
    test_trsm<Kokkos::complex<float>,Kokkos::complex<float>,TestExecSpace> ("LLCN",alpha);
    test_trsm<Kokkos::complex<float>,Kokkos::complex<float>,TestExecSpace> ("LLCU",alpha);
    test_trsm<Kokkos::complex<float>,Kokkos::complex<float>,TestExecSpace> ("LUNN",alpha);
    test_trsm<Kokkos::complex<float>,Kokkos::complex<float>,TestExecSpace> ("LUNU",alpha);
    test_trsm<Kokkos::complex<float>,Kokkos::complex<float>,TestExecSpace> ("LUCN",alpha);
    test_trsm<Kokkos::complex<float>,Kokkos::complex<float>,TestExecSpace> ("LUCU",alpha);

    test_trsm<Kokkos::complex<float>,Kokkos::complex<float>,TestExecSpace> ("RLNN",alpha);
    test_trsm<Kokkos::complex<float>,Kokkos::complex<float>,TestExecSpace> ("RLNU",alpha);
    test_trsm<Kokkos::complex<float>,Kokkos::complex<float>,TestExecSpace> ("RLCN",alpha);
    test_trsm<Kokkos::complex<float>,Kokkos::complex<float>,TestExecSpace> ("RLCU",alpha);
    test_trsm<Kokkos::complex<float>,Kokkos::complex<float>,TestExecSpace> ("RUNN",alpha);
    test_trsm<Kokkos::complex<float>,Kokkos::complex<float>,TestExecSpace> ("RUNU",alpha);
    test_trsm<Kokkos::complex<float>,Kokkos::complex<float>,TestExecSpace> ("RUCN",alpha);
    test_trsm<Kokkos::complex<float>,Kokkos::complex<float>,TestExecSpace> ("RUCU",alpha);
  Kokkos::Profiling::popRegion();
}
#endif

#endif//KOKKOSKERNELS_ENABLE_TPL_CUBLAS || KOKKOSKERNELS_ENABLE_TPL_BLAS
