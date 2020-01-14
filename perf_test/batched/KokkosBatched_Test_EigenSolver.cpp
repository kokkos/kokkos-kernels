/// Kokkos headers
#include "Kokkos_Core.hpp"
#include "impl/Kokkos_Timer.hpp"
#include "Kokkos_Random.hpp"

#if  defined(KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA)
#if !defined(KOKKOS_ENABLE_CUDA) || (8000 <= CUDA_VERSION)
#define KOKKOSBATCHED_TEST_EIGENSOLVER
#endif
#endif

#if defined(KOKKOSBATCHED_TEST_EIGENSOLVER)

#include "KokkosBatched_Util.hpp"

#define KOKKOSBATCHED_PROFILE 1
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSBATCHED_PROFILE)
#include "cuda_profiler_api.h"
#endif

typedef Kokkos::DefaultExecutionSpace exec_space;
typedef typename exec_space::memory_space memory_space;
typedef Kokkos::DefaultHostExecutionSpace host_space;
typedef typename host_space::memory_space host_memory_space;

typedef Kokkos::LayoutRight layout_right;
typedef double value_type;

//using namespace KokkosBatched;

namespace PerfTest {
  struct Problem {
    using value_type_3d_view_exec = Kokkos::View<value_type***,             exec_space>;
    using value_type_3d_view_host = Kokkos::View<value_type***,layout_right,host_space>;

    int _N, _Blk;

    value_type_3d_view_exec _A_kokkos;
    value_type_3d_view_host _A_host;

    Problem() {}

    int setRandom(const int N, const int Blk) {
      _N   = N   < 0 ?  1 :   N;
      _Blk = Blk < 0 ? 11 : Blk;

      _A_kokkos = value_type_3d_view_exec("A_mat_kokkos", _N, _Blk, _Blk);
      _A_host   = value_type_3d_view_host("A_mat_host",   _N, _Blk, _Blk);

      const value_type one(1);
      Kokkos::Random_XorShift64_Pool<host_space> random(13245);
      Kokkos::fill_random(_A_host, random, one);

      auto A_kokkos_host = Kokkos::create_mirror_view(host_memory_space(), _A_kokkos);
      Kokkos::RangePolicy<host_space> policy(0, _N);
      Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int i) {
          for (int j=0,jend=A_kokkos_host.extent(1);j<jend;++j)
            for (int k=0,kend=A_kokkos_host.extent(2);k<kend;++k)
              A_kokkos_host(i,j,k) = _A_host(i,j,k);
        });
      Kokkos::deep_copy(_A_kokkos, A_kokkos_host);
      return 0;
    }

    int setFromFile(const char *name, const int N) {
      std::ifstream infile(name);
      if (!infile.is_open()) {
        printf("Error: oepning file %s\n", name);
        return -1;
      } else {
        _N = N < 0 ? 1 : N;
        infile >> _Blk;
        _A_host = value_type_3d_view_host("A_host", _N, _Blk, _Blk);
        for (int i=0;i<_Blk;++i)
          for (int j=0;j<_Blk;++j)
            infile >> _A_host(0, i, j);
        if (_N > 1) { /// clone A into N host array
          Kokkos::RangePolicy<host_space> policy(1, _N);
          Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int i) {
              for (int j=0,jend=_A_host.extent(1);j<jend;++j)
                for (int k=0,kend=_A_host.extent(2);k<kend;++k)
                  _A_host(i,j,k) = _A_host(0,j,k);
            });
        }
        { /// copy A into A mirror
          auto A_kokkos_host = Kokkos::create_mirror_view(typename host_space::memory_space(), _A_kokkos);
          Kokkos::RangePolicy<host_space> policy(0, _N);
          Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int i) {
              for (int j=0,jend=A_kokkos_host.extent(1);j<jend;++j)
                for (int k=0,kend=A_kokkos_host.extent(2);k<kend;++k)
                  A_kokkos_host(i,j,k) = _A_host(i,j,k);
            });
          Kokkos::deep_copy(_A_kokkos, A_kokkos_host);
        }
      }
      return 0;
    }
  };

#if defined(__KOKKOSBATCHED_INTEL_MKL__)
  struct TestMKL {
    using value_type_2d_view = Kokkos::View<value_type**,   layout_right,host_space>;
    using value_type_3d_view = Kokkos::View<value_type***,  layout_right,host_space>;
    using value_type_4d_view = Kokkos::View<value_type****, layout_right,host_space>;

    int _N, _Blk;

    value_type_3d_view _A; /// N, Blk, Blk
    value_type_3d_view _E; /// N, 2, Blk
    value_type_4d_view _V; /// N, 2, Blk, Blk
    value_type_2d_view _W; /// N, getWorkSpaceSize

    int getWorkSpaceSize() {
      int lwork_mkl = -1;
      {
        double work_query;
        LAPACKE_dgeev_work(LAPACK_ROW_MAJOR,
                           //'N', 'V',
                           'V', 'V',
                           _Blk,
                           NULL, _Blk,
                           NULL, NULL,
                           NULL, _Blk,
                           NULL, _Blk,
                           &work_query,
                           lwork_mkl);
        lwork_mkl = int(work_query);
      }
      return lwork_mkl;
    }

    template<typename ArgViewType>
    void setProblem(const ArgViewType &A) {
      const value_type zero(0);
      Kokkos::deep_copy(_A, A);
      Kokkos::deep_copy(_E, zero);
      Kokkos::deep_copy(_V, zero);
      Kokkos::deep_copy(_W, zero);
    }

    inline void operator()(const int &i) const {
      LAPACKE_dgeev_work(LAPACK_ROW_MAJOR,
                         //'N', 'V',
                         'V', 'V',
                         _Blk,
                         (value_type*)&_A(i,0,0), int(_A.stride(1)),
                         &_E(i,0,0), &_E(i,1,0),
                         &_V(i,0,0,0), int(_V.stride(2)),
                         &_V(i,1,0,0), int(_V.stride(2)),
                         &_W(i,0), int(_W.extent(1)));
    }

    double runTest() {
      Kokkos::Impl::Timer timer;
      timer.reset();
      {
        Kokkos::RangePolicy<host_space> policy(0, _N);
        Kokkos::parallel_for(policy, *this);
        Kokkos::fence();
      }
      const double t = timer.seconds();
      return t;
    }

    TestMKL(const int N, const int Blk)
      : _N(N),
        _Blk(Blk),
        _A("A_mkl", N, Blk, Blk),
        _E("E_mkl", N, 2, Blk),
        _V("V_mkl", N, 2, Blk, Blk),
        _W("W_mkl", N, getWorkSpaceSize()) {}
  };
#endif

  struct TestCheck {
    using value_type_3d_view = Kokkos::View<value_type***, layout_right,host_space>;
    using value_type_4d_view = Kokkos::View<value_type****,layout_right,host_space>;

    using complex_value_type_2d_view = Kokkos::View<std::complex<value_type>**  ,layout_right,host_space>;
    using complex_value_type_3d_view = Kokkos::View<std::complex<value_type>*** ,layout_right,host_space>;
    using complex_value_type_4d_view = Kokkos::View<std::complex<value_type>****,layout_right,host_space>;

    std::string _name;

    int _N, _Blk;
    bool _vl_stores_col_vectors;

    value_type_3d_view _A_problem;
    value_type_3d_view _E;
    value_type_4d_view _V;

    complex_value_type_2d_view _Ec;
    complex_value_type_4d_view _Vc;
    complex_value_type_3d_view _Ac;

    struct ConvertToComplexTag {};
    struct CheckLeftEigenvectorTag {};
    struct CheckRightEigenvectorTag {};

    inline
    void operator()(const ConvertToComplexTag &, const int &i) const {
      const value_type zero(0);

      // for convenience, create a complex eigenvalues and eigenvectors
      auto er = Kokkos::subview(_E, i, 0, Kokkos::ALL());
      auto ei = Kokkos::subview(_E, i, 1, Kokkos::ALL());
      auto VL = Kokkos::subview(_V, i, 0, Kokkos::ALL(), Kokkos::ALL());
      auto VR = Kokkos::subview(_V, i, 1, Kokkos::ALL(), Kokkos::ALL());

      for (int l=0;l<_Blk;) {
        auto e  = Kokkos::subview(_Ec, i, l);
        auto vl = Kokkos::subview(_Vc, i, 0, Kokkos::ALL(), l);
        auto vr = Kokkos::subview(_Vc, i, 1, Kokkos::ALL(), l);

        if (ei(l) == zero) {
          // real eigenvalue
          e() = std::complex<value_type>(er(l), ei(l));
          for (int k=0;k<_Blk;++k) {
            vl(k) = _vl_stores_col_vectors ? VL(k,l) : VL(l,k);
            vr(k) = VR(k,l);
          }
          l += 1;
        } else {
          // complex eigenvalues
          auto ep0 = e;
          auto ep1 = Kokkos::subview(_Ec, i, l+1);

          ep0() = std::complex<value_type>(er(l  ), ei(l  ));
          ep1() = std::complex<value_type>(er(l+1), ei(l+1));

          auto vl0 = vl;
          auto vr0 = vr;
          auto vl1 = Kokkos::subview(_Vc, i, 0, Kokkos::ALL(), l+1);
          auto vr1 = Kokkos::subview(_Vc, i, 1, Kokkos::ALL(), l+1);

          for (int k=0;k<_Blk;++k) {
            const value_type vl_kl  = _vl_stores_col_vectors ? VL(k,l  ) :  VL(l  ,k);
            const value_type vl_klp = _vl_stores_col_vectors ? VL(k,l+1) : -VL(l+1,k);
            vl0(k) = std::complex<value_type>(vl_kl,  vl_klp);
            vl1(k) = std::complex<value_type>(vl_kl, -vl_klp);
            vr0(k) = std::complex<value_type>(VR(k,l),  VR(k,l+1));
            vr1(k) = std::complex<value_type>(VR(k,l), -VR(k,l+1));
          }
          l += 2;
        }
      }
    }

    inline
    void operator()(const CheckLeftEigenvectorTag &, const int &i) const {
      auto Ac = Kokkos::subview(_Ac,        i,    Kokkos::ALL(), Kokkos::ALL());
      auto Ap = Kokkos::subview(_A_problem, i,    Kokkos::ALL(), Kokkos::ALL());

      auto e  = Kokkos::subview(_Ec       , i,    Kokkos::ALL());
      auto VL = Kokkos::subview(_Vc       , i, 0, Kokkos::ALL(), Kokkos::ALL());

      // set Ac = VL'*A
      for (int k0=0;k0<_Blk;++k0)
        for (int k1=0;k1<_Blk;++k1) {
          std::complex<value_type> tmp(0);
          for (int p=0;p<_Blk;++p)
            tmp += std::conj(VL(p,k0))*Ap(p,k1);
          Ac(k0,k1) = tmp;
        }

      // check Ac - E VL' = 0
      for (int k0=0;k0<_Blk;++k0)
        for (int k1=0;k1<_Blk;++k1)
          Ac(k0,k1) -= e(k0)*std::conj(VL(k1,k0));
    }

    inline
    void operator()(const CheckRightEigenvectorTag &, const int &i) const {
      auto Ac = Kokkos::subview(_Ac,        i,    Kokkos::ALL(), Kokkos::ALL());
      auto Ap = Kokkos::subview(_A_problem, i,    Kokkos::ALL(), Kokkos::ALL());

      auto e  = Kokkos::subview(_Ec       , i,    Kokkos::ALL());
      auto VR = Kokkos::subview(_Vc       , i, 1, Kokkos::ALL(), Kokkos::ALL());

      // set Ac = A*VR
      for (int k0=0;k0<_Blk;++k0)
        for (int k1=0;k1<_Blk;++k1) {
          std::complex<value_type> tmp(0);
          for (int p=0;p<_Blk;++p)
            tmp += Ap(k0,p)*VR(p,k1);
          Ac(k0,k1) = tmp;
        }

      // check Ac - VR E   = 0
      for (int k0=0;k0<_Blk;++k0)
        for (int k1=0;k1<_Blk;++k1)
          Ac(k0,k1) -= VR(k0,k1)*e(k1);
    }

    template<typename MViewType>
    double computeNormSquared(const MViewType &M) {
      double norm = 0;
      for (int k=0;k<_N;++k)
        for (int i=0;i<_Blk;++i)
          for (int j=0;j<_Blk;++j) {
            const auto val = std::abs(M(k,i,j));
            norm += val*val;
          }
      return norm;
    }
    std::pair<bool,bool> checkTest(double tol = 1e-6) {
      // reconstruct matrix and compute diff
      Kokkos::parallel_for(Kokkos::RangePolicy<host_space,ConvertToComplexTag>(0, _N), *this);
      Kokkos::fence();

      const double q = _N;
      const double norm_ref = computeNormSquared(_A_problem);

      Kokkos::parallel_for(Kokkos::RangePolicy<host_space,CheckLeftEigenvectorTag>(0, _N), *this);
      Kokkos::fence();
      const double norm_left = computeNormSquared(_Ac);

      Kokkos::parallel_for(Kokkos::RangePolicy<host_space,CheckRightEigenvectorTag>(0, _N), *this);
      Kokkos::fence();
      const double norm_right = computeNormSquared(_Ac);

      const bool left_pass  = std::sqrt(norm_left /norm_ref/q) < tol;
      const bool right_pass = std::sqrt(norm_right/norm_ref/q) < tol;

      printf(" --- Testing %s\n", _name.data());
      printf(" --- VL^H*A - E*VL^H: ref norm %e, diff %e\n", norm_ref, norm_left);
      printf(" --- A*VR - VR*E    : ref norm %e, diff %e\n", norm_ref, norm_right);

      return std::pair<bool,bool>(left_pass, right_pass);
    }

    template<typename AViewType,
             typename EViewType,
             typename VViewType>
    TestCheck(std::string name, 
              const int N, const int Blk,
              const AViewType &A_problem,
              const EViewType &E,
              const VViewType &V,
              const bool vl_stores_col_vectors)
      : _name(name), 
        _N(N),
        _Blk(Blk),
        _A_problem("A_problem_check", N, Blk, Blk),
        _E("E_check", N, 2, Blk),
        _V("V_check", N, 2, Blk, Blk),
        _Ec("Ec_mkl", N, Blk),
        _Vc("Vc_mkl", N, 2, Blk, Blk),
        _Ac("Ac_mkl", N, Blk, Blk),
        _vl_stores_col_vectors(vl_stores_col_vectors) {
      auto A_tmp = Kokkos::create_mirror_view_and_copy(host_memory_space(), A_problem);
      auto E_tmp = Kokkos::create_mirror_view_and_copy(host_memory_space(), E);
      auto V_tmp = Kokkos::create_mirror_view_and_copy(host_memory_space(), V);

      Kokkos::deep_copy(_A_problem, A_tmp);
      Kokkos::deep_copy(_E, E_tmp);
      Kokkos::deep_copy(_V, V_tmp);
    }
  };
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSBATCHED_PROFILE)
    cudaProfilerStop();
#endif
    Kokkos::print_configuration(std::cout);

    ///
    /// input arguments parsing
    ///
    int N = -1; /// # of problems (batch size)
    int Blk = -1;     /// block dimension
    char *filename = NULL;
    double tol = 1e-6;
    for (int i=1;i<argc;++i) {
      const std::string& token = argv[i];
      if (token == std::string("-N")) N = std::atoi(argv[++i]);
      if (token == std::string("-B")) Blk = std::atoi(argv[++i]);
      if (token == std::string("-F")) filename = argv[++i];
      if (token == std::string("-T")) tol = std::atof(argv[++i]);
    }
    const int niter_beg = -2, niter_end = 3;

    ///
    /// problem setting
    ///
    PerfTest::Problem problem;
    {
      if (filename == NULL)
        problem.setRandom(N, Blk);
      else
        problem.setFromFile(filename, N);
      printf(" :::: Testing (filename = %s, N = %d, Blk = %d, niter = %d)\n",
             filename == NULL ? "random" : filename, problem._N, problem._Blk, niter_end);
    }

    ///
    /// MKL testing if available
    ///
#if defined(__KOKKOSBATCHED_INTEL_MKL__)
    {
      PerfTest::TestMKL eig_mkl(problem._N, problem._Blk);
      double t_mkl(0);
      for (int iter=niter_beg;iter<niter_end;++iter) {
        eig_mkl.setProblem(problem._A_host);
        const double t = eig_mkl.runTest();
        t_mkl += (iter >= 0)*t;
      }

      PerfTest::TestCheck check("MKL", 
                                problem._N,
                                problem._Blk,
                                problem._A_host,
                                eig_mkl._E,
                                eig_mkl._V,
                                true);
      const auto pass = check.checkTest(tol);
      printf("MKL Test\n");
      printf("========\n");
      printf("MKL           Eigensolver left  test %s with a tol %e\n", (pass.first  ? "passed" : "fail"), tol);
      printf("MKL           Eigensolver right test %s with a tol %e\n", (pass.second ? "passed" : "fail"), tol);
      
      const double t_mkl_per_problem = (t_mkl/double(niter_end*problem._N));
      const double n_mkl_problems_per_second = 1.0/t_mkl_per_problem;
      printf("MKL           Eigensolver Time: %e seconds , %e seconds per problem , %e problems per second\n", t_mkl, t_mkl_per_problem, n_mkl_problems_per_second);
    }
#endif

  }
  Kokkos::finalize();

  return 0;
}

#else
int main() {
  return 0;
}
#endif

