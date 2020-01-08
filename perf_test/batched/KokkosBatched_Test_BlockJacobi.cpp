/// Kokkos headers
#include "Kokkos_Core.hpp"
#include "impl/Kokkos_Timer.hpp"
#include "Kokkos_Random.hpp"

#if  defined(KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA)
#if !defined(KOKKOS_ENABLE_CUDA) || (8000 <= CUDA_VERSION)
#define KOKKOSBATCHED_TEST_BLOCKJACOBI
#endif 
#endif


#if defined(KOKKOSBATCHED_TEST_BLOCKJACOBI)

/// KokkosKernels headers
#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_Vector.hpp"

#include <Kokkos_ArithTraits.hpp>
#include <KokkosBatched_Util.hpp>
#include <KokkosBatched_Vector.hpp>
#include <KokkosBatched_Copy_Decl.hpp>
#include <KokkosBatched_Copy_Impl.hpp>
#include <KokkosBatched_SetIdentity_Decl.hpp>
#include <KokkosBatched_SetIdentity_Impl.hpp>
#include <KokkosBatched_AddRadial_Decl.hpp>
#include <KokkosBatched_AddRadial_Impl.hpp>
#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBatched_Gemm_Serial_Impl.hpp>
#include <KokkosBatched_Gemm_Team_Impl.hpp>
#include <KokkosBatched_Gemv_Decl.hpp>
#include <KokkosBatched_Gemv_Serial_Impl.hpp>
#include <KokkosBatched_Gemv_Team_Impl.hpp>
#include <KokkosBatched_Trsm_Decl.hpp>
#include <KokkosBatched_Trsm_Serial_Impl.hpp>
#include <KokkosBatched_Trsm_Team_Impl.hpp>
#include <KokkosBatched_Trsv_Decl.hpp>
#include <KokkosBatched_Trsv_Serial_Impl.hpp>
#include <KokkosBatched_Trsv_Team_Impl.hpp>
#include <KokkosBatched_LU_Decl.hpp>
#include <KokkosBatched_LU_Serial_Impl.hpp>
#include <KokkosBatched_LU_Team_Impl.hpp>
#include <KokkosBatched_SolveLU_Decl.hpp>

#define KOKKOSBATCHED_PROFILE 1
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSBATCHED_PROFILE)
#include "cuda_profiler_api.h"
#endif

#define KOKKOSBATCHED_USE_128BIT_MEMORY_INST

typedef Kokkos::DefaultExecutionSpace exec_space;
typedef typename exec_space::memory_space memory_space;
typedef Kokkos::DefaultHostExecutionSpace host_space;

typedef double value_type;

/// 128*128*128/16*5 * (2*8) / 16
///
/// simd typedefs
///
using namespace KokkosBatched;

static constexpr int vector_length = DefaultVectorLength<value_type,memory_space>::value;
#if defined(KOKKOSBATCHED_USE_128BIT_MEMORY_INST)
static constexpr int internal_vector_length = DefaultInternalVectorLength<value_type,memory_space>::value;
#else
static constexpr int internal_vector_length = 1;
#endif

typedef Vector<SIMD<value_type>,vector_length> vector_type;
#if defined(KOKKOSBATCHED_USE_128BIT_MEMORY_INST)
typedef Vector<SIMD<value_type>,internal_vector_length> internal_vector_type;
#else
typedef value_type internal_vector_type;
#endif

template<typename ActiveMemorySpace>
struct FactorizeModeAndAlgo;

template<>
struct FactorizeModeAndAlgo<Kokkos::HostSpace> {
  typedef Mode::Serial mode_type;
  typedef Algo::Level3::Blocked algo_type;   
};

#if defined(KOKKOS_ENABLE_CUDA)
template<>
struct FactorizeModeAndAlgo<Kokkos::CudaSpace> {
  typedef Mode::Team mode_type;
  typedef Algo::Level3::Unblocked algo_type;   
};
#endif
template<typename ActiveMemorySpace>
using SolveMultipleModeAndAlgo = FactorizeModeAndAlgo<ActiveMemorySpace>;

template<typename ActiveMemorySpace>
struct SolveSingleModeAndAlgo;

template<>
struct SolveSingleModeAndAlgo<Kokkos::HostSpace> {
  typedef Mode::Serial mode_type;
  typedef Algo::Level2::Blocked algo_type;   
};

#if defined(KOKKOS_ENABLE_CUDA)
template<>
struct SolveSingleModeAndAlgo<Kokkos::CudaSpace> {
  typedef Mode::Team mode_type;
  typedef Algo::Level2::Unblocked algo_type;   
};
#endif

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSBATCHED_PROFILE)
    cudaProfilerStop();
#endif
    Kokkos::print_configuration(std::cout);

    //typedef Kokkos::Details::ArithTraits<value_type> ats;
    Kokkos::Impl::Timer timer;

    ///
    /// input arguments parsing
    ///
    int N = 128*128*128; /// # of problems (batch size)
    int Blk = 5;     /// block dimension
    int Nvec = 1;
    int S = 0; /// scratch size
    int niter = 1;
    for (int i=1;i<argc;++i) {
      const std::string& token = argv[i];
      if (token == std::string("-N")) N = std::atoi(argv[++i]);
      if (token == std::string("-B")) Blk = std::atoi(argv[++i]);
      if (token == std::string("-Nvec")) Nvec = std::atoi(argv[++i]);
      if (token == std::string("-S")) S = std::atoi(argv[++i]);
      if (token == std::string("-Niter")) niter = std::atoi(argv[++i]);
    }

    printf(" :::: Testing (N = %d, Blk = %d, vl = %d, vi = %d, niter = %d)\n", 
           N, Blk, vector_length, internal_vector_length, niter);


    ///
    /// problem container
    ///

    /// double 16
    Kokkos::View<vector_type***,Kokkos::LayoutRight,exec_space> Av("A",
                                                                   N/vector_length, Blk, Blk);

    /// double
    Kokkos::View<value_type****,Kokkos::LayoutRight,exec_space> As((value_type*)Av.data(),
                                                                   Av.extent(0),
                                                                   Av.extent(1),
                                                                   Av.extent(2),
                                                                   vector_length);

    /// double 2
    Kokkos::View<internal_vector_type****,Kokkos::LayoutRight,exec_space> Ai((internal_vector_type*)Av.data(),
                                                                             Av.extent(0),
                                                                             Av.extent(1),
                                                                             Av.extent(2),
                                                                             vector_length/internal_vector_length);
    /// double 16
    Kokkos::View<vector_type***,Kokkos::LayoutRight,exec_space> xv("x",
                                                                   N/vector_length, Blk, Nvec);

    /// double
    Kokkos::View<value_type****,Kokkos::LayoutRight,exec_space> xs((value_type*)xv.data(),
                                                                    xv.extent(0),
                                                                    xv.extent(1),
                                                                    xv.extent(2),
                                                                    vector_length);

    /// double 2
    Kokkos::View<internal_vector_type****,Kokkos::LayoutRight,exec_space> xi((internal_vector_type*)xv.data(),
                                                                             xv.extent(0),
                                                                             xv.extent(1),
                                                                             xv.extent(2),
                                                                             vector_length/internal_vector_length);

    /// double 16
    Kokkos::View<vector_type***,Kokkos::LayoutRight,exec_space> bv("b",
                                                                   N/vector_length, Blk, Nvec);

    /// double
    Kokkos::View<value_type****,Kokkos::LayoutRight,exec_space> bs((value_type*)bv.data(),
                                                                   bv.extent(0),
                                                                   bv.extent(1),
                                                                   bv.extent(2),
                                                                   vector_length);

    /// double 2
    Kokkos::View<internal_vector_type****,Kokkos::LayoutRight,exec_space> bi((internal_vector_type*)bv.data(),
                                                                             bv.extent(0),
                                                                             bv.extent(1),
                                                                             bv.extent(2),
                                                                             vector_length/internal_vector_length);
    

    /// double copy of A
    Kokkos::View<value_type****,Kokkos::LayoutRight,exec_space> Acopy("Acopy",
                                                                      As.extent(0),
                                                                      As.extent(1),
                                                                      As.extent(2),
                                                                      As.extent(3));

    Kokkos::View<value_type****,Kokkos::LayoutRight,exec_space> rs("rs",
                                                                   bs.extent(0), 
                                                                   bs.extent(1), 
                                                                   bs.extent(2),
                                                                   bs.extent(3));
    
#if defined(KOKKOSBATCHED_USE_128BIT_MEMORY_INST)
    auto AA = Ai;
    auto bb = bi;
    auto xx = xi;
#else
    auto AA = As;
    auto bb = bs;
    auto xx = xs;
#endif

    /// randomize input scalar view
    Kokkos::Random_XorShift64_Pool<exec_space> random(13245);
    Kokkos::fill_random(As, random, value_type(1.0));
    Kokkos::fill_random(bs, random, value_type(1.0));

    /// to verify the result, prepare for a copy of A
    Kokkos::deep_copy(Acopy, As);

    ///
    /// inverse blocks 
    ///
    if (1) {
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSBATCHED_PROFILE)
      cudaProfilerStart();
#endif
      timer.reset();
      using policy_type = Kokkos::TeamPolicy<exec_space>;
      using member_type = typename policy_type::member_type;
      int thread_team_size = 0;
      if        (Blk < 8)  { thread_team_size =  32; 
      } else if (Blk < 12) { thread_team_size =  64;
      } else               { thread_team_size = 128; }
      const int vector_loop_size = AA.extent(3), team_size = std::is_same<memory_space,typename host_space::memory_space>::value ? 1 : thread_team_size/vector_loop_size;
      using scratch_view_type = Kokkos::View<internal_vector_type***,Kokkos::LayoutRight,exec_space::scratch_memory_space,Kokkos::MemoryUnmanaged>;
      const int per_team_scratch = scratch_view_type::shmem_size(Blk, Blk, vector_loop_size); 
      policy_type policy(AA.extent(0), team_size, vector_loop_size);
      Kokkos::parallel_for
        ("inverse blocks",
         policy.set_scratch_size(0,Kokkos::PerTeam(S > per_team_scratch ? S : per_team_scratch)), 
         KOKKOS_LAMBDA(const member_type &member) {
	  typedef FactorizeModeAndAlgo<Kokkos::Impl::ActiveExecutionMemorySpace> default_mode_and_algo_type;
	  typedef default_mode_and_algo_type::mode_type mode_type; 
	  typedef default_mode_and_algo_type::algo_type algo_type;

          const int i = member.league_rank();
          Kokkos::View<internal_vector_type***,Kokkos::LayoutRight,exec_space::scratch_memory_space,Kokkos::MemoryUnmanaged>
            SS(member.team_scratch(0), Blk, Blk, vector_loop_size);
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, AA.extent(3)),[&](const int &v) {
              auto A = Kokkos::subview(AA, i, Kokkos::ALL(), Kokkos::ALL(), v);
              auto B = Kokkos::subview(SS,    Kokkos::ALL(), Kokkos::ALL(), v);
              SetIdentity<member_type,mode_type>::invoke(member, B);
              LU<member_type,mode_type,algo_type>::invoke(member, A);
              SolveLU<member_type,Trans::NoTranspose,mode_type,algo_type>::invoke(member, A, B);
              Copy<member_type,Trans::NoTranspose,mode_type>::invoke(member, B, A);
            });
        });
      Kokkos::fence();
      const double t = timer.seconds();
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSBATCHED_PROFILE)
      cudaProfilerStop();
#endif
      printf("inverse block time = %f , # of inverse block per min = %f \n", t, 1.0/t*60);
    }

    ///
    /// solve the matrix 20 times
    ///
    if (1) {
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSBATCHED_PROFILE)
      cudaProfilerStart();
#endif
      timer.reset();
      using policy_type = Kokkos::TeamPolicy<exec_space>;
      using member_type = typename policy_type::member_type;
      int thread_team_size = 0;
      if        (Blk < 8)  { thread_team_size =  32; 
      } else if (Blk < 12) { thread_team_size =  64;
      } else               { thread_team_size = 128; }
      const int vector_loop_size = AA.extent(3), team_size = std::is_same<memory_space,typename host_space::memory_space>::value ? 1 : thread_team_size/vector_loop_size;
      policy_type policy(AA.extent(0), team_size, vector_loop_size);
      for (int iter=0;iter<niter;++iter) {
        if (Nvec == 1) {
          Kokkos::parallel_for
            ("solve for single right hand side",
             policy.set_scratch_size(0,Kokkos::PerTeam(S)), KOKKOS_LAMBDA(const member_type &member) {
              typedef SolveSingleModeAndAlgo<Kokkos::Impl::ActiveExecutionMemorySpace> default_mode_and_algo_type;
              typedef default_mode_and_algo_type::mode_type mode_type; 
              typedef default_mode_and_algo_type::algo_type algo_type;
              
              const int i = member.league_rank();
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, AA.extent(5)),[&](const int &v) {
                  auto Ainv = Kokkos::subview(AA, i, Kokkos::ALL(), Kokkos::ALL(), v);
                  auto x    = Kokkos::subview(xx, i, Kokkos::ALL(),             0, v);
                  auto b    = Kokkos::subview(bb, i, Kokkos::ALL(),             0, v);
                  Gemv<member_type,
                       Trans::NoTranspose,mode_type,algo_type>
                    ::invoke(member, 1.0, Ainv, b, 0.0, x);
                });
            });
        } else {
          Kokkos::parallel_for
            ("solve for multiple right hand side",
             policy.set_scratch_size(0,Kokkos::PerTeam(S)), KOKKOS_LAMBDA(const member_type &member) {
              typedef SolveMultipleModeAndAlgo<Kokkos::Impl::ActiveExecutionMemorySpace> default_mode_and_algo_type;
              typedef default_mode_and_algo_type::mode_type mode_type; 
              typedef default_mode_and_algo_type::algo_type algo_type;
              
              const int i = member.league_rank();
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, AA.extent(5)),[&](const int &v) {
                  auto Ainv = Kokkos::subview(AA, i, Kokkos::ALL(), Kokkos::ALL(), v);
                  auto X    = Kokkos::subview(xx, i, Kokkos::ALL(), Kokkos::ALL(), v);
                  auto B    = Kokkos::subview(bb, i, Kokkos::ALL(), Kokkos::ALL(), v);
                  Gemm<member_type,
                       Trans::NoTranspose,Trans::NoTranspose,mode_type,algo_type>
                    ::invoke(member, 1.0, Ainv, B, 0.0, X);
                });
            });
        }

        Kokkos::fence();
      }
      const double t = timer.seconds();
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSBATCHED_PROFILE)
      cudaProfilerStop();
#endif
      printf("solve time = %f , # of solves per min = %f\n", t, 1.0/t*60*niter);
    }
    
    ///
    /// compute residual
    ///
    if (1) {
      typedef KokkosBatched::Algo::Level2::Unblocked algo_type;
      using policy_type = Kokkos::TeamPolicy<exec_space>;
      using member_type = typename policy_type::member_type;
      policy_type policy(Acopy.extent(0), Kokkos::AUTO(), Acopy.extent(5));
      Kokkos::parallel_for
        ("compute residual",
         policy, KOKKOS_LAMBDA(const member_type &member) {
          const int i = member.league_rank();
          
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, Acopy.extent(5)),[&](const int &v) {
              auto A = Kokkos::subview(Acopy, i, Kokkos::ALL(), Kokkos::ALL(), v);
              for (int jvec=0;jvec<Nvec;++jvec) {
                auto x = Kokkos::subview(xs, i, Kokkos::ALL(), jvec, v);
                auto b = Kokkos::subview(bs, i, Kokkos::ALL(), jvec, v);
                auto r = Kokkos::subview(rs, i, Kokkos::ALL(), jvec, v);
                TeamCopy<member_type,
                         Trans::NoTranspose>
                  ::invoke(member, b, r);
                TeamGemv<member_type,
                         Trans::NoTranspose,algo_type>
                  ::invoke(member, -1.0, A, x, 1.0, r);
              }
            });
        });
      Kokkos::fence();
      auto rs_host = Kokkos::create_mirror_view(rs);
      auto bs_host = Kokkos::create_mirror_view(bs);
      Kokkos::deep_copy(rs_host, rs);
      Kokkos::deep_copy(bs_host, bs);
      Kokkos::fence();
      {
        double norm2 = 0, diff2 = 0;
        for (int i0=0,i0end=rs.extent(0);i0<i0end;++i0) // N/vector_length
          for (int i1=0,i1end=rs.extent(1);i1<i1end;++i1) // Blk
            for (int i2=0,i2end=rs.extent(2);i2<i2end;++i2) // Nvec
              for (int i3=0,i3end=rs.extent(3);i3<i3end;++i3) {// vector_length
                const auto val = bs_host(i0,i1,i2,i3);
                const auto res = rs_host(i0,i1,i2,i3);
                norm2 += val*val;
                diff2 += res*res;
              }
        printf("rel error = %e\n", diff2/norm2);
      }
    }
  }
  Kokkos::finalize();

  return 0;
}

#else
int main() {
  return 0;
}
#endif

