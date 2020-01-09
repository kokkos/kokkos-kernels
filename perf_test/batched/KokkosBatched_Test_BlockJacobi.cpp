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
struct MatrixModeAndAlgo;

template<>
struct MatrixModeAndAlgo<Kokkos::HostSpace> {
  typedef Mode::Serial mode_type;
  typedef Algo::Level3::Blocked algo_type;   
};

#if defined(KOKKOS_ENABLE_CUDA)
template<>
struct MatrixModeAndAlgo<Kokkos::CudaSpace> {
  typedef Mode::Team mode_type;
  typedef Algo::Level3::Unblocked algo_type;   
};
#endif

template<typename ActiveMemorySpace>
struct VectorModeAndAlgo;

template<>
struct VectorModeAndAlgo<Kokkos::HostSpace> {
  typedef Mode::Serial mode_type;
  typedef Algo::Level2::Blocked algo_type;   
};

#if defined(KOKKOS_ENABLE_CUDA)
template<>
struct VectorModeAndAlgo<Kokkos::CudaSpace> {
  typedef Mode::Team mode_type;
  typedef Algo::Level2::Unblocked algo_type;   
};
#endif

template<typename ActiveMemorySpace>
using CopyModeAndAlgo = MatrixModeAndAlgo<ActiveMemorySpace>;

template<typename ActiveMemorySpace>
using FactorizeModeAndAlgo = MatrixModeAndAlgo<ActiveMemorySpace>;

template<typename ActiveMemorySpace>
using SolveMultipleModeAndAlgo = MatrixModeAndAlgo<ActiveMemorySpace>;

template<typename ActiveMemorySpace>
using SolveSingleModeAndAlgo = VectorModeAndAlgo<ActiveMemorySpace>;

/// 
/// This is an example of block diagonal jacobi solver. This code uses non-trivial data
/// compact data format to increase efficiency of vectorization. Thus, it requires packing
/// if uses want to use traditional matrix format i.e., array of blocks for block jacobi 
/// preconditioner and multi-vector for right hand side. As this code is for demonstration
/// of the block jacobi solver, we do not fuse packing with other solver components. However,
/// a realistic application would fuse the packing code with other parts of the solver e.g., 
/// residual computing (sparse matvec) and compute inverse block jacobi to amortize the cost 
/// of packing.
///
/// For a demonstration purpose, this code has three phases:
/// 1) user data preparation and packing into vector-friendly format,
/// 2) construct inverse jacobi,
/// 3) apply the preconditioner in an interative sense.
/// At the end, the code perform validation by computing its residual.
///

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
    {
      /// input check
      if (N%vector_length) {
        printf("Error: given N(%d) is not a multiplication of the pack size (%d)\n", N, vector_length);
        printf("  In general, this is not a requirement but this code is for demonstration and it requires N to be multiplication of the pack size.\n");
        return -1;
      }
    }
    printf(" :::: Testing (N = %d, Blk = %d, vl = %d, vi = %d, niter = %d)\n", 
           N, Blk, vector_length, internal_vector_length, niter);

    ///
    /// expected traditional interface from users
    ///
    Kokkos::View<value_type***,exec_space> A_given("A given block matrices",
                                                   N, Blk, Blk);
    Kokkos::View<value_type**,Kokkos::LayoutLeft,exec_space> b_given("B given block right hand side multi vector",
                                                                     N*Blk, Nvec);
    Kokkos::View<value_type**,Kokkos::LayoutLeft,exec_space> x_solution("X solution in the standard multi vector",
                                                                        N*Blk, Nvec);
    Kokkos::View<value_type**,Kokkos::LayoutLeft,exec_space> r_residual("R residual for validation",
                                                                        N*Blk, Nvec);
    
    /// randomize input scalar view
    Kokkos::Random_XorShift64_Pool<exec_space> random(13245);
    Kokkos::fill_random(A_given, random, value_type(1.0));
    Kokkos::fill_random(b_given, random, value_type(1.0));

    ///
    /// packed container
    ///

    /// double 16 vector packed
    Kokkos::View<vector_type***,Kokkos::LayoutRight,exec_space> Av("A",
                                                                   N/vector_length, Blk, Blk);
    
    /// double 2 internal vector interpretation for GPUs
    /// for CPUs, the internal_vector_type is double
    Kokkos::View<internal_vector_type****,Kokkos::LayoutRight,exec_space> Ai((internal_vector_type*)Av.data(),
                                                                             Av.extent(0),
                                                                             Av.extent(1),
                                                                             Av.extent(2),
                                                                             vector_length/internal_vector_length);
    
    /// double
    Kokkos::View<value_type****,Kokkos::LayoutRight,exec_space> As((value_type*)Av.data(),
                                                                   Av.extent(0),
                                                                   Av.extent(1),
                                                                   Av.extent(2),
                                                                   vector_length);    

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
    
#if defined(KOKKOSBATCHED_USE_128BIT_MEMORY_INST)
    auto AA = Ai;
    auto bb = bi;
    auto xx = xi;
#else
    auto AA = As;
    auto bb = bs;
    auto xx = xs;
#endif

    /// kokkos parallel policy 
    using policy_type = Kokkos::TeamPolicy<exec_space>;
    using member_type = typename policy_type::member_type;
    int thread_team_size = 0;

    /// packing
    if (1) {
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSBATCHED_PROFILE)
      cudaProfilerStart();
#endif
      timer.reset();
      if        (Blk < 8)  { thread_team_size =  32; 
      } else if (Blk < 12) { thread_team_size =  64;
      } else               { thread_team_size = 128; }
      const int team_size = std::is_same<memory_space,typename host_space::memory_space>::value ? 1 : thread_team_size/vector_length;
      const int vector_size = std::is_same<memory_space,typename host_space::memory_space>::value ? 1 : vector_length;
      policy_type policy(As.extent(0), team_size, vector_size);
      Kokkos::parallel_for
        ("copy from the user blocks",
         policy,
         KOKKOS_LAMBDA(const member_type &member) {
	  typedef CopyModeAndAlgo<Kokkos::Impl::ActiveExecutionMemorySpace> default_mode_and_algo_type;
	  typedef default_mode_and_algo_type::mode_type mode_type; 

          const int i = member.league_rank();
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, vector_length),[&](const int &v) {
              const int offset = i*vector_length+v;

              auto A     = Kokkos::subview(A_given, offset,     Kokkos::ALL(), Kokkos::ALL());
              auto b     = Kokkos::subview(b_given, Kokkos::pair<int,int>(offset*Blk,(offset+1)*Blk), Kokkos::ALL());

              auto Apack = Kokkos::subview(As, i, Kokkos::ALL(), Kokkos::ALL(), v);
              auto bpack = Kokkos::subview(bs, i, Kokkos::ALL(), Kokkos::ALL(), v);

              Copy<member_type,Trans::NoTranspose,mode_type>::invoke(member, A, Apack);
              Copy<member_type,Trans::NoTranspose,mode_type>::invoke(member, b, bpack);
            });
        });
      Kokkos::fence();
      const double t = timer.seconds();
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSBATCHED_PROFILE)
      cudaProfilerStop();
#endif
      printf("packing time = %f\n", t);
    }

    ///
    /// inverse blocks 
    ///
    if (1) {
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSBATCHED_PROFILE)
      cudaProfilerStart();
#endif
      timer.reset();
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
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, vector_loop_size),[&](const int &v) {
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
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, vector_loop_size),[&](const int &v) {
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
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, vector_loop_size),[&](const int &v) {
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
      printf("solve time = %f , # of solves per min = %f\n", t/double(niter), 1.0/t*60*niter);
    }
    
    ///
    /// unpacking and compute residual
    ///
    if (1) {
      typedef KokkosBatched::Algo::Level2::Unblocked algo_type;
      policy_type policy(As.extent(0), Kokkos::AUTO());
      Kokkos::parallel_for
        ("compute residual",
         policy, KOKKOS_LAMBDA(const member_type &member) {
          const int i = member.league_rank();
          
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, vector_length),[&](const int &v) {
              const int ii = i*vector_length + v;
              auto A = Kokkos::subview(A_given, ii, Kokkos::ALL(), Kokkos::ALL());
              for (int jvec=0;jvec<Nvec;++jvec) {
                const int offset = ii*Blk;
                const auto block_vector_range = Kokkos::pair<int,int>(offset,offset+Blk);

                auto xp  = Kokkos::subview(xs, i, Kokkos::ALL(), jvec, v);
                auto x = Kokkos::subview(x_solution, block_vector_range, jvec);
                auto b = Kokkos::subview(b_given,    block_vector_range, jvec);
                auto r = Kokkos::subview(r_residual, block_vector_range, jvec);
                /// solution unpack
                TeamCopy<member_type,Trans::NoTranspose>          ::invoke(member, xp, x);
                /// residual
                TeamCopy<member_type,Trans::NoTranspose>          ::invoke(member, b, r);
                TeamGemv<member_type,Trans::NoTranspose,algo_type>::invoke(member, -1.0, A, x, 1.0, r);
              }
            });
        });
      Kokkos::fence();
      auto r_host = Kokkos::create_mirror_view(r_residual);
      auto b_host = Kokkos::create_mirror_view(b_given);
      Kokkos::deep_copy(r_host, r_residual);
      Kokkos::deep_copy(b_host, b_given);
      Kokkos::fence();
      {
        double norm2 = 0, diff2 = 0;
        for (int i0=0,i0end=r_host.extent(0);i0<i0end;++i0) // N*Blk
          for (int i1=0,i1end=r_host.extent(1);i1<i1end;++i1) { // Nvec
            const auto val = b_host(i0,i1);
            const auto res = r_host(i0,i1);
            norm2 += val*val;
            diff2 += res*res;
            ///printf("val %e, res %e\n", val, res);
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

