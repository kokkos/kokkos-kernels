/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include <iomanip>

#if defined(__KOKKOSKERNELS_NVIDIA_CUBLAS__)
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "cublas_api.h"
#endif

#include "Kokkos_Core.hpp"
#include "impl/Kokkos_Timer.hpp"

#include "KokkosKernels_Vector.hpp"

#include "KokkosKernels_Gemm_Decl.hpp"
#include "KokkosKernels_Gemm_Serial_Impl.hpp"
#include "KokkosKernels_Gemm_Team_Impl.hpp"

namespace KokkosKernels {
  namespace Batched {
    namespace Experimental {
      namespace PerfTest {

#undef FLOP_MUL
#undef FLOP_ADD
#define FLOP_MUL 1.0
#define FLOP_ADD 1.0

        double FlopCount(int mm, int nn, int kk) {
          double m = (double)mm;    double n = (double)nn;    double k = (double)kk;
          return (FLOP_MUL*(m*n*k) +
                  FLOP_ADD*(m*n*k));
        }

        template<int BlkSize, int VectorLength, typename ValueType, typename DeviceSpaceType, typename AlgoTagType>
        void Gemm(const int N) {
          typedef Kokkos::Schedule<Kokkos::Static> ScheduleType;

          const double flop = (N*VectorLength)*FlopCount(BlkSize,BlkSize,BlkSize);
          const double tmax = 1.0e15;

          typedef Kokkos::DefaultHostExecutionSpace HostSpaceType;

          const int iter_begin = -3, iter_end = 10;
          Kokkos::Impl::Timer timer;

          Kokkos::View<ValueType***,Kokkos::LayoutLeft,HostSpaceType>
            amat("amat", N*VectorLength, BlkSize, BlkSize),
            bmat("bmat", N*VectorLength, BlkSize, BlkSize),
            cref("cref", N*VectorLength, BlkSize, BlkSize);

          {
            Random<ValueType> random;
            for (int k=0;k<N*VectorLength;++k)
              for (int i=0;i<BlkSize;++i)
                for (int j=0;j<BlkSize;++j) {
                  amat(k, i, j) = random.value();
                  bmat(k, i, j) = random.value();
                }
          }

          // P100 L2 cache 4MB per core
          constexpr size_t LLC_CAPACITY = 56*4*1024*1024;
          Flush<LLC_CAPACITY,DeviceSpaceType> flush;

#if defined(__KOKKOSKERNELS_NVIDIA_CUBLAS__)
          {
            ///
            /// CUBLAS Strided version
            ///
            const Kokkos::LayoutStride stride(N*VectorLength, BlkSize*BlkSize,
                                              BlkSize, 1,
                                              BlkSize, BlkSize);

            Kokkos::View<ValueType***,Kokkos::LayoutStride,DeviceSpaceType>
              a("a", stride),
              b("b", stride),
              c("c", stride);

            double tavg = 0, tmin = tmax;

            cublasStatus_t stat;
            cublasHandle_t handle;

            stat = cublasCreate(&handle);
            if (stat != CUBLAS_STATUS_SUCCESS)
              Kokkos::abort("CUBLAS initialization failed\n");

            auto amat_device = Kokkos::create_mirror_view(typename DeviceSpaceType::memory_space(), amat);
            auto bmat_device = Kokkos::create_mirror_view(typename DeviceSpaceType::memory_space(), bmat);

            Kokkos::deep_copy(amat_device, amat);
            Kokkos::deep_copy(bmat_device, bmat);

            DeviceSpaceType::fence();

            const double one(1.0), zero(0.0);
            {
              tavg = 0; tmin = tmax;

              for (int iter=iter_begin;iter<iter_end;++iter) {
                // flush
                flush.run();

                // initialize matrices
                Kokkos::deep_copy(a, amat_device);
                Kokkos::deep_copy(b, bmat_device);
                Kokkos::deep_copy(c, 0);

                DeviceSpaceType::fence();
                timer.reset();

                stat = cublasDgemmStridedBatched(handle,
                                                 CUBLAS_OP_N,
                                                 CUBLAS_OP_N,
                                                 BlkSize, BlkSize, BlkSize,
                                                 &one,
                                                 (const ValueType*)a.data(), BlkSize, BlkSize*BlkSize,
                                                 (const ValueType*)b.data(), BlkSize, BlkSize*BlkSize,
                                                 &zero,
                                                 (ValueType*)c.data(), BlkSize, BlkSize*BlkSize,
                                                 N*VectorLength);

                DeviceSpaceType::fence();
                const double t = timer.seconds();
                tmin = std::min(tmin, t);
                tavg += (iter >= 0)*t;
              }
              tavg /= iter_end;

              auto csol = Kokkos::create_mirror_view(typename HostSpaceType::memory_space(), c);
              Kokkos::deep_copy(csol, c);
              Kokkos::deep_copy(cref, csol);

              std::cout << std::setw(8) << "CUBLAS"
                        << std::setw(8) << "Strided"
                        << " BlkSize = " << std::setw(3) << BlkSize
                        << " TeamSize = N/A" 
                        << " time = " << std::scientific << tmin
                        << " avg flop/s = " << (flop/tavg)
                        << " max flop/s = " << (flop/tmin)
                        << std::endl;
            }
            cublasDestroy(handle);
          }
#endif

          {
            ///
            /// Range policy version
            ///
            Kokkos::View<ValueType***,DeviceSpaceType>
              a("a", N*VectorLength, BlkSize, BlkSize),
              b("b", N*VectorLength, BlkSize, BlkSize),
              c("c", N*VectorLength, BlkSize, BlkSize);

            double tavg = 0, tmin = tmax;
            {
              const Kokkos::RangePolicy<DeviceSpaceType,ScheduleType> policy(0, N*VectorLength);

              for (int iter=iter_begin;iter<iter_end;++iter) {
                // flush
                flush.run();

                // initialize matrices
                Kokkos::deep_copy(a, amat);
                Kokkos::deep_copy(b, bmat);
                Kokkos::deep_copy(c, 0);

                DeviceSpaceType::fence();
                timer.reset();

                Kokkos::parallel_for
                  (policy,
                   KOKKOS_LAMBDA(const int k) {
                    auto aa = Kokkos::subview(a, k, Kokkos::ALL(), Kokkos::ALL());
                    auto bb = Kokkos::subview(b, k, Kokkos::ALL(), Kokkos::ALL());
                    auto cc = Kokkos::subview(c, k, Kokkos::ALL(), Kokkos::ALL());
                    
                    Serial::
                      Gemm<Trans::NoTranspose,Trans::NoTranspose,AlgoTagType>::
                      invoke(1.0, aa, bb, 1.0, cc);
                  });

                DeviceSpaceType::fence();
                const double t = timer.seconds();
                tmin = std::min(tmin, t);
                tavg += (iter >= 0)*t;
              }
              tavg /= iter_end;

              auto csol = Kokkos::create_mirror_view(typename HostSpaceType::memory_space(), c);
              Kokkos::deep_copy(csol, c);

              double diff = 0;
              for (int i=0;i<cref.dimension(0);++i)
                for (int j=0;j<cref.dimension(1);++j)
                  for (int k=0;k<cref.dimension(2);++k)
                    diff += std::abs(cref(i,j,k) - csol(i,j,k));

              std::cout << std::setw(8) << "Kokkos"
                        << std::setw(8) << "Range"
                        << " BlkSize = " << std::setw(3) << BlkSize
                        << " TeamSize = N/A" 
                        << " time = " << std::scientific << tmin
                        << " avg flop/s = " << (flop/tavg)
                        << " max flop/s = " << (flop/tmin)
                        << " diff to ref = " << diff
                        << std::endl;
            }
          }

          {
            ///
            /// Team policy version (almost same scheduling with range policy)
            ///
            Kokkos::View<ValueType***,DeviceSpaceType>
              a("a", N*VectorLength, BlkSize, BlkSize),
              b("b", N*VectorLength, BlkSize, BlkSize),
              c("c", N*VectorLength, BlkSize, BlkSize);

            double tavg = 0, tmin = tmax;
            {
              typedef Kokkos::TeamPolicy<DeviceSpaceType,ScheduleType> policy_type;
              typedef typename policy_type::member_type member_type;

              int team_size = 128, thres = 1024;
              if (std::is_same<AlgoTagType,Algo::Gemm::Blocked>::value) thres = 256;

              while (team_size*VectorLength>=thres) team_size /= 2;

              const policy_type policy(N/team_size, team_size, VectorLength);
              for (int iter=iter_begin;iter<iter_end;++iter) {
                // flush
                flush.run();

                // initialize matrices
                Kokkos::deep_copy(a, amat);
                Kokkos::deep_copy(b, bmat);
                Kokkos::deep_copy(c, 0);

                DeviceSpaceType::fence();
                timer.reset();

                Kokkos::parallel_for
                  (policy,
                   KOKKOS_LAMBDA(const member_type &member) {
                    const int kbeg = (member.league_rank()*(member.team_size()*VectorLength) +
                                      member.team_rank()*VectorLength);
                    Kokkos::parallel_for
                      (Kokkos::ThreadVectorRange(member, VectorLength),
                       [&](const int &k) {
                        const int kk = kbeg + k;
                        if (kk < N*VectorLength) {
                          auto aa = Kokkos::subview(a, kk, Kokkos::ALL(), Kokkos::ALL());
                          auto bb = Kokkos::subview(b, kk, Kokkos::ALL(), Kokkos::ALL());
                          auto cc = Kokkos::subview(c, kk, Kokkos::ALL(), Kokkos::ALL());
                          
                          Serial::
                            Gemm<Trans::NoTranspose,Trans::NoTranspose,AlgoTagType>::
                            invoke(1.0, aa, bb, 1.0, cc);
                        }
                      });
                  });
                
                DeviceSpaceType::fence();
                const double t = timer.seconds();
                tmin = std::min(tmin, t);
                tavg += (iter >= 0)*t;
              }
              tavg /= iter_end;

              auto csol = Kokkos::create_mirror_view(typename HostSpaceType::memory_space(), c);
              Kokkos::deep_copy(csol, c);

              double diff = 0;
              for (int i=0;i<cref.dimension(0);++i)
                for (int j=0;j<cref.dimension(1);++j)
                  for (int k=0;k<cref.dimension(2);++k)
                    diff += std::abs(cref(i,j,k) - csol(i,j,k));

              std::cout << std::setw(8) << "Kokkos"
                        << std::setw(8) << "Team v1"
                        << " BlkSize = " << std::setw(3) << BlkSize
                        << " TeamSize = " << std::setw(3) << team_size 
                        << " time = " << std::scientific << tmin
                        << " avg flop/s = " << (flop/tavg)
                        << " max flop/s = " << (flop/tmin)
                        << " diff to ref = " << diff
                        << std::endl;
            }
          }

          {
            ///
            /// Team policy version (team parallel)
            ///
            Kokkos::View<ValueType***,DeviceSpaceType>
              a("a", N*VectorLength, BlkSize, BlkSize),
              b("b", N*VectorLength, BlkSize, BlkSize),
              c("c", N*VectorLength, BlkSize, BlkSize);

            double tavg = 0, tmin = tmax;
            {
              typedef Kokkos::TeamPolicy<DeviceSpaceType,ScheduleType> policy_type;
              typedef typename policy_type::member_type member_type;

              int team_size = BlkSize*BlkSize, thres = 1024;
              if (std::is_same<AlgoTagType,Algo::Gemm::Blocked>::value) thres = 512;

              while (team_size*VectorLength>=thres) --team_size;

              policy_type policy(N, team_size, VectorLength);
              for (int iter=iter_begin;iter<iter_end;++iter) {
                // flush
                flush.run();

                // initialize matrices
                Kokkos::deep_copy(a, amat);
                Kokkos::deep_copy(b, bmat);
                Kokkos::deep_copy(c, 0);

                DeviceSpaceType::fence();
                timer.reset();

                Kokkos::parallel_for
                  (policy,
                   KOKKOS_LAMBDA(const member_type &member) {
                    const int kbeg = member.league_rank()*VectorLength;
                    Kokkos::parallel_for
                      (Kokkos::ThreadVectorRange(member, VectorLength),
                       [&](const int &k) {
                        const int kk = kbeg + k;
                        if (kk < N*VectorLength) {
                          auto aa = Kokkos::subview(a, kk, Kokkos::ALL(), Kokkos::ALL());
                          auto bb = Kokkos::subview(b, kk, Kokkos::ALL(), Kokkos::ALL());
                          auto cc = Kokkos::subview(c, kk, Kokkos::ALL(), Kokkos::ALL());
                          
                          Team::
                            Gemm<member_type,Trans::NoTranspose,Trans::NoTranspose,AlgoTagType>::
                            invoke(member, 1.0, aa, bb, 1.0, cc);
                        }
                      });
                  });
                
                DeviceSpaceType::fence();
                const double t = timer.seconds();
                tmin = std::min(tmin, t);
                tavg += (iter >= 0)*t;
              }
              tavg /= iter_end;

              auto csol = Kokkos::create_mirror_view(typename HostSpaceType::memory_space(), c);
              Kokkos::deep_copy(csol, c);

              double diff = 0;
              for (int i=0;i<cref.dimension(0);++i)
                for (int j=0;j<cref.dimension(1);++j)
                  for (int k=0;k<cref.dimension(2);++k)
                    diff += std::abs(cref(i,j,k) - csol(i,j,k));

              std::cout << std::setw(8) << "Kokkos"
                        << std::setw(8) << "Team v2"
                        << " BlkSize = " << std::setw(3) << BlkSize
                        << " TeamSize = " << std::setw(3) << team_size
                        << " time = " << std::scientific << tmin
                        << " avg flop/s = " << (flop/tavg)
                        << " max flop/s = " << (flop/tmin)
                        << " diff to ref = " << diff
                        << std::endl;
            }
          }

          {
            ///
            /// Team policy version (handmade)
            ///
            Kokkos::View<ValueType***,DeviceSpaceType>
              a("a", N*VectorLength, BlkSize, BlkSize),
              b("b", N*VectorLength, BlkSize, BlkSize),
              c("c", N*VectorLength, BlkSize, BlkSize);

            double tavg = 0, tmin = tmax;
            {
              typedef typename DeviceSpaceType::scratch_memory_space scratch_space;
              typedef Kokkos::View<ValueType***,Kokkos::LayoutLeft,scratch_space> scratch_view_type;

              typedef Kokkos::TeamPolicy<DeviceSpaceType,ScheduleType> policy_type;
              typedef typename policy_type::member_type member_type;

              int team_size = BlkSize*BlkSize, thres = 1024;
              while (team_size*VectorLength>=thres) --team_size;
              const policy_type policy(N, team_size, VectorLength);
              
              // const int 
              //   shlvl = 0, 
              //   per_team_scratch = 2*scratch_view_type::shmem_size(VectorLength, BlkSize, BlkSize);

              //if (per_team_scratch/1024 < 48)
              for (int iter=iter_begin;iter<iter_end;++iter) {
                // flush
                flush.run();

                // initialize matrices
                Kokkos::deep_copy(a, amat);
                Kokkos::deep_copy(b, bmat);
                Kokkos::deep_copy(c, 0);

                DeviceSpaceType::fence();
                timer.reset();

                Kokkos::parallel_for
                  //(policy.set_scratch_size(shlvl, Kokkos::PerTeam(per_team_scratch)),
                  (policy,
                   KOKKOS_LAMBDA(const member_type &member) {
                    const int kbeg = member.league_rank()*VectorLength;
                    Kokkos::parallel_for
                      (Kokkos::ThreadVectorRange(member, VectorLength),
                       [&](const int &k) {
                        const int kk = kbeg + k;
                        if (kk < N*VectorLength) {
                          // scratch_view_type sa(member.team_scratch(shlvl), VectorLength, BlkSize, BlkSize);
                          // scratch_view_type sb(member.team_scratch(shlvl), VectorLength, BlkSize, BlkSize);
                          // Kokkos::parallel_for
                          //   (Kokkos::TeamThreadRange(member,0,BlkSize*BlkSize),
                          //    [&](const int &ij) {
                          //     const int i = ij/BlkSize, j = ij%BlkSize;
                          //     sa(k, i, j) = a(kk, i, j);
                          //     sb(k, i, j) = b(kk, i, j);
                          //   });
                          // member.team_barrier();
                          // Kokkos::parallel_for
                          //   (Kokkos::TeamThreadRange(member,0,BlkSize*BlkSize),
                          //    [&](const int &ij) {
                          //     const int i = ij/BlkSize, j = ij%BlkSize;                              
                          //     ValueType cval = 0;
                          //     for (int p=0;p<BlkSize;++p)
                          //       cval += sa(k, i, p)*sb(k, p, j);
                          //     c(kk, i, j) += cval;
                          //   });
                          Kokkos::parallel_for
                            (Kokkos::TeamThreadRange(member,0,BlkSize*BlkSize),
                             [&](const int &ij) {
                              const int i = ij/BlkSize, j = ij%BlkSize;                              
                              ValueType cval = 0;
                              for (int p=0;p<BlkSize;++p)
                                cval += a(kk, i, p)*b(kk, p, j);
                              c(kk, i, j) += cval;
                            });
                        }
                      });
                  });
                
                DeviceSpaceType::fence();
                const double t = timer.seconds();
                tmin = std::min(tmin, t);
                tavg += (iter >= 0)*t;
              }
              tavg /= iter_end;

              auto csol = Kokkos::create_mirror_view(typename HostSpaceType::memory_space(), c);
              Kokkos::deep_copy(csol, c);

              double diff = 0;
              for (int i=0;i<cref.dimension(0);++i)
                for (int j=0;j<cref.dimension(1);++j)
                  for (int k=0;k<cref.dimension(2);++k)
                    diff += std::abs(cref(i,j,k) - csol(i,j,k));

              std::cout << std::setw(8) << "Kokkos"
                        << std::setw(8) << "Team v4"
                        << " BlkSize = " << std::setw(3) << BlkSize
                        << " TeamSize = " << std::setw(3) << team_size
                        << " time = " << std::scientific << tmin
                        << " avg flop/s = " << (flop/tavg)
                        << " max flop/s = " << (flop/tmin)
                        << " diff to ref = " << diff
                        << std::endl;
            }
          }
          // {
          //   ///
          //   /// Team policy version (team parallel + scratch)
          //   ///
          //   Kokkos::View<ValueType***,DeviceSpaceType>
          //     a("a", N*VectorLength, BlkSize, BlkSize),
          //     b("b", N*VectorLength, BlkSize, BlkSize),
          //     c("c", N*VectorLength, BlkSize, BlkSize);

          //   double tavg = 0, tmin = tmax;
          //   {
          //     typedef typename DeviceSpaceType::scratch_memory_space scratch_space;
          //     typedef Kokkos::View<ValueType***,Kokkos::LayoutLeft,scratch_space> scratch_view_type;

          //     typedef Kokkos::TeamPolicy<DeviceSpaceType,ScheduleType> policy_type;
          //     typedef typename policy_type::member_type member_type;

          //     int team_size = BlkSize*BlkSize, thres = 1024;
          //     if (std::is_same<AlgoTagType,Algo::Gemm::Blocked>::value) thres = 512;

          //     while (team_size*VectorLength>=thres) --team_size;

          //     const int shlvl = 0; // shared memory level
          //     const size_t shsize = 3*scratch_view_type::shmem_size(vl,blk,blk))),
          //     const policy_type policy(N, team_size, VectorLength);
              
          //     if (shsize/1024 < 48) 
          //     for (int iter=iter_begin;iter<iter_end;++iter) {
          //       // flush
          //       flush.run();

          //       // initialize matrices
          //       Kokkos::deep_copy(a, amat);
          //       Kokkos::deep_copy(b, bmat);
          //       Kokkos::deep_copy(c, 0);

          //       DeviceSpaceType::fence();
          //       timer.reset();

          //       Kokkos::parallel_for
          //         (policy.set_scratch_size(shlvl, Kokkos::PerTeam(shsize)),
          //          KOKKOS_LAMBDA(const member_type &member) {
          //           scratch_view_type sa(member.team_scratch(shlvl), VectorLength, blk, blk);
          //           scratch_view_type sb(member.team_scratch(shlvl), VectorLength, blk, blk);

          //           const int kbeg = member.league_rank()*VectorLength;
          //           Kokkos::parallel_for
          //             (Kokkos::ThreadVectorRange(member, VectorLength),
          //              [&](const int &k) {
          //               const int kk = kbeg + k;
          //               if (kk < N*VectorLength) {
          //                 auto aa = Kokkos::subview(a, kk, Kokkos::ALL(), Kokkos::ALL());
          //                 auto bb = Kokkos::subview(b, kk, Kokkos::ALL(), Kokkos::ALL());
          //                 auto cc = Kokkos::subview(c, kk, Kokkos::ALL(), Kokkos::ALL());

          //                 auto saa = Kokkos::subview(sa, k, Kokkos::ALL(), Kokkos::ALL());
          //                 auto sbb = Kokkos::subview(sb, k, Kokkos::ALL(), Kokkos::ALL());
          //                 auto scc = Kokkos::subview(sc, k, Kokkos::ALL(), Kokkos::ALL());

          //                 Team::Copy<member_type,Trans::NoTranspose>::invoke(member, saa, aa);
          //                 Team::Copy<member_type,Trans::NoTranspose>::invoke(member, sbb, bb);
          //                 Team::Copy<member_type,Trans::NoTranspose>::invoke(member, scc, cc);
          //                 member.barrier();
          //                 Team::
          //                   Gemm<member_type,Trans::NoTranspose,Trans::NoTranspose,AlgoTagType>::
          //                   invoke(member, 1.0, saa, sbb, 1.0, scc);
          //                 member.barrier();
          //                 Team::Copy<member_type,Trans::NoTranspose>::invoke(member, cc, scc);
          //               }
          //             });
          //         });
                
          //       DeviceSpaceType::fence();
          //       const double t = timer.seconds();
          //       tmin = std::min(tmin, t);
          //       tavg += (iter >= 0)*t;
          //     }
          //     tavg /= iter_end;

          //     auto csol = Kokkos::create_mirror_view(typename HostSpaceType::memory_space(), c);
          //     Kokkos::deep_copy(csol, c);

          //     double diff = 0;
          //     for (int i=0;i<cref.dimension(0);++i)
          //       for (int j=0;j<cref.dimension(1);++j)
          //         for (int k=0;k<cref.dimension(2);++k)
          //           diff += std::abs(cref(i,j,k) - csol(i,j,k));

          //     std::cout << std::setw(8) << "Kokkos"
          //               << std::setw(8) << "Team v3"
          //               << " BlkSize = " << std::setw(3) << BlkSize
          //               << " TeamSize = " << std::setw(3) << team_size
          //               << " time = " << std::scientific << tmin
          //               << " avg flop/s = " << (flop/tavg)
          //               << " max flop/s = " << (flop/tmin)
          //               << " diff to ref = " << diff
          //               << std::endl;
          //   }
          // }

          std::cout << std::endl;
        }
      }
    }
  }
}

using namespace KokkosKernels::Batched::Experimental;

template<int VectorLength, 
         typename ValueType,
         typename AlgoTagType>
void run(const int N) {
  typedef Kokkos::DefaultExecutionSpace ExecSpace;

  std::cout << "ExecSpace::  ";
  if (std::is_same<ExecSpace,Kokkos::Serial>::value)
    std::cout << "Kokkos::Serial " << std::endl;
  else
    ExecSpace::print_configuration(std::cout, false);

  PerfTest::Gemm< 4, VectorLength, ValueType, ExecSpace, AlgoTagType>(N);
  PerfTest::Gemm< 8, VectorLength, ValueType, ExecSpace, AlgoTagType>(N);
  PerfTest::Gemm<16, VectorLength, ValueType, ExecSpace, AlgoTagType>(N);
  PerfTest::Gemm<20, VectorLength, ValueType, ExecSpace, AlgoTagType>(N);
  PerfTest::Gemm<32, VectorLength, ValueType, ExecSpace, AlgoTagType>(N);
  //PerfTest::Gemm<64, VectorLength, ValueType, ExecSpace, AlgoTagType>(N);

  // PerfTest::Gemm< 3, VectorLength, ValueType, ExecSpace, AlgoTagType>(N);
  // PerfTest::Gemm< 5, VectorLength, ValueType, ExecSpace, AlgoTagType>(N);
  // PerfTest::Gemm<10, VectorLength, ValueType, ExecSpace, AlgoTagType>(N);
  // PerfTest::Gemm<15, VectorLength, ValueType, ExecSpace, AlgoTagType>(N);
}

int main(int argc, char *argv[]) {

  Kokkos::initialize(argc, argv);

  const int ntest = 1;
  //const int N[6] = { 256, 512, 768, 1024, 1280, 1536 };
  int N[1] = { 128*128 };

  for (int i=1;i<argc;++i) {
    const std::string& token = argv[i];
    if (token == std::string("-N")) N[0] = std::atoi(argv[++i]);
  }

  constexpr int VectorLength = 16;

  {
    for (int i=0;i<ntest;++i) {
      std::cout << " N = " << N[i] << std::endl;

      std::cout << "\n Testing LayoutLeft-" << VectorLength << " and Algo::Gemm::Unblocked\n";      
      run<VectorLength,double,Algo::Gemm::Unblocked>(N[i]/VectorLength);

      std::cout << "\n Testing LayoutLeft-" << VectorLength << " and Algo::Gemm::Blocked\n";      
      run<VectorLength,double,Algo::Gemm::Blocked>(N[i]/VectorLength);
    }
  }

  Kokkos::finalize();

  return 0;
}
