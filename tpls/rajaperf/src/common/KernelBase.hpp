//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJAPerf_KernelBase_HPP
#define RAJAPerf_KernelBase_HPP

#include "common/RAJAPerfSuite.hpp"
//#include "common/RPTypes.hpp"
#include "common/RunParams.hpp"
#ifndef RAJAPERF_INFRASTRUCTURE_ONLY
#include "RAJA/util/Timer.hpp"
#include "common/DataUtils.hpp"
#else
#include "common/BuiltinTimer.hpp"
#endif
#if defined(RAJA_ENABLE_CUDA)
#include "RAJA/policy/cuda/raja_cudaerrchk.hpp"
#endif
#if defined(RAJA_ENABLE_HIP)
#include "RAJA/policy/hip/raja_hiperrchk.hpp"
#endif

#include <string>
#include <iostream>

namespace rajaperf {

/*!
 *******************************************************************************
 *
 * \brief Pure virtual base class for all Suite kernels.
 *
 *******************************************************************************
 */
class KernelBase
{
public:

 // KernelBase(KernelID kid, const RunParams& params);
  KernelBase(std::string name, const RunParams& params);

#ifndef RAJAPERF_INFRASTRUCTURE_ONLY
   using TimerType = RAJA::Timer;
#else
   using TimerType = rajaperf::ChronoTimer;
#endif
  virtual ~KernelBase();

  const std::string& getName() const { return name; }
  void setName(const std::string& new_name) { name = new_name; }

  Index_type getDefaultSize() const { return default_size; }
  Index_type getDefaultReps() const { return default_reps; }

  SizeSpec getSizeSpec() {return run_params.getSizeSpec();}

  void setDefaultSize(Index_type size) { default_size = size; }
  void setDefaultReps(Index_type reps) { default_reps = reps; }

  Index_type getRunSize() const;
  Index_type getRunReps() const;

  bool wasVariantRun(VariantID vid) const 
    { return num_exec[vid] > 0; }

  double getMinTime(VariantID vid) const { return min_time[vid]; }
  double getMaxTime(VariantID vid) const { return max_time[vid]; }
  double getTotTime(VariantID vid) { return tot_time[vid]; }
  Checksum_type getChecksum(VariantID vid) const { return checksum[vid]; }

  bool hasVariantToRun(VariantID vid) const { return has_variant_to_run[vid]; }

  void setVariantDefined(VariantID vid);

  void execute(VariantID vid);

  void synchronize()
  {
#if defined(RAJA_ENABLE_CUDA)
    if ( running_variant == Base_CUDA ||
         running_variant == Lambda_CUDA ||
         running_variant == RAJA_CUDA ||
         running_variant == RAJA_WORKGROUP_CUDA ) {
      cudaErrchk( cudaDeviceSynchronize() );
    }
#endif
#if defined(RAJA_ENABLE_HIP)
    if ( running_variant == Base_HIP ||
         running_variant == Lambda_HIP ||
         running_variant == RAJA_HIP ||
         running_variant == RAJA_WORKGROUP_HIP ) {
      hipErrchk( hipDeviceSynchronize() );
    }
#endif
  }

  void startTimer()
  {
    synchronize();
    timer.start();
  }

  void stopTimer()
  {
    synchronize();
    timer.stop(); recordExecTime();
  }

  void resetTimer(
          ) {
      timer.reset();
  }

  //
  // Virtual and pure virtual methods that may/must be implemented
  // by each concrete kernel class.
  //

  virtual Index_type getItsPerRep() const { return getRunSize(); }

  virtual void print(std::ostream& os) const; 

  virtual void runKernel(VariantID vid);

  virtual void setUp(VariantID vid) = 0;
  virtual void updateChecksum(VariantID vid) = 0;
  virtual void tearDown(VariantID vid) = 0;

  virtual void runSeqVariant(VariantID vid) = 0;
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
  virtual void runOpenMPVariant(VariantID vid) = 0;
#endif
#if defined(RAJA_ENABLE_CUDA)
  virtual void runCudaVariant(VariantID vid) = 0;
#endif
#if defined(RAJA_ENABLE_HIP)
  virtual void runHipVariant(VariantID vid) = 0;
#endif
#if defined(RAJA_ENABLE_TARGET_OPENMP)
  virtual void runOpenMPTargetVariant(VariantID vid) = 0;
#endif

#if defined(RUN_KOKKOS) or defined(RAJAPERF_INFRASTRUCTURE_ONLY)
  virtual void runKokkosVariant(VariantID vid) = 0;
#endif // RUN_KOKKOS

protected:
  const RunParams& run_params;

  Checksum_type checksum[NumVariants];

private:
  KernelBase() = delete;

  void recordExecTime(); 

  std::string name;

  Index_type default_size;
  Index_type default_reps;

  VariantID running_variant; 

  int num_exec[NumVariants];
  TimerType timer;

  TimerType::ElapsedType min_time[NumVariants];
  TimerType::ElapsedType max_time[NumVariants];
  TimerType::ElapsedType tot_time[NumVariants];
  bool has_variant_to_run[NumVariants];
};

}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
