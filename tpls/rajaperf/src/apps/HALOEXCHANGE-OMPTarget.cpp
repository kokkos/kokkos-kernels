//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HALOEXCHANGE.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{

  //
  // Define threads per team for target execution
  //
  const size_t threads_per_team = 256;

#define HALOEXCHANGE_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  for (Index_type v = 0; v < m_num_vars; ++v) { \
    allocAndInitOpenMPDeviceData(vars[v], m_vars[v], m_var_size, did, hid); \
  } \
  for (Index_type l = 0; l < num_neighbors; ++l) { \
    allocAndInitOpenMPDeviceData(buffers[l], m_buffers[l], m_num_vars*m_pack_index_list_lengths[l], did, hid); \
    allocAndInitOpenMPDeviceData(pack_index_lists[l], m_pack_index_lists[l], m_pack_index_list_lengths[l], did, hid); \
    allocAndInitOpenMPDeviceData(unpack_index_lists[l], m_unpack_index_lists[l], m_unpack_index_list_lengths[l], did, hid); \
  }

#define HALOEXCHANGE_DATA_TEARDOWN_OMP_TARGET \
  for (Index_type l = 0; l < num_neighbors; ++l) { \
    deallocOpenMPDeviceData(unpack_index_lists[l], did); \
    deallocOpenMPDeviceData(pack_index_lists[l], did); \
    deallocOpenMPDeviceData(buffers[l], did); \
  } \
  for (Index_type v = 0; v < m_num_vars; ++v) { \
    getOpenMPDeviceData(m_vars[v], vars[v], m_var_size, hid, did); \
    deallocOpenMPDeviceData(vars[v], did); \
  }


void HALOEXCHANGE::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  HALOEXCHANGE_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    HALOEXCHANGE_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type l = 0; l < num_neighbors; ++l) {
        Real_ptr buffer = buffers[l];
        Int_ptr list = pack_index_lists[l];
        Index_type  len  = pack_index_list_lengths[l];
        for (Index_type v = 0; v < num_vars; ++v) {
          Real_ptr var = vars[v];
          #pragma omp target is_device_ptr(buffer, list, var) device( did )
          #pragma omp teams distribute parallel for schedule(static, 1)
          for (Index_type i = 0; i < len; i++) {
            HALOEXCHANGE_PACK_BODY;
          }
          buffer += len;
        }
      }

      for (Index_type l = 0; l < num_neighbors; ++l) {
        Real_ptr buffer = buffers[l];
        Int_ptr list = unpack_index_lists[l];
        Index_type  len  = unpack_index_list_lengths[l];
        for (Index_type v = 0; v < num_vars; ++v) {
          Real_ptr var = vars[v];
          #pragma omp target is_device_ptr(buffer, list, var) device( did )
          #pragma omp teams distribute parallel for schedule(static, 1)
          for (Index_type i = 0; i < len; i++) {
            HALOEXCHANGE_UNPACK_BODY;
          }
          buffer += len;
        }
      }

    }
    stopTimer();

    HALOEXCHANGE_DATA_TEARDOWN_OMP_TARGET;

  } else if ( vid == RAJA_OpenMPTarget ) {

    HALOEXCHANGE_DATA_SETUP_OMP_TARGET;

    using EXEC_POL = RAJA::omp_target_parallel_for_exec<threads_per_team>;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type l = 0; l < num_neighbors; ++l) {
        Real_ptr buffer = buffers[l];
        Int_ptr list = pack_index_lists[l];
        Index_type  len  = pack_index_list_lengths[l];
        for (Index_type v = 0; v < num_vars; ++v) {
          Real_ptr var = vars[v];
          auto haloexchange_pack_base_lam = [=](Index_type i) {
                HALOEXCHANGE_PACK_BODY;
              };
          RAJA::forall<EXEC_POL>(
              RAJA::TypedRangeSegment<Index_type>(0, len),
              haloexchange_pack_base_lam );
          buffer += len;
        }
      }

      for (Index_type l = 0; l < num_neighbors; ++l) {
        Real_ptr buffer = buffers[l];
        Int_ptr list = unpack_index_lists[l];
        Index_type  len  = unpack_index_list_lengths[l];
        for (Index_type v = 0; v < num_vars; ++v) {
          Real_ptr var = vars[v];
          auto haloexchange_unpack_base_lam = [=](Index_type i) {
                HALOEXCHANGE_UNPACK_BODY;
              };
          RAJA::forall<EXEC_POL>(
              RAJA::TypedRangeSegment<Index_type>(0, len),
              haloexchange_unpack_base_lam );
          buffer += len;
        }
      }

    }
    stopTimer();

    HALOEXCHANGE_DATA_TEARDOWN_OMP_TARGET;

  } else {
     std::cout << "\n HALOEXCHANGE : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
