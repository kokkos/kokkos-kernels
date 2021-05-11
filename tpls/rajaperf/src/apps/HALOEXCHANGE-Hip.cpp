//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HALOEXCHANGE.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{

  //
  // Define thread block size for HIP execution
  //
  const size_t block_size = 256;


#define HALOEXCHANGE_DATA_SETUP_HIP \
  for (Index_type v = 0; v < m_num_vars; ++v) { \
    allocAndInitHipDeviceData(vars[v], m_vars[v], m_var_size); \
  } \
  for (Index_type l = 0; l < num_neighbors; ++l) { \
    allocAndInitHipDeviceData(buffers[l], m_buffers[l], m_num_vars*m_pack_index_list_lengths[l]); \
    allocAndInitHipDeviceData(pack_index_lists[l], m_pack_index_lists[l], m_pack_index_list_lengths[l]); \
    allocAndInitHipDeviceData(unpack_index_lists[l], m_unpack_index_lists[l], m_unpack_index_list_lengths[l]); \
  }

#define HALOEXCHANGE_DATA_TEARDOWN_HIP \
  for (Index_type l = 0; l < num_neighbors; ++l) { \
    deallocHipDeviceData(unpack_index_lists[l]); \
    deallocHipDeviceData(pack_index_lists[l]); \
    deallocHipDeviceData(buffers[l]); \
  } \
  for (Index_type v = 0; v < m_num_vars; ++v) { \
    getHipDeviceData(m_vars[v], vars[v], m_var_size); \
    deallocHipDeviceData(vars[v]); \
  }

__global__ void haloexchange_pack(Real_ptr buffer, Int_ptr list, Real_ptr var,
                                  Index_type len)
{
   Index_type i = threadIdx.x + blockIdx.x * blockDim.x;

   if (i < len) {
     HALOEXCHANGE_PACK_BODY;
   }
}

__global__ void haloexchange_unpack(Real_ptr buffer, Int_ptr list, Real_ptr var,
                                    Index_type len)
{
   Index_type i = threadIdx.x + blockIdx.x * blockDim.x;

   if (i < len) {
     HALOEXCHANGE_UNPACK_BODY;
   }
}


void HALOEXCHANGE::runHipVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  HALOEXCHANGE_DATA_SETUP;

  if ( vid == Base_HIP ) {

    HALOEXCHANGE_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type l = 0; l < num_neighbors; ++l) {
        Real_ptr buffer = buffers[l];
        Int_ptr list = pack_index_lists[l];
        Index_type  len  = pack_index_list_lengths[l];
        for (Index_type v = 0; v < num_vars; ++v) {
          Real_ptr var = vars[v];
          dim3 nthreads_per_block(block_size);
          dim3 nblocks((len + block_size-1) / block_size);
          hipLaunchKernelGGL((haloexchange_pack), nblocks, nthreads_per_block, 0, 0,
              buffer, list, var, len);
          buffer += len;
        }
      }
      synchronize();

      for (Index_type l = 0; l < num_neighbors; ++l) {
        Real_ptr buffer = buffers[l];
        Int_ptr list = unpack_index_lists[l];
        Index_type  len  = unpack_index_list_lengths[l];
        for (Index_type v = 0; v < num_vars; ++v) {
          Real_ptr var = vars[v];
          dim3 nthreads_per_block(block_size);
          dim3 nblocks((len + block_size-1) / block_size);
          hipLaunchKernelGGL((haloexchange_unpack), nblocks, nthreads_per_block, 0, 0,
              buffer, list, var, len);
          buffer += len;
        }
      }
      synchronize();

    }
    stopTimer();

    HALOEXCHANGE_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    HALOEXCHANGE_DATA_SETUP_HIP;

    using EXEC_POL = RAJA::hip_exec<block_size, true /*async*/>;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type l = 0; l < num_neighbors; ++l) {
        Real_ptr buffer = buffers[l];
        Int_ptr list = pack_index_lists[l];
        Index_type  len  = pack_index_list_lengths[l];
        for (Index_type v = 0; v < num_vars; ++v) {
          Real_ptr var = vars[v];
          auto haloexchange_pack_base_lam = [=] __device__ (Index_type i) {
                HALOEXCHANGE_PACK_BODY;
              };
          RAJA::forall<EXEC_POL>(
              RAJA::TypedRangeSegment<Index_type>(0, len),
              haloexchange_pack_base_lam );
          buffer += len;
        }
      }
      synchronize();

      for (Index_type l = 0; l < num_neighbors; ++l) {
        Real_ptr buffer = buffers[l];
        Int_ptr list = unpack_index_lists[l];
        Index_type  len  = unpack_index_list_lengths[l];
        for (Index_type v = 0; v < num_vars; ++v) {
          Real_ptr var = vars[v];
          auto haloexchange_unpack_base_lam = [=] __device__ (Index_type i) {
                HALOEXCHANGE_UNPACK_BODY;
              };
          RAJA::forall<EXEC_POL>(
              RAJA::TypedRangeSegment<Index_type>(0, len),
              haloexchange_unpack_base_lam );
          buffer += len;
        }
      }
      synchronize();

    }
    stopTimer();

    HALOEXCHANGE_DATA_TEARDOWN_HIP;

#ifdef RAJA_ENABLE_HIP_INDIRECT_FUNCTION_CALL

  } else if ( vid == RAJA_WORKGROUP_HIP ) {

    HALOEXCHANGE_DATA_SETUP_HIP;

    using AllocatorHolder = RAJAPoolAllocatorHolder<RAJA::hip::pinned_mempool_type>;
    using Allocator = AllocatorHolder::Allocator<char>;

    AllocatorHolder allocatorHolder;

    using workgroup_policy = RAJA::WorkGroupPolicy <
                                 RAJA::hip_work_async<workgroup_block_size>,
                                 RAJA::unordered_hip_loop_y_block_iter_x_threadblock_average,
                                 RAJA::constant_stride_array_of_objects >;

    using workpool = RAJA::WorkPool< workgroup_policy,
                                     Index_type,
                                     RAJA::xargs<>,
                                     Allocator >;

    using workgroup = RAJA::WorkGroup< workgroup_policy,
                                       Index_type,
                                       RAJA::xargs<>,
                                       Allocator >;

    using worksite = RAJA::WorkSite< workgroup_policy,
                                     Index_type,
                                     RAJA::xargs<>,
                                     Allocator >;

    // do a warmup to avoid timing one-time allocations
    {
      workpool pool(allocatorHolder.template getAllocator<char>());
      for (Index_type l = 0; l < num_neighbors; ++l) {
        Index_type len = pack_index_list_lengths[l];
        for (Index_type v = 0; v < num_vars; ++v) {
          pool.enqueue(
              RAJA::TypedRangeSegment<Index_type>(0, len),
              [=] __device__ (Index_type) { });
        }
      }
      workgroup group = pool.instantiate();
      worksite site = group.run();
      synchronize();
    }

    startTimer();
    {
      workpool pool_pack  (allocatorHolder.template getAllocator<char>());
      workpool pool_unpack(allocatorHolder.template getAllocator<char>());

      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type l = 0; l < num_neighbors; ++l) {
          Real_ptr buffer = buffers[l];
          Int_ptr list = pack_index_lists[l];
          Index_type len = pack_index_list_lengths[l];
          for (Index_type v = 0; v < num_vars; ++v) {
            Real_ptr var = vars[v];
            auto haloexchange_pack_base_lam = [=] __device__ (Index_type i) {
                  HALOEXCHANGE_PACK_BODY;
                };
            pool_pack.enqueue(
                RAJA::TypedRangeSegment<Index_type>(0, len),
                haloexchange_pack_base_lam );
            buffer += len;
          }
        }
        workgroup group_pack = pool_pack.instantiate();
        worksite site_pack = group_pack.run();
        synchronize();

        for (Index_type l = 0; l < num_neighbors; ++l) {
          Real_ptr buffer = buffers[l];
          Int_ptr list = unpack_index_lists[l];
          Index_type len = unpack_index_list_lengths[l];
          for (Index_type v = 0; v < num_vars; ++v) {
            Real_ptr var = vars[v];
            auto haloexchange_unpack_base_lam = [=] __device__ (Index_type i) {
                  HALOEXCHANGE_UNPACK_BODY;
                };
            pool_unpack.enqueue(
                RAJA::TypedRangeSegment<Index_type>(0, len),
                haloexchange_unpack_base_lam );
            buffer += len;
          }
        }
        workgroup group_unpack = pool_unpack.instantiate();
        worksite site_unpack = group_unpack.run();
        synchronize();

      }
    }
    stopTimer();

    HALOEXCHANGE_DATA_TEARDOWN_HIP;

#endif

  } else {
     std::cout << "\n HALOEXCHANGE : Unknown Hip variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
