//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HALOEXCHANGE.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{


void HALOEXCHANGE::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();

  HALOEXCHANGE_DATA_SETUP;

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type l = 0; l < num_neighbors; ++l) {
          Real_ptr buffer = buffers[l];
          Int_ptr list = pack_index_lists[l];
          Index_type  len  = pack_index_list_lengths[l];
          for (Index_type v = 0; v < num_vars; ++v) {
            Real_ptr var = vars[v];
            #pragma omp parallel for
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
            #pragma omp parallel for
            for (Index_type i = 0; i < len; i++) {
              HALOEXCHANGE_UNPACK_BODY;
            }
            buffer += len;
          }
        }

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

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
            #pragma omp parallel for
            for (Index_type i = 0; i < len; i++) {
              haloexchange_pack_base_lam(i);
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
            auto haloexchange_unpack_base_lam = [=](Index_type i) {
                  HALOEXCHANGE_UNPACK_BODY;
                };
            #pragma omp parallel for
            for (Index_type i = 0; i < len; i++) {
              haloexchange_unpack_base_lam(i);
            }
            buffer += len;
          }
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      using EXEC_POL = RAJA::omp_parallel_for_exec;

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

      break;
    }

    default : {
      std::cout << "\n HALOEXCHANGE : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace apps
} // end namespace rajaperf
