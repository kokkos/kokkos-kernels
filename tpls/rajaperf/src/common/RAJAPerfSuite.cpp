//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJAPerfSuite.hpp"

#include "RunParams.hpp"

#ifndef RAJAPERF_INFRASTRUCTURE_ONLY
#include "PerfsuiteKernelDefinitions.hpp"
#endif

namespace rajaperf {
 /*******************************************************************************
 *
 * \brief Array of names for each VARIANT in suite.
 *
 * IMPORTANT: This is only modified when a new kernel is added to the suite.
 *
 *            IT MUST BE KEPT CONSISTENT (CORRESPONDING ONE-TO-ONE) WITH
 *            ENUM OF VARIANT IDS IN HEADER FILE!!!
 *
 *******************************************************************************
 */
    static const std::string VariantNames[] =
            {

                    std::string("Base_Seq"),
                    std::string("Lambda_Seq"),
                    std::string("RAJA_Seq"),

                    std::string("Base_OpenMP"),
                    std::string("Lambda_OpenMP"),
                    std::string("RAJA_OpenMP"),

                    std::string("Base_OMPTarget"),
                    std::string("RAJA_OMPTarget"),

                    std::string("Base_CUDA"),
                    std::string("Lambda_CUDA"),
                    std::string("RAJA_CUDA"),
                    std::string("RAJA_WORKGROUP_CUDA"),

                    std::string("Base_HIP"),
                    std::string("Lambda_HIP"),
                    std::string("RAJA_HIP"),
                    std::string("RAJA_WORKGROUP_HIP"),

                    std::string("Kokkos_Lambda"),
                    std::string("Kokkos_Functor"),

                    std::string("Unknown Variant")  // Keep this at the end and DO NOT remove....

            }; // END VariantNames


/*
 *******************************************************************************
 *
 * Return variant name associated with VariantID enum value.
 *
 *******************************************************************************
 */
    const std::string &getVariantName(VariantID vid) {
        return VariantNames[vid];
    }

/*!
 *******************************************************************************
 *
 * Return true if variant associated with VariantID enum value is available 
 * to run; else false.
 *
 *******************************************************************************
 */
    bool isVariantAvailable(VariantID vid) {
        bool ret_val = false;

        if (vid == Base_Seq) {
            ret_val = true;
        }
#if defined(RUN_RAJA_SEQ)
        if (vid == Lambda_Seq ||
            vid == RAJA_Seq) {
            ret_val = true;
        }
#endif

#if defined(RUN_KOKKOS) or defined(RAJAPERF_INFRASTRUCTURE_ONLY)
        if (vid == Kokkos_Lambda ||
            vid == Kokkos_Functor) {
            ret_val = true;
        }
#endif // RUN_KOKKOS

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
        if ( vid == Base_OpenMP ||
             vid == Lambda_OpenMP ||
             vid == RAJA_OpenMP ) {
          ret_val = true;
        }
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
        if ( vid == Base_OpenMPTarget ||
             vid == RAJA_OpenMPTarget ) {
          ret_val = true;
        }
#endif

#if defined(RAJA_ENABLE_CUDA)
        if ( vid == Base_CUDA ||
             vid == Lambda_CUDA ||
             vid == RAJA_CUDA ||
             vid == RAJA_WORKGROUP_CUDA ) {
          ret_val = true;
        }
#endif

#if defined(RAJA_ENABLE_HIP)
        if ( vid == Base_HIP ||
             vid == Lambda_HIP ||
             vid == RAJA_HIP ||
             vid == RAJA_WORKGROUP_HIP ) {
          ret_val = true;
        }
#endif

        return ret_val;
    }


}  // closing brace for rajaperf namespace
