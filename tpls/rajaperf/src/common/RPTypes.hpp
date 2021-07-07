//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Basic data types used in the Suite.
///

#ifndef RAJAPerf_RPTypes_HPP
#define RAJAPerf_RPTypes_HPP
#ifndef RAJAPERF_INFRASTRUCTURE_ONLY
#include "RAJA/util/types.hpp"
#endif
//
// Only one of the following (double or float) should be defined.
// 
#define RP_USE_DOUBLE
//#undef RP_USE_DOUBLE

//#define RP_USE_FLOAT
#undef RP_USE_FLOAT

#define RP_USE_COMPLEX
//#undef RP_USE_COMPLEX

#if defined(RP_USE_COMPLEX)
#include <complex>
#endif


namespace rajaperf
{


/*!
 ******************************************************************************
 *
 * \brief Type used for indexing in all kernel repetition loops.
 *
 * It is volatile to ensure that kernels will not be optimized away by 
 * compilers, which can happen in some circumstances.
 *
 ******************************************************************************
 */
using RepIndex_type = volatile int;


/*!
 ******************************************************************************
 *
 * \brief Types used for all kernel loop indexing.
 *
 ******************************************************************************
 */
#ifndef RAJAPERF_INFRASTRUCTURE_ONLY
using Index_type = RAJA::Index_type;
#else
using Index_type = int64_t;
#endif
///
using Index_ptr = Index_type*;


/*!
 ******************************************************************************
 *
 * \brief Integer types used in kernels.
 *
 ******************************************************************************
 */
using Int_type = int;
///
using Int_ptr = Int_type*;


/*!
 ******************************************************************************
 *
 * \brief Type used for all kernel checksums.
 *
 ******************************************************************************
 */
using Checksum_type = long double;


/*!
 ******************************************************************************
 *
 * \brief Floating point types used in kernels.
 *
 ******************************************************************************
 */
#if defined(RP_USE_DOUBLE)
///
using Real_type = double;

#elif defined(RP_USE_FLOAT)
///
using Real_type = float;

#else
#error Real_type is undefined!

#endif

using Real_ptr = Real_type*;

#if defined(RP_USE_COMPLEX)
///
using Complex_type = std::complex<Real_type>;

using Complex_ptr = Complex_type*;
#endif




}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
