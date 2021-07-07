//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Utility methods for generating output reports.
///

#ifndef RAJAPerf_OutputUtils_HPP
#define RAJAPerf_OutputUtils_HPP

#include <string>

namespace rajaperf
{

/*!
 * \brief Recursively construct directories based on a relative or 
 * absolute path name.  
 * 
 * Return string name of directory if created successfully, else empty string.
 */
std::string recursiveMkdir(const std::string& in_path);

}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
