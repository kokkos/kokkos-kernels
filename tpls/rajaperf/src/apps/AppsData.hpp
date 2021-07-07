//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJAPerf_AppsData_HPP
#define RAJAPerf_AppsData_HPP

#include "common/RPTypes.hpp"

namespace rajaperf
{
namespace apps
{

//
// Some macros used in kernels to mimic real app code style.
//
#define NDPTRSET(jp, kp,v,v0,v1,v2,v3,v4,v5,v6,v7)  \
   v0 = v ;   \
   v1 = v0 + 1 ;  \
   v2 = v0 + jp ; \
   v3 = v1 + jp ; \
   v4 = v0 + kp ; \
   v5 = v1 + kp ; \
   v6 = v2 + kp ; \
   v7 = v3 + kp ;

#define NDSET2D(jp,v,v1,v2,v3,v4)  \
   v4 = v ;   \
   v1 = v4 + 1 ;  \
   v2 = v1 + jp ;  \
   v3 = v4 + jp ;

#define zabs2(z)    ( real(z)*real(z)+imag(z)*imag(z) )


//
// Domain structure to mimic structured mesh loops code style.
//
class ADomain
{
public:

   ADomain() = delete;

   ADomain( Index_type rzmax, Index_type ndims ) 
      : ndims(ndims), NPNL(2), NPNR(1)
   {
      imin = NPNL;
      jmin = NPNL;
      imax = rzmax + NPNR;
      jmax = rzmax + NPNR;
      jp = imax - imin + 1 + NPNL + NPNR;

      if ( ndims == 2 ) {
         kmin = 0;
         kmax = 0;
         kp = 0;
         nnalls = jp * (jmax - jmin + 1 + NPNL + NPNR) ;
      } else if ( ndims == 3 ) {
         kmin = NPNL;
         kmax = rzmax + NPNR;
         kp = jp * (jmax - jmin + 1 + NPNL + NPNR);
         nnalls = kp * (kmax - kmin + 1 + NPNL + NPNR) ;
      }

      fpn = 0;
      lpn = nnalls - 1;
      frn = fpn + NPNL * (kp + jp) + NPNL;
      lrn = lpn - NPNR * (kp + jp) - NPNR;

      fpz = frn - jp - kp - 1;
      lpz = lrn;

      real_zones = new Index_type[nnalls];
      for (Index_type i = 0; i < nnalls; ++i) real_zones[i] = -1;

      n_real_zones = 0;

      if ( ndims == 2 ) {

         for (Index_type j = jmin; j < jmax; j++) {
            for (Index_type i = imin; i < imax; i++) {
               Index_type ip = i + j*jp ;

               Index_type id = n_real_zones;
               real_zones[id] = ip;
               n_real_zones++;
            }
         }

      } else if ( ndims == 3 ) {

         for (Index_type k = kmin; k < kmax; k++) { 
            for (Index_type j = jmin; j < jmax; j++) {
               for (Index_type i = imin; i < imax; i++) {
                  Index_type ip = i + j*jp + kp*k ;

                  Index_type id = n_real_zones;
                  real_zones[id] = ip;
                  n_real_zones++;
               }
            }
         } 

      }

   }

   ~ADomain() 
   {
      if (real_zones) delete [] real_zones; 
   }

   Index_type ndims;
   Index_type NPNL;
   Index_type NPNR;

   Index_type imin;
   Index_type jmin;
   Index_type kmin;
   Index_type imax;
   Index_type jmax;
   Index_type kmax;

   Index_type jp;
   Index_type kp;
   Index_type nnalls;

   Index_type fpn;
   Index_type lpn;
   Index_type frn;
   Index_type lrn;

   Index_type fpz;
   Index_type lpz;

   Index_type* real_zones;
   Index_type  n_real_zones;
};

//
// Routines for initializing mesh positions for 2d/3d domains.
//
void setMeshPositions_2d(Real_ptr x, Real_type dx,
                         Real_ptr y, Real_type dy,
                         const ADomain& domain);

void setMeshPositions_3d(Real_ptr x, Real_type dx,
                         Real_ptr y, Real_type dy,
                         Real_ptr z, Real_type dz,
                         const ADomain& domain);

} // end namespace apps
} // end namespace rajaperf

#endif  // closing endif for header file include guard
