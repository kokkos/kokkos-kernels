//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "AppsData.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{

//
// Set mesh positions for 2d mesh.
//
void setMeshPositions_2d(Real_ptr x, Real_type dx,
                         Real_ptr y, Real_type dy,
                         const ADomain& domain)
{
  if (domain.ndims != 2) {
    std::cout << "\n******* ERROR!!! domain is not 2d *******" << std::endl;
    return;
  }

  Index_type imin = domain.imin;
  Index_type imax = domain.imax;
  Index_type jmin = domain.jmin;
  Index_type jmax = domain.jmax;

  Index_type jp = domain.jp;

  Index_type npnl = domain.NPNL; 
  Index_type npnr = domain.NPNR; 

  Real_ptr x1, x2, x3, x4;
  Real_ptr y1, y2, y3, y4;
  NDSET2D(domain.jp, x, x1,x2,x3,x4) ;
  NDSET2D(domain.jp, y, y1,y2,y3,y4) ;

  for (Index_type j = jmin - npnl; j < jmax + npnr; j++) {
     for (Index_type i = imin - npnl; i < imax + npnr; i++) {
        Index_type iz = i + j*jp ;

        x3[iz] = x4[iz] = i*dx;
        x1[iz] = x2[iz] = (i+1)*dx;

        y1[iz] = y4[iz] = j*dy;
        y2[iz] = y3[iz] = (j+1)*dy;

     }
  }
}


//
// Set mesh positions for 2d mesh.
//
void setMeshPositions_3d(Real_ptr x, Real_type dx,
                         Real_ptr y, Real_type dy,
                         Real_ptr z, Real_type dz,
                         const ADomain& domain)
{
  if (domain.ndims != 3) {
    std::cout << "\n******* ERROR!!! domain is not 3d *******" << std::endl;
    return;
  }

  Index_type imin = domain.imin;
  Index_type imax = domain.imax;
  Index_type jmin = domain.jmin;
  Index_type jmax = domain.jmax;
  Index_type kmin = domain.kmin;
  Index_type kmax = domain.kmax;

  Index_type jp = domain.jp;
  Index_type kp = domain.kp;

  Index_type npnl = domain.NPNL; 
  Index_type npnr = domain.NPNR; 

  Real_ptr x0, x1, x2, x3, x4, x5, x6, x7;
  Real_ptr y0, y1, y2, y3, y4, y5, y6, y7;
  Real_ptr z0, z1, z2, z3, z4, z5, z6, z7;
  NDPTRSET(domain.jp, domain.kp, x,x0,x1,x2,x3,x4,x5,x6,x7) ;
  NDPTRSET(domain.jp, domain.kp, y,y0,y1,y2,y3,y4,y5,y6,y7) ;
  NDPTRSET(domain.jp, domain.kp, z,z0,z1,z2,z3,z4,z5,z6,z7) ;

  for (Index_type k = kmin - npnl; k < kmax + npnr; k++) {
     for (Index_type j = jmin - npnl; j < jmax + npnr; j++) {
        for (Index_type i = imin - npnl; i < imax + npnr; i++) {
           Index_type iz = i + j*jp + kp*k ;

           x0[iz] = x2[iz] = x4[iz] = x6[iz] = i*dx;
           x1[iz] = x3[iz] = x5[iz] = x7[iz] = (i+1)*dx;

           y0[iz] = y1[iz] = y4[iz] = y5[iz] = j*dy;
           y2[iz] = y3[iz] = y6[iz] = y7[iz] = (j+1)*dy;

           z0[iz] = z1[iz] = z2[iz] = z3[iz] = k*dz;
           z4[iz] = z5[iz] = z6[iz] = z7[iz] = (k+1)*dz;

        }
     }
  }
}

} // end namespace apps
} // end namespace rajaperf
