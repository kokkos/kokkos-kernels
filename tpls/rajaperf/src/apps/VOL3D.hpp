//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// VOL3D kernel reference implementation:
///
/// NDPTRSET(m_domain->jp, m_domain->kp, x,x0,x1,x2,x3,x4,x5,x6,x7) ;
/// NDPTRSET(m_domain->jp, m_domain->kp, y,y0,y1,y2,y3,y4,y5,y6,y7) ;
/// NDPTRSET(m_domain->jp, m_domain->kp, z,z0,z1,z2,z3,z4,z5,z6,z7) ;
///
/// for (Index_type i = ibegin ; i < iend ; ++i ) {
///   Real_type x71 = x7[i] - x1[i] ;
///   Real_type x72 = x7[i] - x2[i] ;
///   Real_type x74 = x7[i] - x4[i] ;
///   Real_type x30 = x3[i] - x0[i] ;
///   Real_type x50 = x5[i] - x0[i] ;
///   Real_type x60 = x6[i] - x0[i] ;
///
///   Real_type y71 = y7[i] - y1[i] ;
///   Real_type y72 = y7[i] - y2[i] ;
///   Real_type y74 = y7[i] - y4[i] ;
///   Real_type y30 = y3[i] - y0[i] ;
///   Real_type y50 = y5[i] - y0[i] ;
///   Real_type y60 = y6[i] - y0[i] ;
///
///   Real_type z71 = z7[i] - z1[i] ;
///   Real_type z72 = z7[i] - z2[i] ;
///   Real_type z74 = z7[i] - z4[i] ;
///   Real_type z30 = z3[i] - z0[i] ;
///   Real_type z50 = z5[i] - z0[i] ;
///   Real_type z60 = z6[i] - z0[i] ;
///
///   Real_type xps = x71 + x60 ;
///   Real_type yps = y71 + y60 ;
///   Real_type zps = z71 + z60 ;
///
///   Real_type cyz = y72 * z30 - z72 * y30 ;
///   Real_type czx = z72 * x30 - x72 * z30 ;
///   Real_type cxy = x72 * y30 - y72 * x30 ;
///   vol[i] = xps * cyz + yps * czx + zps * cxy ;
///
///   xps = x72 + x50 ;
///   yps = y72 + y50 ;
///   zps = z72 + z50 ;
///
///   cyz = y74 * z60 - z74 * y60 ;
///   czx = z74 * x60 - x74 * z60 ;
///   cxy = x74 * y60 - y74 * x60 ;
///   vol[i] += xps * cyz + yps * czx + zps * cxy ;
///
///   xps = x74 + x30 ;
///   yps = y74 + y30 ;
///   zps = z74 + z30 ;
///
///   cyz = y71 * z50 - z71 * y50 ;
///   czx = z71 * x50 - x71 * z50 ;
///   cxy = x71 * y50 - y71 * x50 ;
///   vol[i] += xps * cyz + yps * czx + zps * cxy ;
///
///   vol[i] *= vnormq ;
/// }
///

#ifndef RAJAPerf_Apps_VOL3D_HPP
#define RAJAPerf_Apps_VOL3D_HPP

#define VOL3D_DATA_SETUP \
  Real_ptr x = m_x; \
  Real_ptr y = m_y; \
  Real_ptr z = m_z; \
  Real_ptr vol = m_vol; \
\
  const Real_type vnormq = m_vnormq; \
\
  Real_ptr x0,x1,x2,x3,x4,x5,x6,x7 ; \
  Real_ptr y0,y1,y2,y3,y4,y5,y6,y7 ; \
  Real_ptr z0,z1,z2,z3,z4,z5,z6,z7 ;

#define VOL3D_BODY \
  Real_type x71 = x7[i] - x1[i] ; \
  Real_type x72 = x7[i] - x2[i] ; \
  Real_type x74 = x7[i] - x4[i] ; \
  Real_type x30 = x3[i] - x0[i] ; \
  Real_type x50 = x5[i] - x0[i] ; \
  Real_type x60 = x6[i] - x0[i] ; \
 \
  Real_type y71 = y7[i] - y1[i] ; \
  Real_type y72 = y7[i] - y2[i] ; \
  Real_type y74 = y7[i] - y4[i] ; \
  Real_type y30 = y3[i] - y0[i] ; \
  Real_type y50 = y5[i] - y0[i] ; \
  Real_type y60 = y6[i] - y0[i] ; \
 \
  Real_type z71 = z7[i] - z1[i] ; \
  Real_type z72 = z7[i] - z2[i] ; \
  Real_type z74 = z7[i] - z4[i] ; \
  Real_type z30 = z3[i] - z0[i] ; \
  Real_type z50 = z5[i] - z0[i] ; \
  Real_type z60 = z6[i] - z0[i] ; \
 \
  Real_type xps = x71 + x60 ; \
  Real_type yps = y71 + y60 ; \
  Real_type zps = z71 + z60 ; \
 \
  Real_type cyz = y72 * z30 - z72 * y30 ; \
  Real_type czx = z72 * x30 - x72 * z30 ; \
  Real_type cxy = x72 * y30 - y72 * x30 ; \
  vol[i] = xps * cyz + yps * czx + zps * cxy ; \
 \
  xps = x72 + x50 ; \
  yps = y72 + y50 ; \
  zps = z72 + z50 ; \
 \
  cyz = y74 * z60 - z74 * y60 ; \
  czx = z74 * x60 - x74 * z60 ; \
  cxy = x74 * y60 - y74 * x60 ; \
  vol[i] += xps * cyz + yps * czx + zps * cxy ; \
 \
  xps = x74 + x30 ; \
  yps = y74 + y30 ; \
  zps = z74 + z30 ; \
 \
  cyz = y74 * z60 - z74 * y60 ; \
  czx = z74 * x60 - x74 * z60 ; \
  cxy = x74 * y60 - y74 * x60 ; \
  vol[i] += xps * cyz + yps * czx + zps * cxy ; \
 \
  xps = x74 + x30 ; \
  yps = y74 + y30 ; \
  zps = z74 + z30 ; \
 \
  cyz = y71 * z50 - z71 * y50 ; \
  czx = z71 * x50 - x71 * z50 ; \
  cxy = x71 * y50 - y71 * x50 ; \
  vol[i] += xps * cyz + yps * czx + zps * cxy ; \
 \
  vol[i] *= vnormq ;


#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace apps
{
class ADomain;

class VOL3D : public KernelBase
{
public:

  VOL3D(const RunParams& params);

  ~VOL3D();

  Index_type getItsPerRep() const;

  void setUp(VariantID vid);
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);
  void runCudaVariant(VariantID vid);
  void runHipVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Real_ptr m_x;
  Real_ptr m_y;
  Real_ptr m_z;
  Real_ptr m_vol;

  Real_type m_vnormq;

  ADomain* m_domain;
  Index_type m_array_length; 
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
