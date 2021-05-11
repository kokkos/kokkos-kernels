//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// HYDRO_2D kernel reference implementation:
///
/// for (Index_type k=1 ; k<kn-1 ; k++) {
///   for (Index_type j=1 ; j<jn-1 ; j++) {
///     za[k][j] = ( zp[k+1][j-1] +zq[k+1][j-1] -zp[k][j-1] -zq[k][j-1] )*
///                ( zr[k][j] +zr[k][j-1] ) / ( zm[k][j-1] +zm[k+1][j-1]);
///     zb[k][j] = ( zp[k][j-1] +zq[k][j-1] -zp[k][j] -zq[k][j] ) *
///                ( zr[k][j] +zr[k-1][j] ) / ( zm[k][j] +zm[k][j-1]);
///   }
/// }
///
/// for (Index_type k=1 ; k<kn-1 ; k++) {
///   for (Index_type j=1 ; j<jn-1 ; j++) {
///     zu[k][j] += s*( za[k][j]   *( zz[k][j] - zz[k][j+1] ) -
///                     za[k][j-1] *( zz[k][j] - zz[k][j-1] ) -
///                     zb[k][j]   *( zz[k][j] - zz[k-1][j] ) +
///                     zb[k+1][j] *( zz[k][j] - zz[k+1][j] ) );
///     zv[k][j] += s*( za[k][j]   *( zr[k][j] - zr[k][j+1] ) -
///                     za[k][j-1] *( zr[k][j] - zr[k][j-1] ) -
///                     zb[k][j]   *( zr[k][j] - zr[k-1][j] ) +
///                     zb[k+1][j] *( zr[k][j] - zr[k+1][j] ) );
///   }
/// }
///
/// for (Index_type k=1 ; k<kn-1 ; k++) {
///   for (Index_type j=1 ; j<jn-1 ; j++) {
///     zrout[k][j] = zr[k][j] + t*zu[k][j];
///     zzout[k][j] = zz[k][j] + t*zv[k][j];
///   }
/// }
///

#ifndef RAJAPerf_Lcals_HYDRO_2D_HPP
#define RAJAPerf_Lcals_HYDRO_2D_HPP


#define HYDRO_2D_DATA_SETUP \
  Real_ptr zadat = m_za; \
  Real_ptr zbdat = m_zb; \
  Real_ptr zmdat = m_zm; \
  Real_ptr zpdat = m_zp; \
  Real_ptr zqdat = m_zq; \
  Real_ptr zrdat = m_zr; \
  Real_ptr zudat = m_zu; \
  Real_ptr zvdat = m_zv; \
  Real_ptr zzdat = m_zz; \
\
  Real_ptr zroutdat = m_zrout; \
  Real_ptr zzoutdat = m_zzout; \
\
  const Real_type s = m_s; \
  const Real_type t = m_t; \
\
  const Index_type kn = m_kn; \
  const Index_type jn = m_jn;

#define HYDRO_2D_BODY1  \
  zadat[j+k*jn] = ( zpdat[j-1+(k+1)*jn] + zqdat[j-1+(k+1)*jn] - \
                    zpdat[j-1+k*jn] - zqdat[j-1+k*jn] ) * \
                  ( zrdat[j+k*jn] + zrdat[j-1+k*jn] ) / \
                  ( zmdat[j-1+k*jn] + zmdat[j-1+(k+1)*jn] ); \
  zbdat[j+k*jn] = ( zpdat[j-1+k*jn] + zqdat[j-1+k*jn] - \
                    zpdat[j+k*jn] - zqdat[j+k*jn] ) * \
                  ( zrdat[j+k*jn] + zrdat[j+(k-1)*jn] ) / \
                  ( zmdat[j+k*jn] + zmdat[j-1+k*jn] );

#define HYDRO_2D_BODY2 \
  zudat[j+k*jn] += s*( zadat[j+k*jn] * ( zzdat[j+k*jn] - zzdat[j+1+k*jn] ) - \
                    zadat[j-1+k*jn] * ( zzdat[j+k*jn] - zzdat[j-1+k*jn] ) - \
                    zbdat[j+k*jn] * ( zzdat[j+k*jn] - zzdat[j+(k-1)*jn] ) + \
                    zbdat[j+(k+1)*jn] * ( zzdat[j+k*jn] - zzdat[j+(k+1)*jn] ) ); \
  zvdat[j+k*jn] += s*( zadat[j+k*jn] * ( zrdat[j+k*jn] - zrdat[j+1+k*jn] ) - \
                    zadat[j-1+k*jn] * ( zrdat[j+k*jn] - zrdat[j-1+k*jn] ) - \
                    zbdat[j+k*jn] * ( zrdat[j+k*jn] - zrdat[j+(k-1)*jn] ) + \
                    zbdat[j+(k+1)*jn] * ( zrdat[j+k*jn] - zrdat[j+(k+1)*jn] ) );

#define HYDRO_2D_BODY3 \
  zroutdat[j+k*jn] = zrdat[j+k*jn] + t*zudat[j+k*jn]; \
  zzoutdat[j+k*jn] = zzdat[j+k*jn] + t*zvdat[j+k*jn]; \


#define HYDRO_2D_VIEWS_RAJA \
  using VIEW_TYPE = RAJA::View<Real_type, RAJA::Layout<2, Index_type, 1> >; \
\
  std::array<RAJA::idx_t, 2> view_perm {{0, 1}}; \
\
  VIEW_TYPE za(zadat, RAJA::make_permuted_layout({{kn, jn}}, view_perm)); \
  VIEW_TYPE zb(zbdat, RAJA::make_permuted_layout({{kn, jn}}, view_perm)); \
  VIEW_TYPE zm(zmdat, RAJA::make_permuted_layout({{kn, jn}}, view_perm)); \
  VIEW_TYPE zp(zpdat, RAJA::make_permuted_layout({{kn, jn}}, view_perm)); \
  VIEW_TYPE zq(zqdat, RAJA::make_permuted_layout({{kn, jn}}, view_perm)); \
  VIEW_TYPE zr(zrdat, RAJA::make_permuted_layout({{kn, jn}}, view_perm)); \
  VIEW_TYPE zu(zudat, RAJA::make_permuted_layout({{kn, jn}}, view_perm)); \
  VIEW_TYPE zv(zvdat, RAJA::make_permuted_layout({{kn, jn}}, view_perm)); \
  VIEW_TYPE zz(zzdat, RAJA::make_permuted_layout({{kn, jn}}, view_perm)); \
  VIEW_TYPE zrout(zroutdat, RAJA::make_permuted_layout({{kn, jn}}, view_perm));\
  VIEW_TYPE zzout(zzoutdat, RAJA::make_permuted_layout({{kn, jn}}, view_perm));

#define HYDRO_2D_BODY1_RAJA  \
  za(k,j) = ( zp(k+1,j-1) + zq(k+1,j-1) - zp(k,j-1) - zq(k,j-1) ) * \
            ( zr(k,j) + zr(k,j-1) ) / ( zm(k,j-1) + zm(k+1,j-1) ); \
  zb(k,j) = ( zp(k,j-1) + zq(k,j-1) - zp(k,j) - zq(k,j) ) * \
            ( zr(k,j) + zr(k-1,j) ) / ( zm(k,j) + zm(k,j-1));

#define HYDRO_2D_BODY2_RAJA \
  zu(k,j) += s*( za(k,j) * ( zz(k,j) - zz(k,j+1) ) - \
                 za(k,j-1) * ( zz(k,j) - zz(k,j-1) ) - \
                 zb(k,j) * ( zz(k,j) - zz(k-1,j) ) + \
                 zb(k+1,j) * ( zz(k,j) - zz(k+1,j) ) ); \
  zv(k,j) += s*( za(k,j) * ( zr(k,j) - zr(k,j+1) ) - \
                 za(k,j-1) * ( zr(k,j) - zr(k,j-1) ) - \
                 zb(k,j) * ( zr(k,j) - zr(k-1,j) ) + \
                 zb(k+1,j) * ( zr(k,j) - zr(k+1,j) ) );

#define HYDRO_2D_BODY3_RAJA \
  zrout(k,j) = zr(k,j) + t*zu(k,j); \
  zzout(k,j) = zz(k,j) + t*zv(k,j);



#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace lcals
{

class HYDRO_2D : public KernelBase
{
public:

  HYDRO_2D(const RunParams& params);

  ~HYDRO_2D();

  void setUp(VariantID vid);
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);
  void runCudaVariant(VariantID vid);
  void runHipVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Real_ptr m_za;
  Real_ptr m_zb;
  Real_ptr m_zm;
  Real_ptr m_zp;
  Real_ptr m_zq;
  Real_ptr m_zr;
  Real_ptr m_zu;
  Real_ptr m_zv;
  Real_ptr m_zz;

  Real_ptr m_zrout;
  Real_ptr m_zzout;

  Real_type m_s;
  Real_type m_t;

  Index_type m_jn;
  Index_type m_kn;

  Index_type m_array_length;
};

} // end namespace lcals
} // end namespace rajaperf

#endif // closing endif for header file include guard
