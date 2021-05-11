//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// LTIMES kernel reference implementation:
///
/// for (Index_type z = 0; z < num_z; ++z ) {
///   for (Index_type g = 0; g < num_g; ++g ) {
///     for (Index_type m = 0; z < num_m; ++m ) {
///       for (Index_type d = 0; d < num_d; ++d ) {
///
///         phi[m+ (g * num_m) + (z * num_m * num_g)] +=
///           ell[d+ (m * num_d)] * psi[d+ (g * num_d) + (z * num_d * num_g];
///
///       }
///     }
///   }
/// }
///
/// The RAJA variants of this kernel use RAJA multi-dimensional data layouts 
/// and views to do the same thing without explicit index calculations (see
/// the loop body definitions below).
///

#ifndef RAJAPerf_Apps_LTIMES_HPP
#define RAJAPerf_Apps_LTIMES_HPP

#define LTIMES_DATA_SETUP \
  Real_ptr phidat = m_phidat; \
  Real_ptr elldat = m_elldat; \
  Real_ptr psidat = m_psidat; \
\
  Index_type num_d = m_num_d; \
  Index_type num_z = m_num_z; \
  Index_type num_g = m_num_g; \
  Index_type num_m = m_num_m;

#define LTIMES_BODY \
  phidat[m+ (g * num_m) + (z * num_m * num_g)] += \
    elldat[d+ (m * num_d)] * psidat[d+ (g * num_d) + (z * num_d * num_g)];

#define LTIMES_BODY_RAJA \
  phi(z, g, m) +=  ell(m, d) * psi(z, g, d);


#define LTIMES_VIEWS_RANGES_RAJA \
  using namespace ltimes_idx; \
\
  using PSI_VIEW = RAJA::TypedView<Real_type, \
                                   RAJA::Layout<3, Index_type, 2>, \
                                   IZ, IG, ID>; \
  using ELL_VIEW = RAJA::TypedView<Real_type, \
                                   RAJA::Layout<2, Index_type, 1>, \
                                   IM, ID>; \
  using PHI_VIEW = RAJA::TypedView<Real_type, \
                                   RAJA::Layout<3, Index_type, 2>, \
                                   IZ, IG, IM>; \
\
  PSI_VIEW psi(psidat, \
               RAJA::make_permuted_layout( {{num_z, num_g, num_d}}, \
                     RAJA::as_array<RAJA::Perm<0, 1, 2> >::get() ) ); \
  ELL_VIEW ell(elldat, \
               RAJA::make_permuted_layout( {{num_m, num_d}}, \
                     RAJA::as_array<RAJA::Perm<0, 1> >::get() ) ); \
  PHI_VIEW phi(phidat, \
               RAJA::make_permuted_layout( {{num_z, num_g, num_m}}, \
                     RAJA::as_array<RAJA::Perm<0, 1, 2> >::get() ) ); \
\
      using IDRange = RAJA::TypedRangeSegment<ID>; \
      using IZRange = RAJA::TypedRangeSegment<IZ>; \
      using IGRange = RAJA::TypedRangeSegment<IG>; \
      using IMRange = RAJA::TypedRangeSegment<IM>;


#include "common/KernelBase.hpp"

#include "RAJA/RAJA.hpp"

namespace rajaperf 
{
class RunParams;

namespace apps
{

//
// These index value types cannot be defined in function scope for
// RAJA CUDA variant to work.
//
namespace ltimes_idx {
  RAJA_INDEX_VALUE(ID, "ID");
  RAJA_INDEX_VALUE(IZ, "IZ");
  RAJA_INDEX_VALUE(IG, "IG");
  RAJA_INDEX_VALUE(IM, "IM");
}

class LTIMES : public KernelBase
{
public:

  LTIMES(const RunParams& params);

  ~LTIMES();

  void setUp(VariantID vid);
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);
  void runCudaVariant(VariantID vid);
  void runHipVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Real_ptr m_phidat;
  Real_ptr m_elldat;
  Real_ptr m_psidat;

  Index_type m_num_d_default; 
  Index_type m_num_z_default; 
  Index_type m_num_g_default; 
  Index_type m_num_m_default; 

  Index_type m_num_d; 
  Index_type m_num_z; 
  Index_type m_num_g; 
  Index_type m_num_m; 

  Index_type m_philen;
  Index_type m_elllen;
  Index_type m_psilen;
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
