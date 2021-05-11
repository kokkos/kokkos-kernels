//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// INT_PREDICT kernel reference implementation:
///
/// Index_type offset = iend - ibegin;
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   px[i] = dm28*px[i + offset * 12] + dm27*px[i + offset * 11] +
///           dm26*px[i + offset * 10] + dm25*px[i + offset *  9] +
///           dm24*px[i + offset *  8] + dm23*px[i + offset *  7] +
///           dm22*px[i + offset *  6] +
///           c0*( px[i + offset *  4] + px[i + offset *  5] ) +
///           px[i + offset *  2];
/// }
///

#ifndef RAJAPerf_Lcals_INT_PREDICT_HPP
#define RAJAPerf_Lcals_INT_PREDICT_HPP


#define INT_PREDICT_DATA_SETUP \
  Real_ptr px = m_px; \
  Real_type dm22 = m_dm22; \
  Real_type dm23 = m_dm23; \
  Real_type dm24 = m_dm24; \
  Real_type dm25 = m_dm25; \
  Real_type dm26 = m_dm26; \
  Real_type dm27 = m_dm27; \
  Real_type dm28 = m_dm28; \
  Real_type c0 = m_c0; \
  const Index_type offset = m_offset;

#define INT_PREDICT_BODY  \
  px[i] = dm28*px[i + offset * 12] + dm27*px[i + offset * 11] + \
          dm26*px[i + offset * 10] + dm25*px[i + offset *  9] + \
          dm24*px[i + offset *  8] + dm23*px[i + offset *  7] + \
          dm22*px[i + offset *  6] + \
          c0*( px[i + offset *  4] + px[i + offset *  5] ) + \
          px[i + offset *  2];


#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace lcals
{

class INT_PREDICT : public KernelBase
{
public:

  INT_PREDICT(const RunParams& params);

  ~INT_PREDICT();

  void setUp(VariantID vid);
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);
  void runCudaVariant(VariantID vid);
  void runHipVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Index_type m_array_length;
  Index_type m_offset;

  Real_ptr m_px;
  Real_type m_px_initval;

  Real_type m_dm22;
  Real_type m_dm23;
  Real_type m_dm24;
  Real_type m_dm25;
  Real_type m_dm26;
  Real_type m_dm27;
  Real_type m_dm28;
  Real_type m_c0;
};

} // end namespace lcals
} // end namespace rajaperf

#endif // closing endif for header file include guard
