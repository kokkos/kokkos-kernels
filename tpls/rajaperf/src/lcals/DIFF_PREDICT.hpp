//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// DIFF_PREDICT kernel reference implementation:
///
/// Index_type offset = iend - ibegin;
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   ar                  = cx[i + offset * 4];
///   br                  = ar - px[i + offset * 4];
///   px[i + offset * 4]  = ar;
///   cr                  = br - px[i + offset * 5];
///   px[i + offset * 5]  = br;
///   ar                  = cr - px[i + offset * 6];
///   px[i + offset * 6]  = cr;
///   br                  = ar - px[i + offset * 7];
///   px[i + offset * 7]  = ar;
///   cr                  = br - px[i + offset * 8];
///   px[i + offset * 8]  = br;
///   ar                  = cr - px[i + offset * 9];
///   px[i + offset * 9]  = cr;
///   br                  = ar - px[i + offset * 10];
///   px[i + offset * 10] = ar;
///   cr                  = br - px[i + offset * 11];
///   px[i + offset * 11] = br;
///   px[i + offset * 13] = cr - px[i + offset * 12];
///   px[i + offset * 12] = cr;
/// }
///

#ifndef RAJAPerf_Lcals_DIFF_PREDICT_HPP
#define RAJAPerf_Lcals_DIFF_PREDICT_HPP


#define DIFF_PREDICT_DATA_SETUP \
  Real_ptr px = m_px; \
  Real_ptr cx = m_cx; \
  const Index_type offset = m_offset;

#define DIFF_PREDICT_BODY  \
  Real_type ar, br, cr; \
\
  ar                  = cx[i + offset * 4];       \
  br                  = ar - px[i + offset * 4];  \
  px[i + offset * 4]  = ar;                       \
  cr                  = br - px[i + offset * 5];  \
  px[i + offset * 5]  = br;                       \
  ar                  = cr - px[i + offset * 6];  \
  px[i + offset * 6]  = cr;                       \
  br                  = ar - px[i + offset * 7];  \
  px[i + offset * 7]  = ar;                       \
  cr                  = br - px[i + offset * 8];  \
  px[i + offset * 8]  = br;                       \
  ar                  = cr - px[i + offset * 9];  \
  px[i + offset * 9]  = cr;                       \
  br                  = ar - px[i + offset * 10]; \
  px[i + offset * 10] = ar;                       \
  cr                  = br - px[i + offset * 11]; \
  px[i + offset * 11] = br;                       \
  px[i + offset * 13] = cr - px[i + offset * 12]; \
  px[i + offset * 12] = cr;


#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace lcals
{

class DIFF_PREDICT : public KernelBase
{
public:

  DIFF_PREDICT(const RunParams& params);

  ~DIFF_PREDICT();

  void setUp(VariantID vid);
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);
  void runCudaVariant(VariantID vid);
  void runHipVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Real_ptr m_px;
  Real_ptr m_cx;

  Index_type m_array_length;
  Index_type m_offset;
};

} // end namespace lcals
} // end namespace rajaperf

#endif // closing endif for header file include guard
