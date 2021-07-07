//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HALOEXCHANGE.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

#include <cmath>

namespace rajaperf
{
namespace apps
{

namespace {

void create_pack_lists(std::vector<Int_ptr>& pack_index_lists,
                       std::vector<Index_type >& pack_index_list_lengths,
                       const Index_type halo_width, const Index_type* grid_dims,
                       const Index_type num_neighbors,
                       VariantID vid);
void create_unpack_lists(std::vector<Int_ptr>& unpack_index_lists,
                         std::vector<Index_type >& unpack_index_list_lengths,
                         const Index_type halo_width, const Index_type* grid_dims,
                         const Index_type num_neighbors,
                         VariantID vid);
void destroy_pack_lists(std::vector<Int_ptr>& pack_index_lists,
                        const Index_type num_neighbors,
                        VariantID vid);
void destroy_unpack_lists(std::vector<Int_ptr>& unpack_index_lists,
                          const Index_type num_neighbors,
                          VariantID vid);

}


HALOEXCHANGE::HALOEXCHANGE(const RunParams& params)
  : KernelBase(rajaperf::Apps_HALOEXCHANGE, params)
{
  m_grid_dims_default[0] = 100;
  m_grid_dims_default[1] = 100;
  m_grid_dims_default[2] = 100;
  m_halo_width_default   = 1;
  m_num_vars_default     = 3;

  setDefaultSize((m_grid_dims_default[0] + 2*m_halo_width_default) *
                 (m_grid_dims_default[1] + 2*m_halo_width_default) *
                 (m_grid_dims_default[2] + 2*m_halo_width_default) *
                 m_num_vars_default);
  setDefaultReps(50);

  setVariantDefined( Base_Seq );
  setVariantDefined( Lambda_Seq );
  setVariantDefined( RAJA_Seq );

  setVariantDefined( Base_OpenMP );
  setVariantDefined( Lambda_OpenMP );
  setVariantDefined( RAJA_OpenMP );

  setVariantDefined( Base_OpenMPTarget );
  setVariantDefined( RAJA_OpenMPTarget );

  setVariantDefined( Base_CUDA );
  setVariantDefined( RAJA_CUDA );
  setVariantDefined( RAJA_WORKGROUP_CUDA );

  setVariantDefined( Base_HIP );
  setVariantDefined( RAJA_HIP );
#ifdef RAJA_ENABLE_HIP_INDIRECT_FUNCTION_CALL
  setVariantDefined( RAJA_WORKGROUP_HIP );
#endif
}

HALOEXCHANGE::~HALOEXCHANGE()
{
}

void HALOEXCHANGE::setUp(VariantID vid)
{
  double cbrt_size_fact = std::cbrt(run_params.getSizeFactor());

  m_grid_dims[0] = cbrt_size_fact * m_grid_dims_default[0];
  m_grid_dims[1] = cbrt_size_fact * m_grid_dims_default[1];
  m_grid_dims[2] = cbrt_size_fact * m_grid_dims_default[2];
  m_halo_width = m_halo_width_default;
  m_num_vars   = m_num_vars_default;

  m_grid_plus_halo_dims[0] = m_grid_dims[0] + 2*m_halo_width;
  m_grid_plus_halo_dims[1] = m_grid_dims[1] + 2*m_halo_width;
  m_grid_plus_halo_dims[2] = m_grid_dims[2] + 2*m_halo_width;
  m_var_size = m_grid_plus_halo_dims[0] *
               m_grid_plus_halo_dims[1] *
               m_grid_plus_halo_dims[2] ;

  m_vars.resize(m_num_vars, nullptr);
  for (Index_type v = 0; v < m_num_vars; ++v) {
    allocAndInitData(m_vars[v], m_var_size, vid);

    Real_ptr var = m_vars[v];

    for (Index_type i = 0; i < m_var_size; i++) {
      var[i] = i + v;
    }
  }

  m_pack_index_lists.resize(s_num_neighbors, nullptr);
  m_pack_index_list_lengths.resize(s_num_neighbors, 0);
  create_pack_lists(m_pack_index_lists, m_pack_index_list_lengths, m_halo_width, m_grid_dims, s_num_neighbors, vid);

  m_unpack_index_lists.resize(s_num_neighbors, nullptr);
  m_unpack_index_list_lengths.resize(s_num_neighbors, 0);
  create_unpack_lists(m_unpack_index_lists, m_unpack_index_list_lengths, m_halo_width, m_grid_dims, s_num_neighbors, vid);

  m_buffers.resize(s_num_neighbors, nullptr);
  for (Index_type l = 0; l < s_num_neighbors; ++l) {
    Index_type buffer_len = m_num_vars * m_pack_index_list_lengths[l];
    allocAndInitData(m_buffers[l], buffer_len, vid);
  }
}

void HALOEXCHANGE::updateChecksum(VariantID vid)
{
  for (Real_ptr var : m_vars) {
    checksum[vid] += calcChecksum(var, m_var_size);
  }
}

void HALOEXCHANGE::tearDown(VariantID vid)
{
  for (int l = 0; l < s_num_neighbors; ++l) {
    deallocData(m_buffers[l]);
  }
  m_buffers.clear();

  destroy_unpack_lists(m_unpack_index_lists, s_num_neighbors, vid);
  m_unpack_index_list_lengths.clear();
  m_unpack_index_lists.clear();

  destroy_pack_lists(m_pack_index_lists, s_num_neighbors, vid);
  m_pack_index_list_lengths.clear();
  m_pack_index_lists.clear();

  for (int v = 0; v < m_num_vars; ++v) {
    deallocData(m_vars[v]);
  }
  m_vars.clear();
}

namespace {

struct Extent
{
  Index_type i_min;
  Index_type i_max;
  Index_type j_min;
  Index_type j_max;
  Index_type k_min;
  Index_type k_max;
};

//
// Function to generate index lists for packing.
//
void create_pack_lists(std::vector<Int_ptr>& pack_index_lists,
                       std::vector<Index_type >& pack_index_list_lengths,
                       const Index_type halo_width, const Index_type* grid_dims,
                       const Index_type num_neighbors,
                       VariantID vid)
{
  std::vector<Extent> pack_index_list_extents(num_neighbors);

  // faces
  pack_index_list_extents[0]  = Extent{halo_width  , halo_width   + halo_width,
                                       halo_width  , grid_dims[1] + halo_width,
                                       halo_width  , grid_dims[2] + halo_width};
  pack_index_list_extents[1]  = Extent{grid_dims[0], grid_dims[0] + halo_width,
                                       halo_width  , grid_dims[1] + halo_width,
                                       halo_width  , grid_dims[2] + halo_width};
  pack_index_list_extents[2]  = Extent{halo_width  , grid_dims[0] + halo_width,
                                       halo_width  , halo_width   + halo_width,
                                       halo_width  , grid_dims[2] + halo_width};
  pack_index_list_extents[3]  = Extent{halo_width  , grid_dims[0] + halo_width,
                                       grid_dims[1], grid_dims[1] + halo_width,
                                       halo_width  , grid_dims[2] + halo_width};
  pack_index_list_extents[4]  = Extent{halo_width  , grid_dims[0] + halo_width,
                                       halo_width  , grid_dims[1] + halo_width,
                                       halo_width  , halo_width   + halo_width};
  pack_index_list_extents[5]  = Extent{halo_width  , grid_dims[0] + halo_width,
                                       halo_width  , grid_dims[1] + halo_width,
                                       grid_dims[2], grid_dims[2] + halo_width};

  // edges
  pack_index_list_extents[6]  = Extent{halo_width  , halo_width   + halo_width,
                                       halo_width  , halo_width   + halo_width,
                                       halo_width  , grid_dims[2] + halo_width};
  pack_index_list_extents[7]  = Extent{halo_width  , halo_width   + halo_width,
                                       grid_dims[1], grid_dims[1] + halo_width,
                                       halo_width  , grid_dims[2] + halo_width};
  pack_index_list_extents[8]  = Extent{grid_dims[0], grid_dims[0] + halo_width,
                                       halo_width  , halo_width   + halo_width,
                                       halo_width  , grid_dims[2] + halo_width};
  pack_index_list_extents[9]  = Extent{grid_dims[0], grid_dims[0] + halo_width,
                                       grid_dims[1], grid_dims[1] + halo_width,
                                       halo_width  , grid_dims[2] + halo_width};
  pack_index_list_extents[10] = Extent{halo_width  , halo_width   + halo_width,
                                       halo_width  , grid_dims[1] + halo_width,
                                       halo_width  , halo_width   + halo_width};
  pack_index_list_extents[11] = Extent{halo_width  , halo_width   + halo_width,
                                       halo_width  , grid_dims[1] + halo_width,
                                       grid_dims[2], grid_dims[2] + halo_width};
  pack_index_list_extents[12] = Extent{grid_dims[0], grid_dims[0] + halo_width,
                                       halo_width  , grid_dims[1] + halo_width,
                                       halo_width  , halo_width   + halo_width};
  pack_index_list_extents[13] = Extent{grid_dims[0], grid_dims[0] + halo_width,
                                       halo_width  , grid_dims[1] + halo_width,
                                       grid_dims[2], grid_dims[2] + halo_width};
  pack_index_list_extents[14] = Extent{halo_width  , grid_dims[0] + halo_width,
                                       halo_width  , halo_width   + halo_width,
                                       halo_width  , halo_width   + halo_width};
  pack_index_list_extents[15] = Extent{halo_width  , grid_dims[0] + halo_width,
                                       halo_width  , halo_width   + halo_width,
                                       grid_dims[2], grid_dims[2] + halo_width};
  pack_index_list_extents[16] = Extent{halo_width  , grid_dims[0] + halo_width,
                                       grid_dims[1], grid_dims[1] + halo_width,
                                       halo_width  , halo_width   + halo_width};
  pack_index_list_extents[17] = Extent{halo_width  , grid_dims[0] + halo_width,
                                       grid_dims[1], grid_dims[1] + halo_width,
                                       grid_dims[2], grid_dims[2] + halo_width};

  // corners
  pack_index_list_extents[18] = Extent{halo_width  , halo_width   + halo_width,
                                       halo_width  , halo_width   + halo_width,
                                       halo_width  , halo_width   + halo_width};
  pack_index_list_extents[19] = Extent{halo_width  , halo_width   + halo_width,
                                       halo_width  , halo_width   + halo_width,
                                       grid_dims[2], grid_dims[2] + halo_width};
  pack_index_list_extents[20] = Extent{halo_width  , halo_width   + halo_width,
                                       grid_dims[1], grid_dims[1] + halo_width,
                                       halo_width  , halo_width   + halo_width};
  pack_index_list_extents[21] = Extent{halo_width  , halo_width   + halo_width,
                                       grid_dims[1], grid_dims[1] + halo_width,
                                       grid_dims[2], grid_dims[2] + halo_width};
  pack_index_list_extents[22] = Extent{grid_dims[0], grid_dims[0] + halo_width,
                                       halo_width  , halo_width   + halo_width,
                                       halo_width  , halo_width   + halo_width};
  pack_index_list_extents[23] = Extent{grid_dims[0], grid_dims[0] + halo_width,
                                       halo_width  , halo_width   + halo_width,
                                       grid_dims[2], grid_dims[2] + halo_width};
  pack_index_list_extents[24] = Extent{grid_dims[0], grid_dims[0] + halo_width,
                                       grid_dims[1], grid_dims[1] + halo_width,
                                       halo_width  , halo_width   + halo_width};
  pack_index_list_extents[25] = Extent{grid_dims[0], grid_dims[0] + halo_width,
                                       grid_dims[1], grid_dims[1] + halo_width,
                                       grid_dims[2], grid_dims[2] + halo_width};

  const Index_type grid_i_stride = 1;
  const Index_type grid_j_stride = grid_dims[0] + 2*halo_width;
  const Index_type grid_k_stride = grid_j_stride * (grid_dims[1] + 2*halo_width);

  for (Index_type l = 0; l < num_neighbors; ++l) {

    Extent extent = pack_index_list_extents[l];

    pack_index_list_lengths[l] = (extent.i_max - extent.i_min) *
                                 (extent.j_max - extent.j_min) *
                                 (extent.k_max - extent.k_min) ;

    allocAndInitData(pack_index_lists[l], pack_index_list_lengths[l], vid);

    Int_ptr pack_list = pack_index_lists[l];

    Index_type list_idx = 0;
    for (Index_type kk = extent.k_min; kk < extent.k_max; ++kk) {
      for (Index_type jj = extent.j_min; jj < extent.j_max; ++jj) {
        for (Index_type ii = extent.i_min; ii < extent.i_max; ++ii) {

          Index_type pack_idx = ii * grid_i_stride +
                         jj * grid_j_stride +
                         kk * grid_k_stride ;

          pack_list[list_idx] = pack_idx;

          list_idx += 1;
        }
      }
    }
  }
}

//
// Function to destroy packing index lists.
//
void destroy_pack_lists(std::vector<Int_ptr>& pack_index_lists,
                       const Index_type num_neighbors,
                       VariantID vid)
{
  (void) vid;

  for (Index_type l = 0; l < num_neighbors; ++l) {
    deallocData(pack_index_lists[l]);
  }
}

//
// Function to generate index lists for unpacking.
//
void create_unpack_lists(std::vector<Int_ptr>& unpack_index_lists,
                         std::vector<Index_type >& unpack_index_list_lengths,
                         const Index_type halo_width, const Index_type* grid_dims,
                         const Index_type num_neighbors,
                         VariantID vid)
{
  std::vector<Extent> unpack_index_list_extents(num_neighbors);

  // faces
  unpack_index_list_extents[0]  = Extent{0                        ,                  halo_width,
                                         halo_width               , grid_dims[1] +   halo_width,
                                         halo_width               , grid_dims[2] +   halo_width};
  unpack_index_list_extents[1]  = Extent{grid_dims[0] + halo_width, grid_dims[0] + 2*halo_width,
                                         halo_width               , grid_dims[1] +   halo_width,
                                         halo_width               , grid_dims[2] +   halo_width};
  unpack_index_list_extents[2]  = Extent{halo_width               , grid_dims[0] +   halo_width,
                                         0                        ,                  halo_width,
                                         halo_width               , grid_dims[2] +   halo_width};
  unpack_index_list_extents[3]  = Extent{halo_width               , grid_dims[0] +   halo_width,
                                         grid_dims[1] + halo_width, grid_dims[1] + 2*halo_width,
                                         halo_width               , grid_dims[2] +   halo_width};
  unpack_index_list_extents[4]  = Extent{halo_width               , grid_dims[0] +   halo_width,
                                         halo_width               , grid_dims[1] +   halo_width,
                                         0                        ,                  halo_width};
  unpack_index_list_extents[5]  = Extent{halo_width               , grid_dims[0] +   halo_width,
                                         halo_width               , grid_dims[1] +   halo_width,
                                         grid_dims[2] + halo_width, grid_dims[2] + 2*halo_width};

  // edges
  unpack_index_list_extents[6]  = Extent{0                        ,                  halo_width,
                                         0                        ,                  halo_width,
                                         halo_width               , grid_dims[2] +   halo_width};
  unpack_index_list_extents[7]  = Extent{0                        ,                  halo_width,
                                         grid_dims[1] + halo_width, grid_dims[1] + 2*halo_width,
                                         halo_width               , grid_dims[2] +   halo_width};
  unpack_index_list_extents[8]  = Extent{grid_dims[0] + halo_width, grid_dims[0] + 2*halo_width,
                                         0                        ,                  halo_width,
                                         halo_width               , grid_dims[2] +   halo_width};
  unpack_index_list_extents[9]  = Extent{grid_dims[0] + halo_width, grid_dims[0] + 2*halo_width,
                                         grid_dims[1] + halo_width, grid_dims[1] + 2*halo_width,
                                         halo_width               , grid_dims[2] +   halo_width};
  unpack_index_list_extents[10] = Extent{0                        ,                  halo_width,
                                         halo_width               , grid_dims[1] +   halo_width,
                                         0                        ,                  halo_width};
  unpack_index_list_extents[11] = Extent{0                        ,                  halo_width,
                                         halo_width               , grid_dims[1] +   halo_width,
                                         grid_dims[2] + halo_width, grid_dims[2] + 2*halo_width};
  unpack_index_list_extents[12] = Extent{grid_dims[0] + halo_width, grid_dims[0] + 2*halo_width,
                                         halo_width               , grid_dims[1] +   halo_width,
                                         0                        ,                  halo_width};
  unpack_index_list_extents[13] = Extent{grid_dims[0] + halo_width, grid_dims[0] + 2*halo_width,
                                         halo_width               , grid_dims[1] +   halo_width,
                                         grid_dims[2] + halo_width, grid_dims[2] + 2*halo_width};
  unpack_index_list_extents[14] = Extent{halo_width               , grid_dims[0] +   halo_width,
                                         0                        ,                  halo_width,
                                         0                        ,                  halo_width};
  unpack_index_list_extents[15] = Extent{halo_width               , grid_dims[0] +   halo_width,
                                         0                        ,                  halo_width,
                                         grid_dims[2] + halo_width, grid_dims[2] + 2*halo_width};
  unpack_index_list_extents[16] = Extent{halo_width               , grid_dims[0] +   halo_width,
                                         grid_dims[1] + halo_width, grid_dims[1] + 2*halo_width,
                                         0                        ,                  halo_width};
  unpack_index_list_extents[17] = Extent{halo_width               , grid_dims[0] +   halo_width,
                                         grid_dims[1] + halo_width, grid_dims[1] + 2*halo_width,
                                         grid_dims[2] + halo_width, grid_dims[2] + 2*halo_width};

  // corners
  unpack_index_list_extents[18] = Extent{0                        ,                  halo_width,
                                         0                        ,                  halo_width,
                                         0                        ,                  halo_width};
  unpack_index_list_extents[19] = Extent{0                        ,                  halo_width,
                                         0                        ,                  halo_width,
                                         grid_dims[2] + halo_width, grid_dims[2] + 2*halo_width};
  unpack_index_list_extents[20] = Extent{0                        ,                  halo_width,
                                         grid_dims[1] + halo_width, grid_dims[1] + 2*halo_width,
                                         0                        ,                  halo_width};
  unpack_index_list_extents[21] = Extent{0                        ,                  halo_width,
                                         grid_dims[1] + halo_width, grid_dims[1] + 2*halo_width,
                                         grid_dims[2] + halo_width, grid_dims[2] + 2*halo_width};
  unpack_index_list_extents[22] = Extent{grid_dims[0] + halo_width, grid_dims[0] + 2*halo_width,
                                         0                        ,                  halo_width,
                                         0                        ,                  halo_width};
  unpack_index_list_extents[23] = Extent{grid_dims[0] + halo_width, grid_dims[0] + 2*halo_width,
                                         0                        ,                  halo_width,
                                         grid_dims[2] + halo_width, grid_dims[2] + 2*halo_width};
  unpack_index_list_extents[24] = Extent{grid_dims[0] + halo_width, grid_dims[0] + 2*halo_width,
                                         grid_dims[1] + halo_width, grid_dims[1] + 2*halo_width,
                                         0                        ,                  halo_width};
  unpack_index_list_extents[25] = Extent{grid_dims[0] + halo_width, grid_dims[0] + 2*halo_width,
                                         grid_dims[1] + halo_width, grid_dims[1] + 2*halo_width,
                                         grid_dims[2] + halo_width, grid_dims[2] + 2*halo_width};

  const Index_type grid_i_stride = 1;
  const Index_type grid_j_stride = grid_dims[0] + 2*halo_width;
  const Index_type grid_k_stride = grid_j_stride * (grid_dims[1] + 2*halo_width);

  for (Index_type l = 0; l < num_neighbors; ++l) {

    Extent extent = unpack_index_list_extents[l];

    unpack_index_list_lengths[l] = (extent.i_max - extent.i_min) *
                                   (extent.j_max - extent.j_min) *
                                   (extent.k_max - extent.k_min) ;

    allocAndInitData(unpack_index_lists[l], unpack_index_list_lengths[l], vid);

    Int_ptr unpack_list = unpack_index_lists[l];

    Index_type list_idx = 0;
    for (Index_type kk = extent.k_min; kk < extent.k_max; ++kk) {
      for (Index_type jj = extent.j_min; jj < extent.j_max; ++jj) {
        for (Index_type ii = extent.i_min; ii < extent.i_max; ++ii) {

          Index_type unpack_idx = ii * grid_i_stride +
                           jj * grid_j_stride +
                           kk * grid_k_stride ;

          unpack_list[list_idx] = unpack_idx;

          list_idx += 1;
        }
      }
    }
  }
}

//
// Function to destroy unpacking index lists.
//
void destroy_unpack_lists(std::vector<Int_ptr>& unpack_index_lists,
                          const Index_type num_neighbors,
                          VariantID vid)
{
  (void) vid;

  for (Index_type l = 0; l < num_neighbors; ++l) {
    deallocData(unpack_index_lists[l]);
  }
}

} // end namespace

} // end namespace apps
} // end namespace rajaperf
