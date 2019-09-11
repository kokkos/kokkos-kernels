/*
//@HEADER
// ************************************************************************
//
//               KokkosKernels 0.9: Linear Algebra and Graph Kernels
//                 Copyright 2017 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/
#ifndef __KOKKOSKERNELS_STATICCRSGRAPH_HPP__
#define __KOKKOSKERNELS_STATICCRSGRAPH_HPP__


#include "KokkosKernels_Utils.hpp"


namespace KokkosKernelsGraphExample {

template<typename OrdinalType, typename Device, typename MemoryTraits, typename SizeType>
class StaticCrsGraph
{

  public:

    typedef OrdinalType                      data_type;
    typedef Device                           device_type;
    typedef typename Device::execution_space execution_space;
    typedef SizeType                         size_type;

    typedef Kokkos::View<const size_type*, device_type> row_map_type;
    typedef Kokkos::View<data_type*, device_type>       entries_type;

    entries_type entries;
    row_map_type row_map;
    OrdinalType  num_cols;

    //! Construct an empty view.
    StaticCrsGraph() : entries(), row_map(), num_cols() 
    {
    }


    //! Copy constructor (shallow copy).
    StaticCrsGraph(const StaticCrsGraph& rhs) : entries(rhs.entries), row_map(rhs.row_map), num_cols(rhs.num_cols) 
    {
    }


    template<class EntriesType, class RowMapType>
    StaticCrsGraph(const EntriesType& entries_, const RowMapType& row_map_) : entries(entries_), row_map(row_map_)
    {
    }


    template<class EntriesType, class RowMapType>
    StaticCrsGraph(const EntriesType& entries_, const RowMapType& row_map_, OrdinalType numCols_)
        : entries(entries_), row_map(row_map_), num_cols(numCols_)
    {
    }


    /** \brief  Assign to a view of the rhs array.
     *          If the old view is the last view
     *          then allocated memory is deallocated.
     */
    StaticCrsGraph& operator=(const StaticCrsGraph& rhs)
    {
        entries = rhs.entries;
        row_map = rhs.row_map;
        return *this;
    }


    KOKKOS_INLINE_FUNCTION
    data_type numCols() const { return num_cols; }


    /**  \brief  Destroy this view of the array.
     *           If the last view then allocated memory is deallocated.
     */
    ~StaticCrsGraph() 
    {
    }


    KOKKOS_INLINE_FUNCTION
    data_type numRows() const
    {
        return (row_map.extent(0) != 0) ? row_map.extent(0) - static_cast<size_type>(1) : static_cast<size_type>(0);
    }
};

}      // namespace KokkosKernelsGraphExample

#endif  // __KOKKOSKERNELS_STATICCRSGRAPH_HPP__


