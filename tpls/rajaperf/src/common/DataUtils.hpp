//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Methods for kernel data allocation, initialization, and deallocation.
///


#ifndef RAJAPerf_DataUtils_HPP
#define RAJAPerf_DataUtils_HPP

#include "RAJAPerfSuite.hpp"
//#include "RPTypes.hpp"

#include <limits>
#include <new>
#include <type_traits>

#if defined(RAJA_ENABLE_CUDA)
#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"
#endif
#if defined(RAJA_ENABLE_HIP)
#include "RAJA/policy/hip/MemUtils_HIP.hpp"
#endif

namespace rajaperf
{

  
/*!
 * Reset counter for data initialization.
 */
void resetDataInitCount();

/*!
 * Increment counter for data initialization.
 */
void incDataInitCount();


/*!
 * \brief Allocate and initialize Int_type data array.
 * 
 * Array is initialized using method initData(Int_ptr& ptr...) below.
 */
void allocAndInitData(Int_ptr& ptr, int len,
                      VariantID vid = NumVariants);

/*!
 * \brief Allocate and initialize aligned Real_type data array.
 *
 * Array is initialized using method initData(Real_ptr& ptr...) below.
 */
void allocAndInitData(Real_ptr& ptr, int len,
                      VariantID vid = NumVariants);

/*!
 * \brief Allocate and initialize aligned Real_type data array.
 * 
 * Array entries are initialized using the method 
 * initDataConst(Real_ptr& ptr...) below.
 */
void allocAndInitDataConst(Real_ptr& ptr, int len, Real_type val,
                           VariantID vid = NumVariants);

/*!
 * \brief Allocate and initialize aligned Real_type data array with random sign.
 *
 * Array is initialized using method initDataRandSign(Real_ptr& ptr...) below.
 */
void allocAndInitDataRandSign(Real_ptr& ptr, int len,
                              VariantID vid = NumVariants);

/*!
 * \brief Allocate and initialize aligned Real_type data array with random
 *        values.
 *
 * Array is initialized using method initDataRandValue(Real_ptr& ptr...) below.
 */
void allocAndInitDataRandValue(Real_ptr& ptr, int len,
                               VariantID vid = NumVariants);

/*!
 * \brief Allocate and initialize aligned Complex_type data array.
 */
void allocAndInitData(Complex_ptr& ptr, int len,
                      VariantID vid = NumVariants);


/*!
 * \brief Free data arrays.
 */
void deallocData(Int_ptr& ptr);
///
void deallocData(Real_ptr& ptr);
///
void deallocData(Complex_ptr& ptr);


/*!
 * \brief Initialize Int_type data array.
 * 
 * Array entries are randomly initialized to +/-1.
 * Then, two randomly-chosen entries are reset, one to 
 * a value > 1, one to a value < -1.
 */
void initData(Int_ptr& ptr, int len,
              VariantID vid = NumVariants);

/*!
 * \brief Initialize Real_type data array.
 *
 * Array entries are set (non-randomly) to positive values
 * in the interval (0.0, 1.0) based on their array position (index)
 * and the order in which this method is called.
 */
void initData(Real_ptr& ptr, int len,
              VariantID vid = NumVariants);

/*!
 * \brief Initialize Real_type data array.
 *
 * Array entries are set to given constant value.
 */
void initDataConst(Real_ptr& ptr, int len, Real_type val,
                   VariantID vid = NumVariants);

/*!
 * \brief Initialize Real_type data array with random sign.
 * 
 * Array entries are initialized in the same way as the method 
 * initData(Real_ptr& ptr...) above, but with random sign.
 */
void initDataRandSign(Real_ptr& ptr, int len,
                      VariantID vid = NumVariants);

/*!
 * \brief Initialize Real_type data array with random values.
 *
 * Array entries are initialized with random values in the interval [0.0, 1.0].
 */
void initDataRandValue(Real_ptr& ptr, int len,
                       VariantID vid = NumVariants);

/*!
 * \brief Initialize Complex_type data array.
 *
 * Real and imaginary array entries are initialized in the same way as the 
 * method allocAndInitData(Real_ptr& ptr...) above.
 */
void initData(Complex_ptr& ptr, int len,
              VariantID vid = NumVariants);

/*!
 * \brief Initialize Real_type scalar data.
 *
 * Data is set similarly to an array enttry in the method 
 * initData(Real_ptr& ptr...) above.
 */
void initData(Real_type& d,
              VariantID vid = NumVariants);

/*!
 * \brief Calculate and return checksum for data arrays.
 * 
 * Checksums are computed as a weighted sum of array entries,
 * where weight is a simple function of elemtn index.
 *
 * Checksumn is multiplied by given scale factor.
 */
long double calcChecksum(Real_ptr d, int len, 
                         Real_type scale_factor = 1.0);
///
long double calcChecksum(Complex_ptr d, int len,
                         Real_type scale_factor = 1.0);


/*!
 * \brief Holds a RajaPool object and provides access to it via a
 *        std allocator compliant type.
 */
template < typename RajaPool >
struct RAJAPoolAllocatorHolder
{

  /*!
   * \brief Std allocator compliant type that uses the pool owned by
   *        the RAJAPoolAllocatorHolder that it came from.
   *
   * Note that this must not outlive the RAJAPoolAllocatorHolder
   * used to create it.
   */
  template < typename T >
  struct Allocator
  {
    using value_type = T;
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;

    Allocator() = default;

    Allocator(Allocator const&) = default;
    Allocator(Allocator &&) = default;

    Allocator& operator=(Allocator const&) = default;
    Allocator& operator=(Allocator &&) = default;

    template < typename U >
    constexpr Allocator(Allocator<U> const& other) noexcept
      : m_pool_ptr(other.get_impl())
    { }

    Allocator select_on_container_copy_construction()
    {
      return *this;
    }

    /*[[nodiscard]]*/
    value_type* allocate(size_t num)
    {
      if (num > std::numeric_limits<size_t>::max() / sizeof(value_type)) {
        throw std::bad_alloc();
      }

      value_type *ptr = nullptr;
      if (m_pool_ptr != nullptr) {
        ptr = m_pool_ptr->template malloc<value_type>(num);
      }

      if (!ptr) {
        throw std::bad_alloc();
      }

      return ptr;
    }

    void deallocate(value_type* ptr, size_t) noexcept
    {
      if (m_pool_ptr != nullptr) {
        m_pool_ptr->free(ptr);
      }
    }

    RajaPool* const& get_impl() const
    {
      return m_pool_ptr;
    }

    template <typename U>
    friend inline bool operator==(Allocator const& lhs, Allocator<U> const& rhs)
    {
      return lhs.get_impl() == rhs.get_impl();
    }

    template <typename U>
    friend inline bool operator!=(Allocator const& lhs, Allocator<U> const& rhs)
    {
      return !(lhs == rhs);
    }

  private:
    friend RAJAPoolAllocatorHolder;

    RajaPool* m_pool_ptr = nullptr;

    constexpr Allocator(RajaPool* pool_ptr) noexcept
      : m_pool_ptr(pool_ptr)
    { }
  };

  template < typename T >
  Allocator<T> getAllocator()
  {
    return Allocator<T>(&m_pool);
  }

  ~RAJAPoolAllocatorHolder()
  {
    // manually free memory arenas, as this is not done automatically
    m_pool.free_chunks();
  }

private:
  RajaPool m_pool;
};

}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
