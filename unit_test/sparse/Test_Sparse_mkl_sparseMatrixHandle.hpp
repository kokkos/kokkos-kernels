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

//#include "Teuchos_UnitTestHarness.hpp"
#include "Kokkos_Sparse_impl_MKL.hpp"
//#include "Teuchos_TypeNameTraits.hpp"
#include "Kokkos_ArithTraits.hpp"
#include <memory>
#include<stdexcept>
#include <iostream>

#ifndef kokkos_complex_double
#define kokkos_complex_double Kokkos::complex<double>
#define kokkos_complex_float Kokkos::complex<float>
#endif

namespace KokkosKernels{

template <typename T>
struct TypeNameTraits
{
    static const char* name()
    {
        return typeid(T).name();
    }
};
}


namespace { // (anonymous)

/** \brief Test that a chunk of code does not throw any exceptions.
 *
 * This macro is not complicated so take a look for yourself!
 *
 * \ingroup teuchos_testing_grp
 */
#define KK_TEST_THROW( code, ExceptType, out, success  ) \
  { \
    std::ostream& l_out = (out); \
    try { \
      l_out << "Test that code {"#code";} throws " \
            << KokkosKernels::TypeNameTraits<ExceptType>::name () << ": "; \
      code; \
      (success) = false; \
      l_out << "failed (code did not throw an exception at all)\n"; \
    } \
    catch (const ExceptType& except) { \
      l_out << "passed\n";                                        \
      l_out << "\nException message for expected exception:\n\n";   \
      { \
        l_out << except.what () << "\n\n"; \
      } \
    } \
    catch (std::exception& except) { \
      l_out << "The code was supposed to throw an exception of type "   \
            << KokkosKernels::TypeNameTraits<ExceptType>::name () << ", but " \
            << "instead threw an exception of type " \
            << typeid (except).name () << ", which is a subclass of " \
            << "std::exception.  The exception's message is:\n\n"; \
      { \
        l_out << except.what () << "\n\n"; \
      } \
      l_out << "failed\n"; \
    } \
    catch (...) { \
      l_out << "The code was supposed to throw an exception of type "   \
            << KokkosKernels::TypeNameTraits<ExceptType>::name () << ", but " \
            << "instead threw an exception of some unknown type, which is " \
            << "not a subclass of std::exception.  This means we cannot " \
            << "show you the exception's message, if it even has one.\n\n"; \
      l_out << "failed\n"; \
    } \
  }

#define KK_TEST_NOTHROW( code, out, success  ) \
  { \
    std::ostream& l_out = (out); \
    try { \
      l_out << "Test that code {"#code";} does not throw : "; \
      code; \
      l_out << "passed\n"; \
    } \
    catch (std::exception& except) { \
      (success) = false; \
      l_out << "The code was not supposed to throw an exception, but " \
            << "instead threw an exception of type " \
            << typeid (except).name () << ", which is a subclass of " \
            << "std::exception.  The exception's message is:\n\n"; \
      { \
        l_out << except.what () << "\n\n"; \
      } \
      l_out << "failed\n"; \
    } \
    catch (...) { \
      (success) = false; \
      l_out << "The code was not supposed to throw an exception, but " \
            << "instead threw an exception of some unknown type, which is " \
            << "not a subclass of std::exception.  This means we cannot " \
            << "show you the exception's message, if it even has one.\n\n"; \
      l_out << "failed\n"; \
    } \
  }


//using Teuchos::TypeNameTraits;
using std::endl;
typedef typename ::Kokkos::View<int*>::HostMirror::execution_space host_execution_space;
typedef typename ::Kokkos::Device<host_execution_space, Kokkos::HostSpace> host_device_type;

template<class ScalarType, class LocalOrdinalType, class OffsetType>
struct TestMklSparseMatrixHandle {
  typedef typename Kokkos::Details::ArithTraits<ScalarType>::val_type val_type;
  typedef typename Kokkos::Details::ArithTraits<val_type>::mag_type mag_type;
  typedef KokkosSparse::CrsMatrix<val_type, LocalOrdinalType, host_device_type,
                                  void, OffsetType> matrix_type;
  typedef typename ::KokkosSparse::Impl::Mkl::WrappedTplMatrixHandle<matrix_type> handle_type;

  static void
  makeTestMatrixArrays (bool& success,
                        //Teuchos::FancyOStream& out,
                        std::ostream &out,
                        Kokkos::View<OffsetType*, host_device_type>& ptr,
                        Kokkos::View<LocalOrdinalType*, host_device_type>& ind,
                        Kokkos::View<val_type*, host_device_type>& val,
                        const LocalOrdinalType numRows,
                        const LocalOrdinalType numCols)
  {
    out << "Make test matrix:" << endl;
    //Teuchos::OSTab tab0 (out);
    out << "ScalarType: " << KokkosKernels::TypeNameTraits<ScalarType>::name () << endl
        << "LocalOrdinalType: " << KokkosKernels::TypeNameTraits<LocalOrdinalType>::name () << endl
        << "OffsetType: " << KokkosKernels::TypeNameTraits<OffsetType>::name () << endl;

    const OffsetType numEnt = (numRows <= 2) ? (2*numRows) : (3*(numRows - 2) + 4);
    out << "numRows: " << numRows << ", numEnt: " << numEnt << endl;

    ptr = Kokkos::View<OffsetType*, host_device_type> ("ptr", numRows + 1);
    ind = Kokkos::View<LocalOrdinalType*, host_device_type> ("ind", numEnt);
    val = Kokkos::View<val_type*, host_device_type> ("val", numEnt);

    OffsetType curPos = 0;
    ptr[0] = curPos;
    for (LocalOrdinalType row = 0; row < numRows; ++row) {
      const LocalOrdinalType col0 = (row - 2) % numCols;
      const LocalOrdinalType col1 = row % numCols;
      const LocalOrdinalType col2 = (row + 2) % numCols;
      const val_type val0 = static_cast<val_type> (static_cast<mag_type> (col0));
      const val_type val1 = static_cast<val_type> (static_cast<mag_type> (col1));
      const val_type val2 = static_cast<val_type> (static_cast<mag_type> (col2));

      //out << " row: " << row << endl;

      if (row == 0) { // 2 entries
        //out << "   - curPos: " << curPos << endl;
        ind[curPos] = col1;
        val[curPos] = val1;
        ++curPos;
        //out << "   - curPos: " << curPos << endl;
        ind[curPos] = col2;
        val[curPos] = val2;
        ++curPos;
      }
      else if (row + 1 == numRows) { // 2 entries
        //out << "   - curPos: " << curPos << endl;
        ind[curPos] = col0;
        val[curPos] = val0;
        ++curPos;
        //out << "   - curPos: " << curPos << endl;
        ind[curPos] = col1;
        val[curPos] = val1;
        ++curPos;
      }
      else { // 3 entries
        //out << "   - curPos: " << curPos << endl;
        ind[curPos] = col0;
        val[curPos] = val0;
        ++curPos;
        //out << "   - curPos: " << curPos << endl;
        ind[curPos] = col1;
        val[curPos] = val1;
        ++curPos;
        //out << "   - curPos: " << curPos << endl;
        ind[curPos] = col2;
        val[curPos] = val2;
        ++curPos;
      }
      ptr[row+1] = curPos;
    }
    out << "Done!" << endl;
  }

  // Create a test matrix, and attempt to wrap it in an MKL TPL handle.
  static std::shared_ptr<handle_type>
  makeHandle (bool& success,
              //Teuchos::FancyOStream& out,
			  std::ostream &out,
              const LocalOrdinalType numRows,
              const LocalOrdinalType numCols)
  {
    Kokkos::View<OffsetType*, host_device_type> ptr;
    Kokkos::View<LocalOrdinalType*, host_device_type> ind;
    Kokkos::View<val_type*, host_device_type> val;

    makeTestMatrixArrays (success, out, ptr, ind, val, numRows, numCols);

    out << "Make KokkosSparse::CrsMatrix" << endl;
    matrix_type A ("A", numRows, numCols, val.dimension_0 (), val, ptr, ind);

    out << "Attempt to make MKL handle" << endl;
    std::shared_ptr<handle_type> handle;
#ifdef HAVE_KOKKOSKERNELS_MKL
    //TEST_NOTHROW( handle = std::shared_ptr<handle_type> (new handle_type (A, false)) );
    KK_TEST_NOTHROW( handle = std::shared_ptr<handle_type> (new handle_type (A, false)), out, success );
    const bool l_result = handle.get () != NULL;
    //TEST_ASSERT( handle.get () != NULL );
    EXPECT_TRUE( l_result );
#else
    //TEST_THROW( handle = std::shared_ptr<handle_type> (new handle_type (A, false)), std::runtime_error );
    KK_TEST_THROW(  handle = std::shared_ptr<handle_type> (new handle_type (A, false)), std::runtime_error , out, success);
    //TEST_ASSERT( handle.get () == NULL );
    const bool l_result = handle.get () != NULL;
    EXPECT_TRUE( !l_result );
#endif // HAVE_KOKKOSKERNELS_MKL
    return handle;
  }
};

template<class ScalarType, class LocalOrdinalType, class OffsetType>
void
testMklSparseMatrixHandleOneCase (bool& success,
                                  //Teuchos::FancyOStream& out,
								  std::ostream &out,
                                  const LocalOrdinalType numRows,
                                  const LocalOrdinalType numCols)
{
  typedef TestMklSparseMatrixHandle<ScalarType, LocalOrdinalType, OffsetType> tester_type;
  (void) tester_type::makeHandle (success, out, numRows, numCols);
}

template<class LocalOrdinalType, class OffsetType, class scalar_type>
void
testAllScalars (bool& success,
				        std::ostream &out,
                const LocalOrdinalType numRows,
                const LocalOrdinalType numCols)
{
  {

    testMklSparseMatrixHandleOneCase<scalar_type, LocalOrdinalType, OffsetType> (success, out, numRows, numCols);

    typedef typename Kokkos::Details::ArithTraits<scalar_type>::val_type value_type;
    typedef typename ::KokkosSparse::Impl::Mkl::RawTplMatrixHandle<value_type> converter_type;
    static_assert (std::is_same<typename converter_type::value_type, value_type>::value,
                   "RawTplMatrixHandle<double>::value_type != double");
#ifdef HAVE_KOKKOSKERNELS_MKL
    static_assert (std::is_same<typename converter_type::internal_value_type, double>::value,
                   "RawTplMatrixHandle<double>::interval_value_type != double");
#endif // HAVE_KOKKOSKERNELS_MKL

    const value_type x_our (3.0);
    const auto x_mkl = converter_type::convertToInternalValue (x_our);
    static_assert (std::is_same<typename std::decay<decltype (x_mkl) >::type, typename converter_type::internal_value_type>::value,
                   "RawTplMatrixHandle<double>::convertToInternalValue returns the wrong type");
    const auto x_back = converter_type::convertFromInternalValue (x_mkl);
    static_assert (std::is_same<typename std::decay<decltype (x_back) >::type, typename converter_type::value_type>::value,
                   "RawTplMatrixHandle<double>::convertFromInternalValue returns the wrong type");
    //TEST_EQUALITY( x_back, x_our );
    EXPECT_TRUE( (x_back == x_our ));

  }
}

template<class LO, class OffsetType, class scalar_type>
void testAllScalarsAndLocalOrdinals (bool& success,std::ostream &out)
{
  {
    out << "Test LocalOrdinalType=int" << endl;
    //Teuchos::OSTab tab0 (out);
    {
      const LO numRows = 30;
      const LO numCols = 15;
      out << "Test numRows=" << numRows << ", numCols=" << numCols << endl;
      //Teuchos::OSTab tab1 (out);
      testAllScalars<LO, OffsetType, scalar_type> (success, out, numRows, numCols);
    }
    {
      const LO numRows = 1;
      const LO numCols = 3;
      out << "Test numRows=" << numRows << ", numCols=" << numCols << endl;
      //Teuchos::OSTab tab1 (out);
      testAllScalars<LO, OffsetType, scalar_type> (success, out, numRows, numCols);
    }
    {
      const LO numRows = 2;
      const LO numCols = 3;
      out << "Test numRows=" << numRows << ", numCols=" << numCols << endl;
      //Teuchos::OSTab tab1 (out);
      testAllScalars<LO, OffsetType, scalar_type> (success, out, numRows, numCols);
    }
  }

}


} // namespace (anonymous)


template <typename scalar_t, typename lno_t, typename size_type, typename device>
void test_mkl_sparseMatrixHandle()
{
  using std::endl;
  using std::endl;
  class NullBuffer : public std::streambuf
  {
  public:
    int overflow(int c) { return c; }
  };
  NullBuffer null_buffer;
  //std::ostream &out = std::cout;
  std::ostream out(&null_buffer);
  bool success = true;
  out << "Run test" << endl;
  testAllScalarsAndLocalOrdinals <lno_t, size_type, scalar_t>(success, out);
  EXPECT_TRUE( success);
}




#define EXECUTE_TEST(SCALAR, ORDINAL, OFFSET, DEVICE) \
TEST_F( TestCategory, sparse ## _ ## mkl_sparseMatrixHandle ## _ ## SCALAR ## _ ## ORDINAL ## _ ## OFFSET ## _ ## DEVICE ) { \
  test_mkl_sparseMatrixHandle<SCALAR,ORDINAL,OFFSET,DEVICE>(); \
}


#if (defined (KOKKOSKERNELS_INST_DOUBLE) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT) \
 && defined (KOKKOSKERNELS_INST_OFFSET_INT) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(double, int, int, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_DOUBLE) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT64_T) \
 && defined (KOKKOSKERNELS_INST_OFFSET_INT) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(double, int64_t, int, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_DOUBLE) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT) \
 && defined (KOKKOSKERNELS_INST_OFFSET_SIZE_T) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(double, int, size_t, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_DOUBLE) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT64_T) \
 && defined (KOKKOSKERNELS_INST_OFFSET_SIZE_T) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(double, int64_t, size_t, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_FLOAT) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT) \
 && defined (KOKKOSKERNELS_INST_OFFSET_INT) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(float, int, int, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_FLOAT) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT64_T) \
 && defined (KOKKOSKERNELS_INST_OFFSET_INT) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(float, int64_t, int, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_FLOAT) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT) \
 && defined (KOKKOSKERNELS_INST_OFFSET_SIZE_T) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(float, int, size_t, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_FLOAT) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT64_T) \
 && defined (KOKKOSKERNELS_INST_OFFSET_SIZE_T) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(float, int64_t, size_t, TestExecSpace)
#endif


#if (defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT) \
 && defined (KOKKOSKERNELS_INST_OFFSET_INT) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(kokkos_complex_double, int, int, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT64_T) \
 && defined (KOKKOSKERNELS_INST_OFFSET_INT) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(kokkos_complex_double, int64_t, int, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT) \
 && defined (KOKKOSKERNELS_INST_OFFSET_SIZE_T) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(kokkos_complex_double, int, size_t, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT64_T) \
 && defined (KOKKOSKERNELS_INST_OFFSET_SIZE_T) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(kokkos_complex_double, int64_t, size_t, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT) \
 && defined (KOKKOSKERNELS_INST_OFFSET_INT) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(kokkos_complex_float, int, int, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT64_T) \
 && defined (KOKKOSKERNELS_INST_OFFSET_INT) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(kokkos_complex_float, int64_t, int, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT) \
 && defined (KOKKOSKERNELS_INST_OFFSET_SIZE_T) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(kokkos_complex_float, int, size_t, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT64_T) \
 && defined (KOKKOSKERNELS_INST_OFFSET_SIZE_T) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(kokkos_complex_float, int64_t, size_t, TestExecSpace)
#endif






