KokkosBatched::Vector
###################

Defined in header `KokkosBatched_Vector.hpp <https://github.com/kokkos/kokkos-kernels/blob/master/batched/dense/src/KokkosBatched_Vector.hpp>`_

.. code:: c++

    namespace KokkosBatched {

    template <typename T, int l>
    class Vector;

    template <typename T, int l>
    struct is_vector<Vector<SIMD<T>, l>> : public std::true_type {};

    template <typename ValueType, typename MemorySpace>
    struct DefaultVectorLength {
      enum : int { value = 1 };
    };

    // Specializations for different types and spaces
    
    template <typename ValueType, typename MemorySpace>
    struct DefaultInternalVectorLength {
      enum : int { value = 1 };
    };
    
    // Specializations for different types and spaces

    template <typename T>
    struct MagnitudeScalarType;

    // Specializations for different types

    }

Provides SIMD-capable vector types for batched linear algebra operations. The Vector template acts as a wrapper around SIMD instructions, allowing efficient vectorization of batched operations.

The KokkosBatched::Vector class is a fundamental building block for the KokkosBatched library, enabling efficient SIMD operations on small vectors. It provides:

1. Architecture-specific SIMD optimizations
2. Value type and size abstractions
3. Load/store operations for memory alignment
4. Template specializations for different hardware architectures

This header also provides traits for determining appropriate vector lengths for different scalar types and memory spaces, allowing for optimal SIMD utilization on different architectures.

Type Requirements
----------------

- ``T`` must be a scalar type (float, double, complex<float>, complex<double>)
- ``l`` must be a positive integer specifying the vector length
- ``MemorySpace`` must be a valid Kokkos memory space

Key Components
-------------

1. **Vector Template**:
   - Primary template for SIMD vectors

2. **DefaultVectorLength**:
   - Provides architecture-specific optimal vector lengths

3. **DefaultInternalVectorLength**:
   - Provides internal vector length optimizations

4. **MagnitudeScalarType**:
   - Extracts the magnitude type from scalar types (e.g., float from complex<float>)

5. **is_vector**:
   - Type trait to check if a type is a Vector

Example
=======

.. code:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Vector.hpp>

    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Get the default vector length for double on the host
        constexpr int vector_length = 
          KokkosBatched::DefaultVectorLength<double, Kokkos::HostSpace>::value;
        
        printf("Default vector length for double on HostSpace: %d\n", vector_length);
        
        // Create a SIMD vector
        KokkosBatched::Vector<KokkosBatched::SIMD<double>, vector_length> vec1(1.0);
        
        // Use within a Kokkos kernel (for demonstration)
        Kokkos::parallel_for(1, KOKKOS_LAMBDA(const int&) {
          // Create a SIMD vector in device code
          KokkosBatched::Vector<KokkosBatched::SIMD<double>, vector_length> vec2(2.0);
          
          // Vectors support arithmetic operations
          // Note: This example is simplified; real usage would be more complex
          // and integrated with other batched operations
          printf("SIMD vector operations demo:\n");
          printf("  Vector length: %d\n", vector_length);
          
          // Printing elements for demonstration (in real code, would use storeAligned)
          printf("  vec2 elements: ");
          for (int i = 0; i < vector_length; ++i) {
            printf("%.1f ", vec2[i]);
          }
          printf("\n");
        });
        
        // Demonstrate vector length traits for various types
        printf("\nDefault vector lengths for different types on HostSpace:\n");
        printf("  float: %d\n", KokkosBatched::DefaultVectorLength<float, Kokkos::HostSpace>::value);
        printf("  double: %d\n", KokkosBatched::DefaultVectorLength<double, Kokkos::HostSpace>::value);
        printf("  complex<float>: %d\n", 
               KokkosBatched::DefaultVectorLength<Kokkos::complex<float>, Kokkos::HostSpace>::value);
        printf("  complex<double>: %d\n", 
               KokkosBatched::DefaultVectorLength<Kokkos::complex<double>, Kokkos::HostSpace>::value);
        
        // Demonstrate magnitude scalar type extraction
        printf("\nMagnitude scalar types:\n");
        printf("  MagnitudeScalarType<float>::type is %s\n", 
               typeid(typename KokkosBatched::MagnitudeScalarType<float>::type).name());
        printf("  MagnitudeScalarType<complex<float>>::type is %s\n", 
               typeid(typename KokkosBatched::MagnitudeScalarType<Kokkos::complex<float>>::type).name());
        
        printf("\nUsing SIMD vectors with KokkosBatched operations:\n");
        printf("  The KokkosBatched::Vector class is primarily used internally by\n");
        printf("  the library to enable SIMD operations for batched linear algebra.\n");
        printf("  End users typically don't need to interact with it directly.\n");
        printf("  Instead, users work with regular Kokkos::View objects, and the\n");
        printf("  KokkosBatched algorithms automatically leverage SIMD when appropriate.\n");
      }
      Kokkos::finalize();
      return 0;
    }
