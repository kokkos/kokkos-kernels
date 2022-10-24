SPARSE -- KokkosKernels sparse interfaces
=========================================

crsmatrix
---------
.. doxygenclass::    KokkosSparse::CrsMatrix
    :members:

spmv
----

.. doxygenfunctions:: KokkosSparse::spmv(KokkosKernels::Experimental::Controls, const char[], const AlphaType&, const AMatrix&, const XVector&, const BetaType&, const YVector&)
.. doxygenfunctions:: KokkosSparse::spmv(KokkosKernels::Experimental::Controls controls, const char mode[], const AlphaType &alpha, const AMatrix &A, const XVector &x, const BetaType &beta, const YVector &y)
.. doxygenfunctions:: KokkosSparse::spmv(KokkosKernels::Experimental::Controls controls, const char mode[], const AlphaType &alpha, const AMatrix &A, const XVector &x, const BetaType &beta, const YVector &y, const RANK_ONE)
.. doxygenfunctions:: KokkosSparse::spmv(KokkosKernels::Experimental::Controls controls, const char mode[], const AlphaType &alpha, const AMatrix &A, const XVector &x, const BetaType &beta, const YVector &y, const RANK_TWO)
.. doxygenfunctions:: KokkosSparse::spmv(const char mode[], const AlphaType &alpha, const AMatrix &A, const XVector &x, const BetaType &beta, const YVector &y)


trsv
----
.. doxygenfunction:: KokkosSparse::trsv

spgemm
------
.. doxygenfunction:: KokkosSparse::spgemm

gauss
-----
.. doxygenfunction:: KokkosSparse::gauss
