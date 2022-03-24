BLAS1 -- KokkosKernels blas1 interfaces
=======================================

.. doxygenfunction:: KokkosBlas::axpby
.. doxygenfunction:: KokkosBlas::dot(const RV &, const XMV &, const YMV &, typename std::enable_if<Kokkos::is_view<RV>::value, int>::type = 0)
.. doxygenfunction:: KokkosBlas::dot(const XVector &, const YVector &)
.. doxygenfunction:: KokkosBlas::fill
.. doxygenfunction:: KokkosBlas::mult
.. doxygenfunction:: KokkosBlas::nrm1(const RV &, const XMV &, typename std::enable_if<Kokkos::is_view<RV>::value, int>::type = 0)
.. doxygenfunction:: KokkosBlas::nrm1(const XVector &)
.. doxygenfunction:: KokkosBlas::nrm2(const RV &R, const XMV &X, typename std::enable_if<Kokkos::is_view<RV>::value, int>::type = 0)
.. doxygenfunction:: KokkosBlas::nrm2(const XVector &x)
.. doxygenfunction:: KokkosBlas::nrm2w(const RV &R, const XMV &X, const XMV &W, typename std::enable_if<Kokkos::is_view<RV>::value, int>::type = 0)
.. doxygenfunction:: KokkosBlas::nrm2w(const XVector &x, const XVector &w)
.. doxygenfunction:: KokkosBlas::nrminf(const RV &R, const XMV &X, typename std::enable_if<Kokkos::is_view<RV>::value, int>::type = 0)
.. doxygenfunction:: KokkosBlas::nrminf(const XVector &x)
.. doxygenfunction:: KokkosBlas::reciprocal
.. doxygenfunction:: KokkosBlas::scal
.. doxygenfunction:: KokkosBlas::sum(const RV &R, const XMV &X, typename std::enable_if<Kokkos::is_view<RV>::value, int>::type = 0)
.. doxygenfunction:: KokkosBlas::update
