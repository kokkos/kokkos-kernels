/*
Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory
Maintained by Tom Scogland <scogland1@llnl.gov>
CODE-756261, All rights reserved.
This file is part of camp.
For details about use and distribution, please read LICENSE and NOTICE from
http://github.com/llnl/camp
*/

#ifndef CAMP_HELPERS_HPP
#define CAMP_HELPERS_HPP

#include <cstddef>
#include <iterator>
#include <utility>

#include "camp/defines.hpp"

namespace camp
{

/// metafunction to get instance of pointer type
template <typename T>
T* declptr();

/// metafunction to get instance of value type
template <typename T>
CAMP_HOST_DEVICE
auto val() noexcept -> decltype(std::declval<T>());

/// metafunction to get instance of const type
template <typename T>
CAMP_HOST_DEVICE
auto cval() noexcept -> decltype(std::declval<T const>());

/// metafunction to expand a parameter pack and ignore result
template <typename... Ts>
CAMP_HOST_DEVICE
#if (__cplusplus >= 201402L)
constexpr
#endif
inline void sink(Ts const& ...)
{
}

// bring common utility routines into scope to allow ADL
using std::begin;
using std::swap;

namespace type
{
  namespace ref
  {
    template <class T>
    struct rem_s {
      using type = T;
    };
    template <class T>
    struct rem_s<T&> {
      using type = T;
    };
    template <class T>
    struct rem_s<T&&> {
      using type = T;
    };

    /// remove reference from T
    template <class T>
    using rem = typename rem_s<T>::type;

    /// add remove reference to T
    template <class T>
    using add = T&;
  }  // end namespace ref

  namespace rvref
  {
    /// add rvalue reference to T
    template <class T>
    using add = T&&;
  }  // end namespace rvref

  namespace ptr
  {
    template <class T>
    struct rem_s {
      using type = T;
    };
    template <class T>
    struct rem_s<T*> {
      using type = T;
    };

    /// remove pointer from T
    template <class T>
    using rem = typename rem_s<T>::type;

    /// add remove pointer to T
    template <class T>
    using add = T*;
  }  // end namespace ptr

  namespace c
  {
    template <class T>
    struct rem_s {
      using type = T;
    };
    template <class T>
    struct rem_s<const T> {
      using type = T;
    };

    /// remove const qualifier from T
    template <class T>
    using rem = typename rem_s<T>::type;

    /// add const qualifier to T
    template <class T>
    using add = const T;
  }  // namespace c

  namespace v
  {
    template <class T>
    struct rem_s {
      using type = T;
    };
    template <class T>
    struct rem_s<volatile T> {
      using type = T;
    };

    /// remove volatile qualifier from T
    template <class T>
    using rem = typename rem_s<T>::type;

    /// add volatile qualifier to T
    template <class T>
    using add = volatile T;
  }  // namespace v

  namespace cv
  {
    template <class T>
    struct rem_s {
      using type = T;
    };
    template <class T>
    struct rem_s<const T> {
      using type = T;
    };
    template <class T>
    struct rem_s<volatile T> {
      using type = T;
    };
    template <class T>
    struct rem_s<const volatile T> {
      using type = T;
    };

    /// remove const and volatile qualifiers from T
    template <class T>
    using rem = typename rem_s<T>::type;

    /// add const and volatile qualifiers to T
    template <class T>
    using add = const volatile T;
  }  // namespace cv
}  // end namespace type

template <typename T>
using decay = type::cv::rem<type::ref::rem<T>>;

template <typename T>
using plain = type::ref::rem<T>;

template <typename T>
using diff_from = decltype(val<plain<T>>() - val<plain<T>>());
template <typename T, typename U>
using diff_between = decltype(val<plain<T>>() - val<plain<U>>());

template <typename T>
using iterator_from = decltype(begin(val<plain<T>>()));

template <class T>
CAMP_HOST_DEVICE constexpr T&& forward(type::ref::rem<T>& t) noexcept
{
  return static_cast<T&&>(t);
}
template <class T>
CAMP_HOST_DEVICE constexpr T&& forward(type::ref::rem<T>&& t) noexcept
{
  return static_cast<T&&>(t);
}

template <typename T>
CAMP_HOST_DEVICE constexpr type::ref::rem<T>&& move(T&& t) noexcept
{
  return static_cast<type::ref::rem<T>&&>(t);
}

template <typename T>
CAMP_HOST_DEVICE void safe_swap(T& t1, T& t2)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  T temp{std::move(t1)};
  t1 = std::move(t2);
  t2 = std::move(temp);
#else
  using std::swap;
  swap(t1, t2);
#endif
}

template <typename T, typename = decltype(sink(swap(val<T>(), val<T>())))>
CAMP_HOST_DEVICE void safe_swap(T& t1, T& t2)
{
  using std::swap;
  swap(t1, t2);
}
}  // namespace camp

#endif /* CAMP_HELPERS_HPP */
