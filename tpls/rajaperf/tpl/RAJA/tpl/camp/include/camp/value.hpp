/*
Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory
Maintained by Tom Scogland <scogland1@llnl.gov>
CODE-756261, All rights reserved.
This file is part of camp.
For details about use and distribution, please read LICENSE and NOTICE from
http://github.com/llnl/camp
*/

#ifndef __CAMP_value_hpp
#define __CAMP_value_hpp

#include "camp/number/number.hpp"

namespace camp
{

/// \cond
namespace detail
{
  struct nothing;
}
/// \endcond

// TODO: document
template <typename val = detail::nothing>
struct value;

template <typename val>
struct value {
  using type = val;
};

template <>
struct value<detail::nothing> {
  using type = value;
};

/// A non-value, in truth tests evaluates to false
using nil = value<>;

/// Test whether a type is a valid camp value
template <typename Val>
struct is_value_s {
  using type = camp::t;
};

/// Test whether a type is a valid camp value
template <typename Val>
using is_value = typename is_value_s<Val>::type;
}  // namespace camp

#endif /* __CAMP_value_hpp */
