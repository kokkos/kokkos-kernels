/*
Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory
Maintained by Tom Scogland <scogland1@llnl.gov>
CODE-756261, All rights reserved.
This file is part of camp.
For details about use and distribution, please read LICENSE and NOTICE from
http://github.com/llnl/camp
*/

#ifndef CAMP_NUMBER_NUMBER_HPP
#define CAMP_NUMBER_NUMBER_HPP

#include "camp/defines.hpp"

namespace camp
{

// TODO: document, consider making use/match std::integral_constant
template <class NumT, NumT v>
struct integral_constant {
  static constexpr NumT value = v;
  using value_type = NumT;
  using type = integral_constant;
  constexpr operator value_type() const noexcept { return value; }
  constexpr value_type operator()() const noexcept { return value; }
};

/**
 * @brief Short-form for a whole number
 *
 * @tparam N The integral value
 */
template <idx_t N>
using num = integral_constant<idx_t, N>;

using true_type = num<true>;
using false_type = num<false>;

using t = num<true>;

}  // end namespace camp

#endif /* CAMP_NUMBER_NUMBER_HPP */
