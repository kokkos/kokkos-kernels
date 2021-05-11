/*
Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory
Maintained by Tom Scogland <scogland1@llnl.gov>
CODE-756261, All rights reserved.
This file is part of camp.
For details about use and distribution, please read LICENSE and NOTICE from
http://github.com/llnl/camp
*/

#ifndef CAMP_NUMBER_IF_HPP
#define CAMP_NUMBER_IF_HPP

#include "camp/value.hpp"

#include <type_traits>

namespace camp
{

// TODO: document
template <bool Cond,
          typename Then = camp::true_type,
          typename Else = camp::false_type>
struct if_cs {
  using type = Then;
};

template <typename Then, typename Else>
struct if_cs<false, Then, Else> {
  using type = Else;
};

// TODO: document
template <bool Cond,
          typename Then = camp::true_type,
          typename Else = camp::false_type>
using if_c = typename if_cs<Cond, Then, Else>::type;

// TODO: document
template <typename Cond,
          typename Then = camp::true_type,
          typename Else = camp::false_type>
struct if_s : if_cs<Cond::value, Then, Else> {
};

template <typename Then, typename Else>
struct if_s<nil, Then, Else> : if_cs<false, Then, Else> {
};

// TODO: document
template <typename... Ts>
using if_ = typename if_s<Ts...>::type;

}  // end namespace camp

#endif /* CAMP_NUMBER_IF_HPP */
