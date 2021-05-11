/*
Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory
Maintained by Tom Scogland <scogland1@llnl.gov>
CODE-756261, All rights reserved.
This file is part of camp.
For details about use and distribution, please read LICENSE and NOTICE from
http://github.com/llnl/camp
*/

#ifndef __CAMP_TYPE_TRAITS_IS_SAME_HPP
#define __CAMP_TYPE_TRAITS_IS_SAME_HPP

#include "camp/defines.hpp"
#include "camp/number/number.hpp"

namespace camp
{

template <typename T, typename U>
struct is_same_s : false_type {
};

template <typename T>
struct is_same_s<T, T> : true_type {
};

#if defined(CAMP_COMPILER_MSVC)
template <typename... Ts>
using is_same = typename is_same_s<Ts...>::type;
#else
template <typename T, typename U>
using is_same = typename is_same_s<T, U>::type;
#endif

template <typename T, typename U>
using is_same_t = is_same<T, U>;

}  // namespace camp

#endif /* __CAMP_TYPE_TRAITS_IS_SAME_HPP */
