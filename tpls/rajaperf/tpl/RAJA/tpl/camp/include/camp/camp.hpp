/*
Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory
Maintained by Tom Scogland <scogland1@llnl.gov>
CODE-756261, All rights reserved.
This file is part of camp.
For details about use and distribution, please read LICENSE and NOTICE from
http://github.com/llnl/camp
*/

#ifndef __CAMP_HPP
#define __CAMP_HPP

#include <array>
#include <type_traits>

#include "camp/defines.hpp"
#include "camp/helpers.hpp"
#include "camp/lambda.hpp"
#include "camp/list.hpp"
#include "camp/make_unique.hpp"
#include "camp/map.hpp"
#include "camp/number.hpp"
#include "camp/size.hpp"
#include "camp/tuple.hpp"
#include "camp/value.hpp"

#include "camp/detail/test.hpp"

namespace camp
{
// Fwd
template <typename Seq>
struct flatten;

// Sequences
//// list

template <typename Seq, typename T>
struct append;
template <typename... Elements, typename T>
struct append<list<Elements...>, T> {
  using type = list<Elements..., T>;
};

template <typename Seq, typename T>
struct prepend;
template <typename... Elements, typename T>
struct prepend<list<Elements...>, T> {
  using type = list<Elements..., T>;
};

template <typename Seq, typename T>
struct extend;
template <typename... Elements, typename... NewElements>
struct extend<list<Elements...>, list<NewElements...>> {
  using type = list<Elements..., NewElements...>;
};

namespace detail
{
  template <typename CurSeq, size_t N, typename... Rest>
  struct flatten_impl;
  template <typename CurSeq>
  struct flatten_impl<CurSeq, 0> {
    using type = CurSeq;
  };
  template <typename... CurSeqElements,
            size_t N,
            typename First,
            typename... Rest>
  struct flatten_impl<list<CurSeqElements...>, N, First, Rest...> {
    using type = typename flatten_impl<list<CurSeqElements..., First>,
                                       N - 1,
                                       Rest...>::type;
  };
  template <typename... CurSeqElements,
            size_t N,
            typename... FirstInnerElements,
            typename... Rest>
  struct flatten_impl<list<CurSeqElements...>,
                      N,
                      list<FirstInnerElements...>,
                      Rest...> {
    using first_inner_flat =
        typename flatten_impl<list<>,
                              sizeof...(FirstInnerElements),
                              FirstInnerElements...>::type;
    using cur_and_first =
        typename extend<list<CurSeqElements...>, first_inner_flat>::type;
    using type = typename flatten_impl<cur_and_first, N - 1, Rest...>::type;
  };
}  // namespace detail

template <typename... Elements>
struct flatten<list<Elements...>>
    : detail::flatten_impl<list<>, sizeof...(Elements), Elements...> {
};

template <typename... Seqs>
struct join;
template <typename Seq1, typename Seq2, typename... Rest>
struct join<Seq1, Seq2, Rest...> {
      using type = typename join<typename extend<Seq1, Seq2>::type, Rest...>::type;
};
template <typename Seq1>
struct join<Seq1> {
      using type = Seq1;
};
template <>
struct join<> {
  using type = list<>;
};

template <template <typename...> class Op, typename T>
struct transform;
template <template <typename...> class Op, typename... Elements>
struct transform<Op, list<Elements...>> {
  using type = list<typename Op<Elements>::type...>;
};

namespace detail
{
  template <template <typename...> class Op, typename Current, typename... Rest>
  struct accumulate_impl;
  template <template <typename...> class Op,
            typename Current,
            typename First,
            typename... Rest>
  struct accumulate_impl<Op, Current, First, Rest...> {
    using current = typename Op<Current, First>::type;
    using type = typename accumulate_impl<Op, current, Rest...>::type;
  };
  template <template <typename...> class Op, typename Current>
  struct accumulate_impl<Op, Current> {
    using type = Current;
  };
}  // namespace detail

template <template <typename...> class Op, typename Initial, typename Seq>
struct accumulate;
template <template <typename...> class Op,
          typename Initial,
          typename... Elements>
struct accumulate<Op, Initial, list<Elements...>> {
  using type = typename detail::accumulate_impl<Op, Initial, Elements...>::type;
};


namespace detail
{
  template<class, class>
  struct product_impl{};
  template<class... Xs, class... Ys>
    struct product_impl<list<Xs...>, list<Ys...>> {
      using type = list<list<Xs..., Ys>...>;
    };
  template<class, class>
  struct product{};
  template<class... Seqs, class... vals>
    struct product<list<Seqs...>, list<vals...>> {
      using type = typename join<typename product_impl<Seqs, list<vals...>>::type...>::type;
    };
} /* detail */
template<class ... Seqs>
using cartesian_product = typename accumulate<detail::product, list<list<>>, list<Seqs...>>::type;

CAMP_MAKE_L(accumulate);

/**
 * @brief Get the index of the first instance of T in L
 */
template <typename T, typename L>
struct index_of;
template <typename T, typename... Elements>
struct index_of<T, list<Elements...>> {
  template <typename Seq, typename Item>
  using inc_until =
      if_<typename std::is_same<T, Item>::type,
          if_c<size<Seq>::value == 1,
               typename prepend<Seq, num<first<Seq>::value>>::type,
               Seq>,
          list<num<first<Seq>::value + 1>>>;
  using indices =
      typename accumulate<inc_until, list<num<0>>, list<Elements...>>::type;
  using type =
      typename if_c<size<indices>::value == 2, first<indices>, camp::nil>::type;
};

template <template <typename...> class Op, typename Seq>
struct filter;

template <template <typename...> class Op, typename... Elements>
struct filter<Op, list<Elements...>> {
  template <typename Seq, typename T>
  using append_if =
      if_<typename Op<T>::type, typename append<Seq, T>::type, Seq>;
  using type = typename accumulate<append_if, list<>, list<Elements...>>::type;
};

CAMP_MAKE_L(filter);

//// size

template <typename T, T... Args>
struct size<int_seq<T, Args...>> {
  constexpr static idx_t value{sizeof...(Args)};
  using type = num<sizeof...(Args)>;
};

}  // end namespace camp

#endif /* __CAMP_HPP */
