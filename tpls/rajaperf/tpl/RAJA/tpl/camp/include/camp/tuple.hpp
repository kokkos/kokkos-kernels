/*
Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory
Maintained by Tom Scogland <scogland1@llnl.gov>
CODE-756261, All rights reserved.
This file is part of camp.
For details about use and distribution, please read LICENSE and NOTICE from
http://github.com/llnl/camp
*/

#ifndef camp_tuple_HPP__
#define camp_tuple_HPP__

/*!
 * \file
 *
 * \brief   Exceptionally basic tuple for host-device support
 */

#include <iostream>
#include <type_traits>

#include "camp/concepts.hpp"
#include "camp/map.hpp"

namespace camp
{

template <typename... Rest>
struct tuple;

template <typename TagList, typename... Elements>
class tagged_tuple;

template <template <typename... Ts> class Tup>
using is_tuple = typename std::is_base_of<tuple<>, Tup<>>::type;

template <typename Tuple>
struct tuple_size;

template <camp::idx_t i, typename T>
struct tuple_element {
  using type = camp::at_v<typename T::TList, i>;
};

template <camp::idx_t i, typename T>
using tuple_element_t = typename tuple_element<i, T>::type;

template <typename T, typename Tuple>
using tuple_ebt_t =
    typename tuple_element<camp::at_key<typename Tuple::TMap, T>::value,
                           Tuple>::type;


namespace internal
{

  template <class T>
  struct unwrap_refwrapper {
    using type = T;
  };

  template <class T>
  struct unwrap_refwrapper<std::reference_wrapper<T>> {
    using type = T&;
  };

  template <class T>
  using special_decay_t =
      typename unwrap_refwrapper<typename std::decay<T>::type>::type;
}  // namespace internal

template <camp::idx_t index, class Tuple>
CAMP_HOST_DEVICE constexpr auto get(const Tuple& t) noexcept
    -> tuple_element_t<index, Tuple> const&;
template <camp::idx_t index, class Tuple>
CAMP_HOST_DEVICE constexpr auto get(Tuple& t) noexcept
    -> tuple_element_t<index, Tuple>&;

template <typename T, class Tuple>
CAMP_HOST_DEVICE constexpr auto get(const Tuple& t) noexcept
    -> tuple_ebt_t<T, Tuple> const&;
template <typename T, class Tuple>
CAMP_HOST_DEVICE constexpr auto get(Tuple& t) noexcept
    -> tuple_ebt_t<T, Tuple>&;

namespace internal
{
  template <camp::idx_t index,
            typename Type,
            bool Empty = std::is_empty<Type>::value>
  struct CAMP_EMPTY_BASES tuple_storage {
    CAMP_HOST_DEVICE constexpr tuple_storage() : val(){};

    CAMP_SUPPRESS_HD_WARN
    template <typename T>
    CAMP_HOST_DEVICE constexpr tuple_storage(T&& v)
        // initializing with (...) instead of {...} for compiler compatability
        // some compilers complain when Type has no members and we use {...} to
        // initialize val
        : val(std::forward<T>(v))
    {
    }

    CAMP_HOST_DEVICE constexpr const Type& get_inner() const noexcept
    {
      return val;
    }

    CAMP_HOST_DEVICE CAMP_CONSTEXPR14 Type& get_inner() noexcept { return val; }

  public:
    Type val;
  };
  template <camp::idx_t index, typename Type>
  struct CAMP_EMPTY_BASES tuple_storage<index, Type, true> : private Type {
    CAMP_HOST_DEVICE constexpr tuple_storage() : Type(){};

    static_assert(std::is_empty<Type>::value,
                  "this specialization should only ever be used for empty "
                  "types");

    CAMP_SUPPRESS_HD_WARN
    template <typename T>
    CAMP_HOST_DEVICE constexpr tuple_storage(T&& v) : Type(std::forward<T>(v))
    {
    }

    CAMP_HOST_DEVICE constexpr const Type& get_inner() const noexcept
    {
      return ((Type const*)this)[0];
    }

    CAMP_HOST_DEVICE CAMP_CONSTEXPR14 Type& get_inner() noexcept
    {
      return ((Type*)this)[0];
    }
  };

  template <typename Indices, typename Typelist>
  struct tuple_helper;

  class expand_tag
  {
  };

  template <typename... Types, camp::idx_t... Indices>
  struct CAMP_EMPTY_BASES
      tuple_helper<camp::idx_seq<Indices...>, camp::list<Types...>>
      : public internal::tuple_storage<Indices, Types>... {

    tuple_helper& operator=(const tuple_helper& rhs) = default;
    constexpr tuple_helper() = default;
    constexpr tuple_helper(tuple_helper const&) = default;
    constexpr tuple_helper(tuple_helper&&) = default;

    template <typename... Args>
    CAMP_HOST_DEVICE constexpr tuple_helper(Args&&... args)
        : tuple_storage<Indices, Types>(std::forward<Args>(args))...
    {
    }

    template <typename T>
    CAMP_HOST_DEVICE constexpr explicit tuple_helper(expand_tag, T&& rhs)
        : tuple_helper(get<Indices>(rhs)...)
    {
    }

    template <typename T>
    CAMP_HOST_DEVICE constexpr explicit tuple_helper(expand_tag, const T& rhs)
        : tuple_helper(get<Indices>(rhs)...)
    {
    }

    template <typename RTuple>
    CAMP_HOST_DEVICE tuple_helper& operator=(const RTuple& rhs)
    {
      return (camp::sink((this->tuple_storage<Indices, Types>::get_inner() =
                              ::camp::get<Indices>(rhs))...),
              *this);
    }
  };

  template <typename Types, typename Indices>
  struct tag_map;
  template <typename... Types, camp::idx_t... Indices>
  struct tag_map<camp::list<Types...>, camp::idx_seq<Indices...>> {
    using type = camp::list<camp::list<Types, camp::num<Indices>>...>;
  };

  template <typename T, camp::idx_t I>
  using tpl_get_store = internal::tuple_storage<I, tuple_element_t<I, T>>;

}  // namespace internal


template <typename... Elements>
struct tuple {
private:
  using Self = tuple;
  using Base = internal::tuple_helper<camp::make_idx_seq_t<sizeof...(Elements)>,
                                      camp::list<Elements...>>;

  template <typename... Ts>
  struct is_pack_this_tuple : false_type {
  };
  template <typename That>
  struct is_pack_this_tuple<That> : std::is_same<tuple, decay<That>> {
  };

public:
  using TList = camp::list<Elements...>;
  using TMap = typename internal::tag_map<
      camp::list<Elements...>,
      camp::make_idx_seq_t<sizeof...(Elements)>>::type;
  using type = tuple;

private:
  Base base;

  template <camp::idx_t index, class Tuple>
  CAMP_HOST_DEVICE constexpr friend auto get(const Tuple& t) noexcept
      -> tuple_element_t<index, Tuple> const&;
  template <camp::idx_t index, class Tuple>
  CAMP_HOST_DEVICE constexpr friend auto get(Tuple& t) noexcept
      -> tuple_element_t<index, Tuple>&;

  template <typename T, class Tuple>
  CAMP_HOST_DEVICE constexpr friend auto get(const Tuple& t) noexcept
      -> tuple_ebt_t<T, Tuple> const&;
  template <typename T, class Tuple>
  CAMP_HOST_DEVICE constexpr friend auto get(Tuple& t) noexcept
      -> tuple_ebt_t<T, Tuple>&;

public:

  CAMP_HOST_DEVICE constexpr explicit tuple(const Elements&... rest)
      : base{rest...}
  {
  }

  template <typename... Args,
            typename std::enable_if<
                !is_pack_this_tuple<Args...>::value>::type* = nullptr>
  CAMP_HOST_DEVICE constexpr explicit tuple(Args&&... rest)
      : base{std::forward<Args>(rest)...}
  {
  }

  template <typename... RTypes>
  CAMP_HOST_DEVICE constexpr explicit tuple(const tuple<RTypes...>& rhs)
      : base(internal::expand_tag{}, rhs)
  {
  }

  template <typename... RTypes>
  CAMP_HOST_DEVICE constexpr explicit tuple(tuple<RTypes...>&& rhs)
      : base(internal::expand_tag{}, rhs)
  {
  }

  template <typename... RTypes>
  CAMP_HOST_DEVICE CAMP_CONSTEXPR14 Self& operator=(const tuple<RTypes...>& rhs)
  {
    base.operator=(rhs);
    return *this;
  }
};

// NOTE: this class should be built on top of tuple.  Any attempt to do that
// causes nvcc9.1 to die in EDG. As soon as nvcc9.1 goes away, this should be
// reduced to just a public derivation of tuple that overrides TMap.
template <typename TagList, typename... Elements>
class tagged_tuple : public tuple<Elements...>
{
  using Self = tagged_tuple;
  using Base = tuple<Elements...>;

public:
  using TList = camp::list<Elements...>;
  using TMap = typename internal::
      tag_map<TagList, camp::make_idx_seq_t<sizeof...(Elements)>>::type;
  using type = tagged_tuple;
  using Base::Base;

  constexpr tagged_tuple() = default;

  constexpr tagged_tuple(tagged_tuple const& o) = default;
  constexpr tagged_tuple(tagged_tuple&& o) = default;

  tagged_tuple& operator=(tagged_tuple const& rhs) = default;
  tagged_tuple& operator=(tagged_tuple&& rhs) = default;

  CAMP_HOST_DEVICE constexpr explicit tagged_tuple(const Base& rhs) : Base{rhs}
  {
  }

  template <typename... RTypes>
  CAMP_HOST_DEVICE constexpr explicit tagged_tuple(
      const tagged_tuple<RTypes...>& rhs)
      : Base(rhs)
  {
  }

  template <typename... RTypes>
  CAMP_HOST_DEVICE constexpr explicit tagged_tuple(
      tagged_tuple<RTypes...>&& rhs)
      : Base(rhs)
  {
  }

  using Base::operator=;
  template <typename... RTypes>
  CAMP_HOST_DEVICE CAMP_CONSTEXPR14 Self& operator=(
      const tagged_tuple<RTypes...>& rhs)
  {
    Base::operator=(rhs);
    return *this;
  }
};

template <>
class tuple<>
{
public:
  using TList = camp::list<>;
  using TMap = TList;
  using type = tuple;
};

template <typename... Tags, typename... Args>
struct as_list_s<tagged_tuple<camp::list<Tags...>, Args...>> {
  using type = list<Args...>;
};

// by index
template <camp::idx_t index, class Tuple>
CAMP_HOST_DEVICE constexpr auto get(const Tuple& t) noexcept
    -> tuple_element_t<index, Tuple> const&
{
  using internal::tpl_get_store;
  static_assert(tuple_size<Tuple>::value > index, "index out of range");
  return static_cast<tpl_get_store<Tuple, index> const&>(t.base).get_inner();
}

template <camp::idx_t index, class Tuple>
CAMP_HOST_DEVICE constexpr auto get(Tuple& t) noexcept
    -> tuple_element_t<index, Tuple>&
{
  using internal::tpl_get_store;
  static_assert(tuple_size<Tuple>::value > index, "index out of range");
  return static_cast<tpl_get_store<Tuple, index>&>(t.base).get_inner();
}

// by type
template <typename T, class Tuple>
CAMP_HOST_DEVICE constexpr auto get(const Tuple& t) noexcept
    -> tuple_ebt_t<T, Tuple> const&
{
  using internal::tpl_get_store;
  using index_type = camp::at_key<typename Tuple::TMap, T>;
  static_assert(!std::is_same<camp::nil, index_type>::value,
                "invalid type index");

  return static_cast<tpl_get_store<Tuple, index_type::value>&>(t.base)
      .get_inner();
}

template <typename T, class Tuple>
CAMP_HOST_DEVICE constexpr auto get(Tuple& t) noexcept -> tuple_ebt_t<T, Tuple>&
{
  using internal::tpl_get_store;
  using index_type = camp::at_key<typename Tuple::TMap, T>;
  static_assert(!std::is_same<camp::nil, index_type>::value,
                "invalid type index");

  return static_cast<tpl_get_store<Tuple, index_type::value>&>(t.base)
      .get_inner();
}

template <typename... Args>
struct tuple_size<tuple<Args...>> : ::camp::num<sizeof...(Args)> {
};

template <typename L, typename... Args>
struct tuple_size<tagged_tuple<L, Args...>> : ::camp::num<sizeof...(Args)> {
};

template <typename T>
struct tuple_size<const T> : num<tuple_size<T>::value> {
};
template <typename T>
struct tuple_size<volatile T> : num<tuple_size<T>::value> {
};
template <typename T>
struct tuple_size<const volatile T> : num<tuple_size<T>::value> {
};

template <typename... Args>
CAMP_HOST_DEVICE constexpr auto make_tuple(Args&&... args)
    -> tuple<internal::special_decay_t<Args>...>
{
  return tuple<internal::special_decay_t<Args>...>{std::forward<Args>(args)...};
}

template <typename TagList, typename... Args>
CAMP_HOST_DEVICE constexpr auto make_tagged_tuple(Args&&... args)
    -> tagged_tuple<TagList, internal::special_decay_t<Args>...>
{
  return tagged_tuple<TagList, internal::special_decay_t<Args>...>{
      std::forward<Args>(args)...};
}

template <typename... Args>
CAMP_HOST_DEVICE constexpr auto forward_as_tuple(Args&&... args) noexcept
    -> tuple<Args&&...>
{
  return tuple<Args&&...>(std::forward<Args>(args)...);
}

template <class... Types>
CAMP_HOST_DEVICE constexpr tuple<Types&...> tie(Types&... args) noexcept
{
  return tuple<Types&...>{args...};
}

template <typename... Lelem,
          typename... Relem,
          camp::idx_t... Lidx,
          camp::idx_t... Ridx>
CAMP_HOST_DEVICE constexpr auto tuple_cat_pair(tuple<Lelem...> const& l,
                                               camp::idx_seq<Lidx...>,
                                               tuple<Relem...> const& r,
                                               camp::idx_seq<Ridx...>) noexcept
    -> tuple<camp::at_v<camp::list<Lelem...>, Lidx>...,
             camp::at_v<camp::list<Relem...>, Ridx>...>
{
  return ::camp::tuple<camp::at_v<camp::list<Lelem...>, Lidx>...,
                       camp::at_v<camp::list<Relem...>, Ridx>...>(
      ::camp::get<Lidx>(l)..., ::camp::get<Ridx>(r)...);
}

template <typename L, typename R>
CAMP_HOST_DEVICE constexpr auto tuple_cat_pair(L const& l, R const& r) noexcept
    -> decltype(tuple_cat_pair(l,
                               camp::idx_seq_from_t<L>{},
                               r,
                               camp::idx_seq_from_t<R>{}))
{
  return tuple_cat_pair(l,
                        camp::idx_seq_from_t<L>{},
                        r,
                        camp::idx_seq_from_t<R>{});
}

CAMP_SUPPRESS_HD_WARN
template <typename Fn, camp::idx_t... Sequence, typename TupleLike>
CAMP_HOST_DEVICE constexpr auto invoke_with_order(TupleLike&& t,
                                                  Fn&& f,
                                                  camp::idx_seq<Sequence...>)
    -> decltype(f(::camp::get<Sequence>(t)...))
{
  return f(::camp::get<Sequence>(t)...);
}

CAMP_SUPPRESS_HD_WARN
template <typename Fn, typename TupleLike>
CAMP_HOST_DEVICE constexpr auto invoke(TupleLike&& t, Fn&& f)
    -> decltype(invoke_with_order(
        std::forward<TupleLike>(t),
        std::forward<Fn>(f),
        camp::make_idx_seq_t<tuple_size<camp::decay<TupleLike>>::value>{}))
{
  return invoke_with_order(
      std::forward<TupleLike>(t),
      std::forward<Fn>(f),
      camp::make_idx_seq_t<tuple_size<camp::decay<TupleLike>>::value>{});
}

namespace detail
{
  template <class T, class Tuple, idx_t... I>
  constexpr T make_from_tuple_impl(Tuple&& t, idx_seq<I...>)
  {
    return T(::camp::get<I>(std::forward<Tuple>(t))...);
  }
}  // namespace detail

/// Instantiate T from tuple contents, like camp::invoke(tuple,constructor) but
/// functional
template <class T, class Tuple>
constexpr T make_from_tuple(Tuple&& t)
{
  return detail::make_from_tuple_impl<T>(
      std::forward<Tuple>(t),
      make_idx_seq_t<tuple_size<type::ref::rem<Tuple>>::value>{});
}
}  // namespace camp

namespace internal
{
template <class Tuple, camp::idx_t... Idxs>
void print_tuple(std::ostream& os, Tuple const& t, camp::idx_seq<Idxs...>)
{
  camp::sink((void*)&(os << (Idxs == 0 ? "" : ", ") << camp::get<Idxs>(t))...);
}
}  // namespace internal

template <class... Args>
auto operator<<(std::ostream& os, camp::tuple<Args...> const& t)
    -> std::ostream&
{
  os << "(";
  internal::print_tuple(os, t, camp::make_idx_seq_t<sizeof...(Args)>{});
  return os << ")";
}


#endif /* camp_tuple_HPP__ */
