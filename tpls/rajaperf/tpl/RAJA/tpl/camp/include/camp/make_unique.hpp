/*
Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory
Maintained by Tom Scogland <scogland1@llnl.gov>
CODE-756261, All rights reserved.
This file is part of camp.
For details about use and distribution, please read LICENSE and NOTICE from
http://github.com/llnl/camp
*/
#ifndef __CAMP_make_unique_hpp
#define __CAMP_make_unique_hpp

#include <memory>

namespace camp {

template <typename T, typename... Args>
constexpr 
inline
std::unique_ptr<T> make_unique(Args&&... args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

} // end of namespace umpire

#endif /* __CAMP_make_unique_hpp */

