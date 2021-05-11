/*
Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory
Maintained by Tom Scogland <scogland1@llnl.gov>
CODE-756261, All rights reserved.
This file is part of camp.
For details about use and distribution, please read LICENSE and NOTICE from
http://github.com/llnl/camp
*/

#ifndef CAMP_VALUE_EVAL_HPP
#define CAMP_VALUE_EVAL_HPP

namespace camp
{

// TODO: document
template <typename Val>
using eval = typename Val::type;

}  // end namespace camp

#endif /* CAMP_VALUE_EVAL_HPP */
