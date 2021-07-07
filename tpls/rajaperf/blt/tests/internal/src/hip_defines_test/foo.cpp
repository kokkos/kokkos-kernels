// Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
// other BLT Project Developers. See the top-level COPYRIGHT file for details
//
// SPDX-License-Identifier: (BSD-3-Clause)

int main()
{
    #if defined(FOO_ONLY) && !defined(BAR_ONLY)
    return 0;
    #else
    return 1;
    #endif
}
