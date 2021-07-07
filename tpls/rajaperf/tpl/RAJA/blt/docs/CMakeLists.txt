# Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
# other BLT Project Developers. See the top-level COPYRIGHT file for details
# 
# SPDX-License-Identifier: (BSD-3-Clause)

cmake_minimum_required(VERSION 3.8)

project( blt_docs )

###############################################################################
# Setup BLT
###############################################################################

if(NOT BLT_SOURCE_DIR)
    set(BLT_SOURCE_DIR "blt")
endif()

include(${BLT_SOURCE_DIR}/SetupBLT.cmake)

if (SPHINX_FOUND)
    blt_add_sphinx_target(blt_sphinx_tutorial_docs)
endif()
