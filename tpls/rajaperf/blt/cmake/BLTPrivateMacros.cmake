# Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
# other BLT Project Developers. See the top-level COPYRIGHT file for details
#
# SPDX-License-Identifier: (BSD-3-Clause)

include(CMakeParseArguments)

## Internal BLT CMake Macros


##-----------------------------------------------------------------------------
## blt_determine_scope(TARGET <target>
##                     SCOPE  <PUBLIC (Default)| INTERFACE | PRIVATE>
##                     OUT    <out variable name>)
##
## Returns the normalized scope string for a given SCOPE and TARGET to be used
## in BLT macros.
##
## TARGET - Name of CMake Target that the property is being added to
##          Note: the only real purpose of this parameter is to make sure we aren't
##                adding returning other than INTERFACE for Interface Libraries
## SCOPE  - case-insensitive scope string, defaults to PUBLIC
## OUT    - variable that is filled with the uppercased scope
##
##-----------------------------------------------------------------------------
macro(blt_determine_scope)

    set(options)
    set(singleValueArgs TARGET SCOPE OUT)
    set(multiValueArgs )

    # Parse the arguments
    cmake_parse_arguments(arg "${options}" "${singleValueArgs}"
                        "${multiValueArgs}" ${ARGN} )

    # Convert to upper case and strip white space
    string(TOUPPER "${arg_SCOPE}" _uppercaseScope)
    string(STRIP "${_uppercaseScope}" _uppercaseScope )

    if("${_uppercaseScope}" STREQUAL "")
        # Default to public
        set(_uppercaseScope PUBLIC)
    elseif(NOT ("${_uppercaseScope}" STREQUAL "PUBLIC" OR
                "${_uppercaseScope}" STREQUAL "INTERFACE" OR
                "${_uppercaseScope}" STREQUAL "PRIVATE"))
        message(FATAL_ERROR "Given SCOPE (${arg_SCOPE}) is not valid, valid options are:"
                            "PUBLIC, INTERFACE, or PRIVATE")
    endif()

    if(TARGET ${arg_TARGET})
        get_property(_targetType TARGET ${arg_TARGET} PROPERTY TYPE)
        if(${_targetType} STREQUAL "INTERFACE_LIBRARY")
            # Interface targets can only set INTERFACE
            if("${_uppercaseScope}" STREQUAL "PUBLIC" OR
               "${_uppercaseScope}" STREQUAL "INTERFACE")
                set(${arg_OUT} INTERFACE)
            else()
                message(FATAL_ERROR "Cannot set PRIVATE scope to Interface Library."
                                    "Change to Scope to INTERFACE.")
            endif()
        else()
            set(${arg_OUT} ${_uppercaseScope})
        endif()
    else()
        set(${arg_OUT} ${_uppercaseScope})
    endif()

    unset(_targetType)
    unset(_uppercaseScope)

endmacro(blt_determine_scope)


##-----------------------------------------------------------------------------
## blt_error_if_target_exists()
##
## Checks if target already exists in CMake project and errors out with given
## error_msg.
##-----------------------------------------------------------------------------
function(blt_error_if_target_exists target_name error_msg)
    if (TARGET ${target_name})
        message(FATAL_ERROR "${error_msg}Duplicate target name: ${target_name}")
    endif()
endfunction()

##-----------------------------------------------------------------------------
## blt_fix_fortran_openmp_flags(<target name>)
##
## Fixes the openmp flags for a Fortran target if they are different from the
## corresponding C/C++ OpenMP flags.
##-----------------------------------------------------------------------------
function(blt_fix_fortran_openmp_flags target_name)

    if (ENABLE_FORTRAN AND ENABLE_OPENMP AND BLT_OPENMP_FLAGS_DIFFER)

        # The OpenMP interface library will have been added as a direct
        # link dependency instead of via flags
        get_target_property(target_link_libs ${target_name} LINK_LIBRARIES)
        if ( target_link_libs )
            # Since this is only called on executable targets we can safely convert
            # from a "real" target back to a "fake" one as this is a sink vertex in
            # the DAG.  Only the link flags need to be modified.
            list(FIND target_link_libs "openmp" _omp_index)
            if(${_omp_index} GREATER -1)
                message(STATUS "Fixing OpenMP Flags for target[${target_name}]")

                # Remove openmp from libraries
                list(REMOVE_ITEM target_link_libs "openmp")
                set_target_properties( ${target_name} PROPERTIES
                                       LINK_LIBRARIES "${target_link_libs}" )

                # Add openmp compile flags verbatim w/ generator expression
                get_target_property(omp_compile_flags openmp INTERFACE_COMPILE_OPTIONS)
                target_compile_options(${target_name} PUBLIC ${omp_compile_flags})

                # Change CXX flags to Fortran flags

                # These are set through blt_add_target_link_flags which needs to use
                # the link_libraries for interface libraries in CMake < 3.13
                if( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.13.0" )
                    get_target_property(omp_link_flags openmp INTERFACE_LINK_OPTIONS)
                else()
                    get_target_property(omp_link_flags openmp INTERFACE_LINK_LIBRARIES)
                endif()

                string( REPLACE "${OpenMP_CXX_FLAGS}" "${OpenMP_Fortran_FLAGS}"
                        correct_omp_link_flags
                        "${omp_link_flags}"
                        )
                if( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.13.0" )
                    target_link_options(${target_name} PRIVATE "${correct_omp_link_flags}")
                else()
                    set_property(TARGET ${target_name} APPEND PROPERTY LINK_FLAGS "${correct_omp_link_flags}")
                endif()
            endif()

            # Handle registered library general case

            # OpenMP is an interface library which doesn't have a LINK_FLAGS property
            # in versions < 3.13
            set(_property_name LINK_FLAGS)
            if( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.13.0" )
                # In CMake 3.13+, LINK_FLAGS was converted to LINK_OPTIONS.
                set(_property_name LINK_OPTIONS)
            endif()
            get_target_property(target_link_flags ${target_name} ${_property_name})
            if ( target_link_flags )

                string( REPLACE "${OpenMP_CXX_FLAGS}" "${OpenMP_Fortran_FLAGS}"
                        correct_link_flags
                        "${target_link_flags}"
                        )
                set_target_properties( ${target_name} PROPERTIES ${_property_name}
                                    "${correct_link_flags}" )
            endif()
        endif()

    endif()

endfunction()

##-----------------------------------------------------------------------------
## blt_find_executable(NAME         <name of program to find>
##                     EXECUTABLES  [exe1 [exe2 ...]])
##
## This macro attempts to find the given executable via either a previously defined
## <UPPERCASE_NAME>_EXECUTABLE or using find_program with the given EXECUTABLES.
## if EXECUTABLES is left empty, then NAME is used.  This macro will only attempt
## to locate the executable if <UPPERCASE_NAME>_ENABLED is TRUE.
##
## If successful the following variables will be defined:
## <UPPERCASE_NAME>_FOUND
## <UPPERCASE_NAME>_EXECUTABLE
##-----------------------------------------------------------------------------
macro(blt_find_executable)

    set(options)
    set(singleValueArgs NAME)
    set(multiValueArgs  EXECUTABLES)

    # Parse the arguments
    cmake_parse_arguments(arg "${options}" "${singleValueArgs}"
                        "${multiValueArgs}" ${ARGN} )

    # Check arguments
    if ( NOT DEFINED arg_NAME )
        message( FATAL_ERROR "Must provide a NAME argument to the 'blt_find_executable' macro" )
    endif()

    string(TOUPPER ${arg_NAME} _ucname)

    message(STATUS "${arg_NAME} support is ${ENABLE_${_ucname}}")
    if (${ENABLE_${_ucname}})
        set(_exes ${arg_NAME})
        if (DEFINED arg_EXECUTABLES)
            set(_exes ${arg_EXECUTABLES})
        endif()

        if (${_ucname}_EXECUTABLE)
            if (NOT EXISTS ${${_ucname}_EXECUTABLE})
                message(FATAL_ERROR "User defined ${_ucname}_EXECUTABLE does not exist. Fix/unset variable or set ENABLE_${_ucname} to OFF.")
            endif()
        else()
            find_program(${_ucname}_EXECUTABLE
                         NAMES ${_exes}
                         DOC "Path to ${arg_NAME} executable")
        endif()

        # Handle REQUIRED and QUIET arguments
        # this will also set ${_ucname}_FOUND to true if ${_ucname}_EXECUTABLE exists
        include(FindPackageHandleStandardArgs)
        find_package_handle_standard_args(${arg_NAME}
                                          "Failed to locate ${arg_NAME} executable"
                                          ${_ucname}_EXECUTABLE)
    endif()
endmacro(blt_find_executable)


##------------------------------------------------------------------------------
## blt_inherit_target_info( TO       <target>
##                          FROM     <target>
##                          OBJECT   [TRUE|FALSE])
##
##  The purpose of this macro is if you want to grab all the inheritable info
##  from the FROM target but don't want to make the TO target depend on it.
##  Which is useful if you don't want to export the FROM target.
##
##  The OBJECT parameter is because object libraries can only inherit certain
##  properties.
##
##  This inherits the following properties:
##    INTERFACE_COMPILE_DEFINITIONS
##    INTERFACE_INCLUDE_DIRECTORIES
##    INTERFACE_LINK_DIRECTORIES
##    INTERFACE_LINK_LIBRARIES
##    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
##------------------------------------------------------------------------------
macro(blt_inherit_target_info)
    set(options)
    set(singleValueArgs TO FROM OBJECT)
    set(multiValueArgs)

    # Parse the arguments
    cmake_parse_arguments(arg "${options}" "${singleValueArgs}"
                        "${multiValueArgs}" ${ARGN} )

    # Check arguments
    if ( NOT DEFINED arg_TO )
        message( FATAL_ERROR "Must provide a TO argument to the 'blt_inherit_target' macro" )
    endif()

    if ( NOT DEFINED arg_FROM )
        message( FATAL_ERROR "Must provide a FROM argument to the 'blt_inherit_target' macro" )
    endif()

    get_target_property(_interface_system_includes
                        ${arg_FROM} INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
    if ( _interface_system_includes )
        target_include_directories(${arg_TO} SYSTEM PUBLIC ${_interface_system_includes})
    endif()

    get_target_property(_interface_includes
                        ${arg_FROM} INTERFACE_INCLUDE_DIRECTORIES)
    if ( _interface_includes )
        target_include_directories(${arg_TO} PUBLIC ${_interface_includes})
    endif()

    get_target_property(_interface_defines
                        ${arg_FROM} INTERFACE_COMPILE_DEFINITIONS)
    if ( _interface_defines )
        target_compile_definitions( ${arg_TO} PUBLIC ${_interface_defines})
    endif()

    if( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.13.0" )
        get_target_property(_interface_link_options
                            ${arg_FROM} INTERFACE_LINK_OPTIONS)
        if ( _interface_link_options )
            target_link_options( ${arg_TO} PUBLIC ${_interface_link_options})
        endif()
    endif()

    get_target_property(_interface_compile_options
                        ${arg_FROM} INTERFACE_COMPILE_OPTIONS)
    if ( _interface_compile_options )
        target_compile_options( ${arg_TO} PUBLIC ${_interface_compile_options})
    endif()

    if ( NOT arg_OBJECT )
        get_target_property(_interface_link_directories
                            ${arg_FROM} INTERFACE_LINK_DIRECTORIES)
        if ( _interface_link_directories )
            target_link_directories( ${arg_TO} PUBLIC ${_interface_link_directories})
        endif()

        get_target_property(_interface_link_libraries
                            ${arg_FROM} INTERFACE_LINK_LIBRARIES)
        if ( _interface_link_libraries )
            target_link_libraries( ${arg_TO} PUBLIC ${_interface_link_libraries})
        endif()
    endif()

endmacro(blt_inherit_target_info)

##------------------------------------------------------------------------------
## blt_expand_depends( DEPENDS_ON [dep1 ...]
##                     RESULT [variable] )
##------------------------------------------------------------------------------
macro(blt_expand_depends)
    set(options)
    set(singleValueArgs RESULT)
    set(multiValueArgs DEPENDS_ON)

    # Parse the arguments
    cmake_parse_arguments(arg "${options}" "${singleValueArgs}"
                        "${multiValueArgs}" ${ARGN} )

    # Expand dependency list
    set(_deps_to_process ${arg_DEPENDS_ON})
    set(_expanded_DEPENDS_ON)
    while(_deps_to_process)
        # Copy the current set of dependencies to process
        set(_current_deps_to_process ${_deps_to_process})
        # and add them to the full expanded list
        list(APPEND _expanded_DEPENDS_ON ${_deps_to_process})
        # Then clear it so we can check if new ones were added
        set(_deps_to_process)
        foreach( dependency ${_current_deps_to_process} )
            string(TOUPPER ${dependency} uppercase_dependency )
            if ( DEFINED _BLT_${uppercase_dependency}_DEPENDS_ON )
                foreach(new_dependency ${_BLT_${uppercase_dependency}_DEPENDS_ON})
                    # Don't add duplicates
                    if (NOT ${new_dependency} IN_LIST _expanded_DEPENDS_ON)
                        list(APPEND _deps_to_process ${new_dependency})
                    endif()
                endforeach()
            endif()
        endforeach()
    endwhile()

    # Write the output to the requested variable
    set(${arg_RESULT} ${_expanded_DEPENDS_ON})
endmacro()


##------------------------------------------------------------------------------
## blt_setup_target( NAME       [name]
##                   DEPENDS_ON [dep1 ...]
##                   OBJECT     [TRUE | FALSE])
##------------------------------------------------------------------------------
macro(blt_setup_target)

    set(options)
    set(singleValueArgs NAME OBJECT)
    set(multiValueArgs DEPENDS_ON)

    # Parse the arguments
    cmake_parse_arguments(arg "${options}" "${singleValueArgs}"
                        "${multiValueArgs}" ${ARGN} )

    # Check arguments
    if ( NOT DEFINED arg_NAME )
        message( FATAL_ERROR "Must provide a NAME argument to the 'blt_setup_target' macro" )
    endif()

    # Default to "real" scope, unless it's an interface library
    set(_private_scope PRIVATE)
    set(_public_scope  PUBLIC)
    get_target_property(_target_type ${arg_NAME} TYPE)
    if("${_target_type}" STREQUAL "INTERFACE_LIBRARY")
        set(_private_scope INTERFACE)
        set(_public_scope  INTERFACE)
    endif()

    # Expand dependency list - avoid "recalculating" if the information already exists
    set(_expanded_DEPENDS_ON)
    if(NOT "${_target_type}" STREQUAL "INTERFACE_LIBRARY")
        get_target_property(_expanded_DEPENDS_ON ${arg_NAME} BLT_EXPANDED_DEPENDENCIES)
    endif()
    if(NOT _expanded_DEPENDS_ON)
        blt_expand_depends(DEPENDS_ON ${arg_DEPENDS_ON} RESULT _expanded_DEPENDS_ON)
    endif()

    # Add dependency's information
    foreach( dependency ${_expanded_DEPENDS_ON} )
        string(TOUPPER ${dependency} uppercase_dependency )

        if ( NOT arg_OBJECT AND _BLT_${uppercase_dependency}_IS_OBJECT_LIBRARY )
            target_sources(${arg_NAME} ${_private_scope} $<TARGET_OBJECTS:${dependency}>)
        endif()

        if ( DEFINED _BLT_${uppercase_dependency}_INCLUDES )
            if ( _BLT_${uppercase_dependency}_TREAT_INCLUDES_AS_SYSTEM )
                target_include_directories( ${arg_NAME} SYSTEM ${_public_scope}
                    ${_BLT_${uppercase_dependency}_INCLUDES} )
            else()
                target_include_directories( ${arg_NAME} ${_public_scope}
                    ${_BLT_${uppercase_dependency}_INCLUDES} )
            endif()
        endif()

        if ( DEFINED _BLT_${uppercase_dependency}_FORTRAN_MODULES )
            target_include_directories( ${arg_NAME} ${_public_scope}
                ${_BLT_${uppercase_dependency}_FORTRAN_MODULES} )
        endif()

        if ( arg_OBJECT )
            # Object libraries need to inherit info from their CMake targets listed
            # in their LIBRARIES
            foreach( _library ${_BLT_${uppercase_dependency}_LIBRARIES} )
                if(TARGET ${_library})
                    blt_inherit_target_info(TO     ${arg_NAME}
                                            FROM   ${_library}
                                            OBJECT ${arg_OBJECT})
                endif()
            endforeach()
        endif()

        if ( arg_OBJECT OR _BLT_${uppercase_dependency}_IS_OBJECT_LIBRARY )
            # We want object libraries to inherit the vital info but not call
            # target_link_libraries() otherwise you have to install the object
            # files associated with the object library which noone wants.
            if ( TARGET ${dependency} )
                blt_inherit_target_info(TO     ${arg_NAME}
                                        FROM   ${dependency}
                                        OBJECT ${arg_OBJECT})
            endif()
        elseif (DEFINED _BLT_${uppercase_dependency}_LIBRARIES)
            # This prevents cmake from adding -l<library name> to the
            # command line for BLT registered libraries which are not
            # actual CMake targets
            if(NOT "${_BLT_${uppercase_dependency}_LIBRARIES}"
                    STREQUAL "BLT_NO_LIBRARIES" )
                target_link_libraries( ${arg_NAME} ${_public_scope}
                    ${_BLT_${uppercase_dependency}_LIBRARIES} )
            endif()
        else()
            target_link_libraries( ${arg_NAME} ${_public_scope} ${dependency} )
        endif()

        if ( DEFINED _BLT_${uppercase_dependency}_DEFINES )
            target_compile_definitions( ${arg_NAME} ${_public_scope}
                ${_BLT_${uppercase_dependency}_DEFINES} )
        endif()

        if ( DEFINED _BLT_${uppercase_dependency}_COMPILE_FLAGS )
            blt_add_target_compile_flags(TO ${arg_NAME}
                                         FLAGS ${_BLT_${uppercase_dependency}_COMPILE_FLAGS} )
        endif()

        if ( NOT arg_OBJECT AND DEFINED _BLT_${uppercase_dependency}_LINK_FLAGS )
            blt_add_target_link_flags(TO ${arg_NAME}
                                      FLAGS ${_BLT_${uppercase_dependency}_LINK_FLAGS} )
        endif()
        # Propagate the overridden linker language, if applicable
        if(TARGET ${dependency})
            # If it's an interface library CMake doesn't even allow us to query the property
            get_target_property(_dep_type ${dependency} TYPE)
            if(NOT "${_dep_type}" STREQUAL "INTERFACE_LIBRARY")
                get_target_property(_blt_link_lang ${dependency} INTERFACE_BLT_LINKER_LANGUAGE_OVERRIDE)
                # TODO: Do we need to worry about overwriting?  Should only ever be HIP or CUDA
                if(_blt_link_lang)
                    set_target_properties(${arg_NAME} PROPERTIES INTERFACE_BLT_LINKER_LANGUAGE_OVERRIDE ${_blt_link_lang})
                endif()
            endif()
        endif()
    endforeach()

endmacro(blt_setup_target)


##------------------------------------------------------------------------------
## blt_setup_cuda_target(NAME <name of target>
##                       SOURCES <list of sources>
##                       DEPENDS_ON <list of dependencies>
##                       LIBRARY_TYPE <STATIC, SHARED, OBJECT, or blank for executables>)
##------------------------------------------------------------------------------
macro(blt_setup_cuda_target)

    set(options)
    set(singleValueArgs NAME LIBRARY_TYPE)
    set(multiValueArgs SOURCES DEPENDS_ON)

    # Parse the arguments
    cmake_parse_arguments(arg "${options}" "${singleValueArgs}"
                            "${multiValueArgs}" ${ARGN} )

    # Check arguments
    if ( NOT DEFINED arg_NAME )
        message( FATAL_ERROR "Must provide a NAME argument to the 'blt_setup_cuda_target' macro")
    endif()

    if ( NOT DEFINED arg_SOURCES )
        message( FATAL_ERROR "Must provide SOURCES to the 'blt_setup_cuda_target' macro")
    endif()

    # Determine if cuda or cuda_runtime are in DEPENDS_ON
    list(FIND arg_DEPENDS_ON "cuda" _cuda_index)
    set(_depends_on_cuda FALSE)
    if(${_cuda_index} GREATER -1)
        set(_depends_on_cuda TRUE)
    endif()
    list(FIND arg_DEPENDS_ON "cuda_runtime" _cuda_runtime_index)
    set(_depends_on_cuda_runtime FALSE)
    if(${_cuda_runtime_index} GREATER -1)
        set(_depends_on_cuda_runtime TRUE)
    endif()

    if (${_depends_on_cuda_runtime} OR ${_depends_on_cuda})
        if (CUDA_LINK_WITH_NVCC)
            set_target_properties( ${arg_NAME} PROPERTIES LINKER_LANGUAGE CUDA)
            # This will be propagated up to executable targets that depend on this
            # library, which will need the HIP linker
            set_target_properties( ${arg_NAME} PROPERTIES INTERFACE_BLT_LINKER_LANGUAGE_OVERRIDE CUDA)
        endif()
    endif()

    if (${_depends_on_cuda})
        # if cuda is in depends_on, flag each file's language as CUDA
        # instead of leaving it up to CMake to decide
        # Note: we don't do this when depending on just 'cuda_runtime'
        set(_cuda_sources)
        set(_non_cuda_sources)
        blt_split_source_list_by_language(SOURCES      ${arg_SOURCES}
                                          C_LIST       _cuda_sources
                                          Fortran_LIST _non_cuda_sources)

        set_source_files_properties( ${_cuda_sources} PROPERTIES
                                     LANGUAGE CUDA)

        if (CUDA_SEPARABLE_COMPILATION)
            set_source_files_properties( ${_cuda_sources} PROPERTIES
                                         CUDA_SEPARABLE_COMPILATION ON)
            set_target_properties( ${arg_NAME} PROPERTIES
                                   CUDA_SEPARABLE_COMPILATION ON)
        endif()

        if (DEFINED arg_LIBRARY_TYPE)
            if (${arg_LIBRARY_TYPE} STREQUAL "static")
                set_target_properties( ${arg_NAME} PROPERTIES
                                       CMAKE_CUDA_CREATE_STATIC_LIBRARY ON)
            else()
                set_target_properties( ${arg_NAME} PROPERTIES
                                       CMAKE_CUDA_CREATE_STATIC_LIBRARY OFF)
            endif()
        endif()
    endif()
endmacro(blt_setup_cuda_target)

##------------------------------------------------------------------------------
## blt_cleanup_hip_globals(FROM_TARGET <target>)
##
## Needed as the SetupHIP macros (specifically, HIP_PREPARE_TARGET_COMMANDS)
## "pollutes" the global HIP_HIPCC_FLAGS with target-specific options.  This
## macro removes the target-specific generator expressions from the global flags
## which have already been copied to source-file-specific instances of the
## run_hipcc script.  Other global flags in HIP_HIPCC_FLAGS, e.g., those set by
## the user, are left untouched.
##------------------------------------------------------------------------------
macro(blt_cleanup_hip_globals)
    set(options)
    set(singleValueArgs FROM_TARGET)
    set(multiValueArgs)

    # Parse the arguments
    cmake_parse_arguments(arg "${options}" "${singleValueArgs}"
                            "${multiValueArgs}" ${ARGN} )

    # Check arguments
    if ( NOT DEFINED arg_FROM_TARGET )
        message( FATAL_ERROR "Must provide a FROM_TARGET argument to the 'blt_cleanup_hip_globals' macro")
    endif()

    # Remove the compile definitions generator expression
    # This must be copied verbatim from HIP_PREPARE_TARGET_COMMANDS,
    # which would have just added it to HIP_HIPCC_FLAGS
    set(_defines_genexpr "$<TARGET_PROPERTY:${arg_FROM_TARGET},COMPILE_DEFINITIONS>")
    set(_defines_flags_genexpr "$<$<BOOL:${_defines_genexpr}>:-D$<JOIN:${_defines_genexpr}, -D>>")
    list(REMOVE_ITEM HIP_HIPCC_FLAGS ${_defines_flags_genexpr})
endmacro(blt_cleanup_hip_globals)

##------------------------------------------------------------------------------
## blt_add_hip_library(NAME         <libname>
##                     SOURCES      [source1 [source2 ...]]
##                     HEADERS      [header1 [header2 ...]]
##                     DEPENDS_ON   [dep1 ...]
##                     LIBRARY_TYPE <STATIC, SHARED, or OBJECT>
##------------------------------------------------------------------------------
macro(blt_add_hip_library)

    set(options)
    set(singleValueArgs NAME LIBRARY_TYPE)
    set(multiValueArgs SOURCES HEADERS DEPENDS_ON)

    # Parse the arguments
    cmake_parse_arguments(arg "${options}" "${singleValueArgs}"
                            "${multiValueArgs}" ${ARGN} )

    # Check arguments
    if ( NOT DEFINED arg_NAME )
        message( FATAL_ERROR "Must provide a NAME argument to the 'blt_add_hip_library' macro")
    endif()

    if ( NOT DEFINED arg_SOURCES )
        message( FATAL_ERROR "Must provide SOURCES to the 'blt_add_hip_library' macro")
    endif()

    # Determine if hip or hip_runtime are in DEPENDS_ON
    list(FIND arg_DEPENDS_ON "hip" _hip_index)
    set(_depends_on_hip FALSE)
    if(${_hip_index} GREATER -1)
        set(_depends_on_hip TRUE)
    endif()
    list(FIND arg_DEPENDS_ON "hip_runtime" _hip_runtime_index)
    set(_depends_on_hip_runtime FALSE)
    if(${_hip_runtime_index} GREATER -1)
        set(_depends_on_hip_runtime TRUE)
    endif()

    if (${_depends_on_hip})
        # if hip is in depends_on, flag each file's language as HIP
        # instead of leaving it up to CMake to decide
        # Note: we don't do this when depending on just 'hip_runtime'
        set(_hip_sources)
        set(_non_hip_sources)
        blt_split_source_list_by_language(SOURCES      ${arg_SOURCES}
                                          C_LIST       _hip_sources
                                          Fortran_LIST _non_hip_sources)

        set_source_files_properties( ${_hip_sources}
                                     PROPERTIES
                                     HIP_SOURCE_PROPERTY_FORMAT TRUE)

        hip_add_library( ${arg_NAME} ${arg_SOURCES} ${arg_LIBRARY_TYPE} )
        blt_cleanup_hip_globals(FROM_TARGET ${arg_NAME})
        # Link to the hip_runtime target so it gets pulled in by targets
        # depending on this target
        target_link_libraries(${arg_NAME} PUBLIC hip_runtime)
    else()
        add_library( ${arg_NAME} ${arg_LIBRARY_TYPE} ${arg_SOURCES} ${arg_HEADERS} )
    endif()

    if (${_depends_on_hip_runtime} OR ${_depends_on_hip})
        # This will be propagated up to executable targets that depend on this
        # library, which will need the HIP linker
        set_target_properties( ${arg_NAME} PROPERTIES INTERFACE_BLT_LINKER_LANGUAGE_OVERRIDE HIP)
    endif()

endmacro(blt_add_hip_library)

##------------------------------------------------------------------------------
## blt_add_hip_executable(NAME         <libname>
##                        SOURCES      [source1 [source2 ...]]
##                        HEADERS      [header1 [header2 ...]]
##                        DEPENDS_ON   [dep1 ...]
##------------------------------------------------------------------------------
macro(blt_add_hip_executable)

    set(options)
    set(singleValueArgs NAME)
    set(multiValueArgs HEADERS SOURCES DEPENDS_ON)

    # Parse the arguments
    cmake_parse_arguments(arg "${options}" "${singleValueArgs}"
                            "${multiValueArgs}" ${ARGN} )

    # Check arguments
    if ( NOT DEFINED arg_NAME )
        message( FATAL_ERROR "Must provide a NAME argument to the 'blt_add_hip_executable' macro")
    endif()

    if ( NOT DEFINED arg_SOURCES )
        message( FATAL_ERROR "Must provide SOURCES to the 'blt_add_hip_executable' macro")
    endif()

    # Determine if hip or hip_runtime are in DEPENDS_ON
    list(FIND arg_DEPENDS_ON "hip" _hip_index)
    set(_depends_on_hip FALSE)
    if(${_hip_index} GREATER -1)
        set(_depends_on_hip TRUE)
    endif()
    list(FIND arg_DEPENDS_ON "hip_runtime" _hip_runtime_index)
    set(_depends_on_hip_runtime FALSE)
    if(${_hip_runtime_index} GREATER -1)
        set(_depends_on_hip_runtime TRUE)
    endif()

    blt_expand_depends(DEPENDS_ON ${arg_DEPENDS_ON} RESULT _expanded_DEPENDS_ON)
    foreach( dependency ${_expanded_DEPENDS_ON} )
        if(TARGET ${dependency})
            get_target_property(_dep_type ${dependency} TYPE)
            if(NOT "${_dep_type}" STREQUAL "INTERFACE_LIBRARY")
                # Propagate the overridden linker language, if applicable
                get_target_property(_blt_link_lang ${dependency} INTERFACE_BLT_LINKER_LANGUAGE_OVERRIDE)
                if(_blt_link_lang STREQUAL "HIP")
                    set(_depends_on_hip_runtime TRUE)
                endif()
            endif()
        endif()
    endforeach()

    if (${_depends_on_hip} OR ${_depends_on_hip_runtime})
        # if hip is in depends_on, flag each file's language as HIP
        # instead of leaving it up to CMake to decide
        # Note: we don't do this when depending on just 'hip_runtime'
        set(_hip_sources)
        set(_non_hip_sources)
        blt_split_source_list_by_language(SOURCES      ${arg_SOURCES}
                                          C_LIST       _hip_sources
                                          Fortran_LIST _non_hip_sources)

        set_source_files_properties( ${_hip_sources}
                                     PROPERTIES
                                     HIP_SOURCE_PROPERTY_FORMAT TRUE)

        hip_add_executable( ${arg_NAME} ${arg_SOURCES} )
        blt_cleanup_hip_globals(FROM_TARGET ${arg_NAME})
    else()
        add_executable( ${arg_NAME} ${arg_SOURCES} ${arg_HEADERS})
    endif()

    # Save the expanded dependencies to avoid recalculating later
    set_target_properties(${arg_NAME} PROPERTIES
                          BLT_EXPANDED_DEPENDENCIES "${_expanded_DEPENDS_ON}")

endmacro(blt_add_hip_executable)

##------------------------------------------------------------------------------
## blt_split_source_list_by_language( SOURCES <sources>
##                                    C_LIST <list name>
##                                    Fortran_LIST <list name>
##                                    Python_LIST <list name>)
##                                    CMAKE_LIST <list name>)
##
## Filters source list by file extension into C/C++, Fortran, Python, and
## CMake source lists based on BLT_C_FILE_EXTS, BLT_Fortran_FILE_EXTS,
## and BLT_CMAKE_FILE_EXTS (global BLT variables). Files named
## "CMakeLists.txt" are also filtered here. Files with no extension
## or generator expressions that are not object libraries (of the form
## "$<TARGET_OBJECTS:nameofobjectlibrary>") will throw fatal errors.
## ------------------------------------------------------------------------------
macro(blt_split_source_list_by_language)

    set(options)
    set(singleValueArgs C_LIST Fortran_LIST Python_LIST CMAKE_LIST)
    set(multiValueArgs SOURCES)

    # Parse the arguments
    cmake_parse_arguments(arg "${options}" "${singleValueArgs}"
                            "${multiValueArgs}" ${ARGN} )

    # Check arguments
    if ( NOT DEFINED arg_SOURCES )
        message( FATAL_ERROR "Must provide a SOURCES argument to the 'blt_split_source_list_by_language' macro" )
    endif()

    # Generate source lists based on language
    foreach(_file ${arg_SOURCES})
        # Allow CMake object libraries but disallow generator expressions
        # in source lists due to this causing all sorts of bad side effects
        if("${_file}" MATCHES "^\\$<TARGET_OBJECTS:")
            continue()
        elseif("${_file}" MATCHES "^\\$<")
            message(FATAL_ERROR "blt_split_source_list_by_language macro does not support generator expressions because CMake does not provide a way to evaluate them. Given generator expression: ${_file}")
        endif()

        get_filename_component(_ext "${_file}" EXT)
        if("${_ext}" STREQUAL "")
            message(FATAL_ERROR "blt_split_source_list_by_language given source file with no extension: ${_file}")
        endif()

        get_filename_component(_name "${_file}" NAME)  

        string(TOLOWER "${_ext}" _ext_lower)

        if("${_ext_lower}" IN_LIST BLT_C_FILE_EXTS)
            if (DEFINED arg_C_LIST)
                list(APPEND ${arg_C_LIST} "${_file}")
            endif()
        elseif("${_ext_lower}" IN_LIST BLT_Fortran_FILE_EXTS)
            if (DEFINED arg_Fortran_LIST)
                list(APPEND ${arg_Fortran_LIST} "${_file}")
            endif()
        elseif("${_ext_lower}" IN_LIST BLT_Python_FILE_EXTS)
            if (DEFINED arg_Python_LIST)
                list(APPEND ${arg_Python_LIST} "${_file}")
            endif()
        elseif("${_ext_lower}" IN_LIST BLT_CMAKE_FILE_EXTS OR "${_name}" STREQUAL "CMakeLists.txt")
            if (DEFINED arg_CMAKE_LIST)
                list(APPEND ${arg_CMAKE_LIST} "${_file}")
            endif()
        else()
            message(FATAL_ERROR "blt_split_source_list_by_language given source file with unknown file extension. Add the missing extension to the corresponding list (BLT_C_FILE_EXTS, BLT_Fortran_FILE_EXTS, BLT_Python_FILE_EXTS, or BLT_CMAKE_FILE_EXTS).\n Unknown file: ${_file}")
        endif()
    endforeach()

endmacro(blt_split_source_list_by_language)


##------------------------------------------------------------------------------
## blt_update_project_sources( TARGET_SOURCES <sources> )
##------------------------------------------------------------------------------
macro(blt_update_project_sources)

    set(options)
    set(singleValueArgs)
    set(multiValueArgs TARGET_SOURCES)

    # Parse the arguments
    cmake_parse_arguments(arg "${options}" "${singleValueArgs}"
                            "${multiValueArgs}" ${ARGN} )

    # Check arguments
    if ( NOT DEFINED arg_TARGET_SOURCES )
        message( FATAL_ERROR "Must provide target sources" )
    endif()

    ## append the target source to the all project sources
    foreach( src ${arg_TARGET_SOURCES} )
        if(IS_ABSOLUTE ${src})
            list(APPEND "${PROJECT_NAME}_ALL_SOURCES" "${src}")
        else()
            list(APPEND "${PROJECT_NAME}_ALL_SOURCES"
                "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
        endif()
    endforeach()

    set( "${PROJECT_NAME}_ALL_SOURCES" "${${PROJECT_NAME}_ALL_SOURCES}"
        CACHE STRING "" FORCE )
    mark_as_advanced("${PROJECT_NAME}_ALL_SOURCES")

endmacro(blt_update_project_sources)


##------------------------------------------------------------------------------
## blt_filter_list( TO <list_var> REGEX <string> OPERATION <string> )
##
## This macro provides the same functionality as cmake's list(FILTER )
## which is only available in cmake-3.6+.
##
## The TO argument (required) is the name of a list variable.
## The REGEX argument (required) is a string containing a regex.
## The OPERATION argument (required) is a string that defines the macro's operation.
## Supported values are "include" and "exclude"
##
## The filter is applied to the input list, which is modified in place.
##------------------------------------------------------------------------------
macro(blt_filter_list)

    set(options )
    set(singleValueArgs TO REGEX OPERATION)
    set(multiValueArgs )

    # Parse arguments
    cmake_parse_arguments(arg "${options}" "${singleValueArgs}"
                            "${multiValueArgs}" ${ARGN} )

    # Check arguments
    if( NOT DEFINED arg_TO )
        message(FATAL_ERROR "blt_filter_list macro requires a TO <list> argument")
    endif()

    if( NOT DEFINED arg_REGEX )
        message(FATAL_ERROR "blt_filter_list macro requires a REGEX <string> argument")
    endif()

    # Ensure OPERATION argument is provided with value "include" or "exclude"
    set(_exclude)
    if( NOT DEFINED arg_OPERATION )
        message(FATAL_ERROR "blt_filter_list macro requires a OPERATION <string> argument")
    elseif(NOT arg_OPERATION MATCHES "^(include|exclude)$")
        message(FATAL_ERROR "blt_filter_list macro's OPERATION argument must be either 'include' or 'exclude'")
    else()
        if(${arg_OPERATION} MATCHES "exclude")
            set(_exclude TRUE)
        else()
            set(_exclude FALSE)
        endif()
    endif()

    # Filter the list
    set(_resultList)
    foreach(elem ${${arg_TO}})
        if(elem MATCHES ${arg_REGEX})
            if(NOT ${_exclude})
                list(APPEND _resultList ${elem})
            endif()
        else()
            if(${_exclude})
                list(APPEND _resultList ${elem})
            endif()
        endif()
    endforeach()

    # Copy result back to input list variable
    set(${arg_TO} ${_resultList})

    unset(_exclude)
    unset(_resultList)
endmacro(blt_filter_list)


##------------------------------------------------------------------------------
## blt_clean_target( TARGET <target name> )
##
## This macro removes duplicates in a small subset of target properties that are
## safe to do so.
##------------------------------------------------------------------------------
macro(blt_clean_target)

    set(options )
    set(singleValueArgs TARGET)
    set(multiValueArgs )

    # Parse arguments
    cmake_parse_arguments(arg "${options}" "${singleValueArgs}"
                            "${multiValueArgs}" ${ARGN} )

    # Properties to remove duplicates from
    set(_dup_properties
        INCLUDE_DIRECTORIES
        INTERFACE_COMPILE_DEFINITIONS
        INTERFACE_INCLUDE_DIRECTORIES
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)

    foreach(_prop ${_dup_properties})
        get_target_property(_values ${arg_TARGET} ${_prop})
        if ( _values )
            list(REMOVE_DUPLICATES _values)
            set_property(TARGET ${arg_TARGET} PROPERTY ${_prop} ${_values})
        endif()
    endforeach()

endmacro(blt_clean_target)
