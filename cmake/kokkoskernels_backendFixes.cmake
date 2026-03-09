
# On recent AMD GPUs (gfx1000+), the following error can occur 
# for builds with optimisation level 0 (-O0, like debug builds):
#   error: Illegal instruction detected: 
#   Invalid dpp_ctrl value: wavefront shifts are not supported on GFX10+
# The ROCm team is aware, but an upstream fix may be far off.
# See https://github.com/ROCm/ROCm/issues/5826
# The recommended workaround is to replace -O0 with -Og,
# but since that can negatively affect debugging in some instances
# (see, for example, https://stackoverflow.com/q/63386189),
# we restrict this change specifically to the affected architectures.
if(KOKKOS_ENABLE_HIP)
	if(Kokkos_ARCH MATCHES "AMD_GFX([0-9]+).*")
		# the issue affects "gfx10+" according to the error message
		if(CMAKE_MATCH_1 GREATER 1000)
			message(STATUS "Detected gfx10+ arch (${Kokkos_ARCH}).")
			message(STATUS "Appending -Og -g to debug build flags (see https://github.com/ROCm/ROCm/issues/5826).")
			set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og -g")
		endif()
	endif()
endif()