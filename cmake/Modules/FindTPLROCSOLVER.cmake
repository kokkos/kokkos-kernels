FIND_PACKAGE(ROCSOLVER)
if(TARGET roc::rocsolver)
## MPL: 12/29/2022: Variable TPL_ROCSOLVER_IMPORTED_NAME follows the requested convention
## of KokkosKernel (method kokkoskernels_import_tpl of kokkoskernels_tpls.cmake)
  SET(TPL_ROCSOLVER_IMPORTED_NAME roc::rocsolver)
  SET(TPL_IMPORTED_NAME roc::rocsolver)
## MPL: 12/29/2022: A target comming from a TPL must follows the requested convention
## of KokkosKernel (method kokkoskernels_link_tpl of kokkoskernels_tpls.cmake)
  ADD_LIBRARY(KokkosKernels::ROCSOLVER ALIAS roc::rocsolver)
ELSE()
  MESSAGE(FATAL_ERROR "Package ROCSOLVER requested but not found")
ENDIF()
