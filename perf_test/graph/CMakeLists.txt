KOKKOSKERNELS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
KOKKOSKERNELS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

KOKKOSKERNELS_ADD_EXECUTABLE(
  graph_color
  SOURCES KokkosGraph_color.cpp       
  )

KOKKOSKERNELS_ADD_EXECUTABLE(
  graph_color_d2
  SOURCES KokkosGraph_color_d2.cpp       
  )

  KOKKOSKERNELS_ADD_EXECUTABLE(
    graph_loadbalance
    SOURCES KokkosGraph_loadbalance.cpp      
    )

KOKKOSKERNELS_ADD_EXECUTABLE(
  graph_mis_d2
  SOURCES KokkosGraph_mis_d2.cpp       
  )

KOKKOSKERNELS_ADD_EXECUTABLE(
  graph_triangle
  SOURCES KokkosGraph_triangle.cpp      
  )

KOKKOSKERNELS_ADD_EXECUTABLE(
 graph_merge
 SOURCES KokkosGraph_merge.cpp      
 )
