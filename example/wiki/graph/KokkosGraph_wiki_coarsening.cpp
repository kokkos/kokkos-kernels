#include "KokkosGraph_wiki_9pt_stencil.hpp"
#include "KokkosGraph_MIS2.hpp"

int main(int argc, char* argv[])
{
  Kokkos::initialize();
  {
    using GraphDemo::numVertices;
    RowmapType rowmapDevice;
    ColindsType colindsDevice;
    //Step 1: Generate the graph on host, allocate space on device, and copy.
    //See function "generate9pt" below.
    GraphDemo::generate9pt(rowmapDevice, colindsDevice);
    //Step 2: Run MIS-2 based coarsening and print the result
    {
      Ordinal numClusters = 0;
      auto labels = KokkosGraph::Experimental::graph_mis2_aggregate<ExecSpace, RowmapType, ColindsType>(
          rowmapDevice, colindsDevice, numClusters);
      std::cout << "Coarsened into " << numClusters << " clusters:\n";
      //coarsening labels can be printed in the same way as colors
      GraphDemo::printColoring(labels, numClusters);
      putchar('\n');
    }
  }
  Kokkos::finalize();
  return 0;
}

