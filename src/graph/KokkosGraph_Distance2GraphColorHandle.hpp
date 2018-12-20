/*
//@HEADER
// ************************************************************************
//
//               KokkosKernels 0.9: Linear Algebra and Graph Kernels
//                 Copyright 2017 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/
#include <fstream>
#include <ostream>

#include <KokkosKernels_Utils.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_MemoryTraits.hpp>

#ifndef _DISTANCE2GRAPHCOLORHANDLE_HPP
#define _DISTANCE2GRAPHCOLORHANDLE_HPP

//#define VERBOSE
namespace KokkosGraph {

//enum ColoringAlgorithm
enum GraphColoringAlgorithmDistance2
{
    COLORING_D2_DEFAULT,             // Distance-2 Graph Coloring default algorithm
    COLORING_D2_MATRIX_SQUARED,      // Distance-2 Graph Coloring using Matrix Squared + D1 Coloring
    COLORING_D2_SERIAL,              // Distance-2 Graph Coloring (SERIAL)
    COLORING_D2,                     // Distance-2 Graph Coloring
    COLORING_D2_VB,                  // Distance-2 Graph Coloring Vertex Based
    COLORING_D2_VB_BIT,              // Distance-2 Graph Coloring Vertex Based BIT
    COLORING_D2_VB_BIT_EF,           // Distance-2 Graph Coloring Vertex Based BIT + Edge Filtering
};


enum ConflictList           // TODO Can this go away?
{
    COLORING_NOCONFLICT,
    COLORING_ATOMIC,
    COLORING_PPS
};

enum ColoringType
{
    Distance1,
    Distance2
};

template<class size_type_, class color_t_, class lno_t_, class ExecutionSpace, class TemporaryMemorySpace, class PersistentMemorySpace>
class Distance2GraphColoringHandle
{

  public:
    typedef ExecutionSpace        HandleExecSpace;
    typedef TemporaryMemorySpace  HandleTempMemorySpace;
    typedef PersistentMemorySpace HandlePersistentMemorySpace;


    typedef typename std::remove_const<size_type_>::type size_type;
    typedef const size_type                              const_size_type;

    typedef typename std::remove_const<lno_t_>::type nnz_lno_t;
    typedef const nnz_lno_t                          const_nnz_lno_t;

    typedef typename std::remove_const<color_t_>::type color_t;
    typedef const color_t                              const_color_t;

    typedef typename Kokkos::View<color_t *, HandlePersistentMemorySpace> color_view_t;

    typedef typename color_view_t::array_layout  color_view_array_layout;
    typedef typename color_view_t::device_type   color_view_device_t;
    typedef typename color_view_t::memory_traits color_view_memory_traits;
    typedef typename color_view_t::HostMirror    color_host_view_t;      // Host view type


    typedef typename Kokkos::View<size_type *, HandleTempMemorySpace>       size_type_temp_work_view_t;
    typedef typename Kokkos::View<size_type *, HandlePersistentMemorySpace> size_type_persistent_work_view_t;

    typedef typename size_type_persistent_work_view_t::HostMirror size_type_persistent_work_host_view_t;      // Host view type

    typedef typename Kokkos::View<nnz_lno_t *, HandleTempMemorySpace>       nnz_lno_temp_work_view_t;
    typedef typename Kokkos::View<nnz_lno_t *, HandlePersistentMemorySpace> nnz_lno_persistent_work_view_t;
    typedef typename nnz_lno_persistent_work_view_t::HostMirror             nnz_lno_persistent_work_host_view_t;      // Host view type

    typedef Kokkos::TeamPolicy<HandleExecSpace> team_policy_t;
    typedef typename team_policy_t::member_type team_member_t;

    typedef typename Kokkos::View<size_t *> non_const_1d_size_type_view_t;

  private:
    ColoringType GraphColoringType;

    // Parameters
    ColoringAlgorithm coloring_algorithm_type;      // VB, VBBIT, VBCS, VBD or EB.
    ConflictList      conflict_list_type;           // whether to use a conflict list or not, and
                                                    // if using it wheter to create it with atomic or parallel prefix sum.

    double min_reduction_for_conflictlist;
    // if used pps is selected to create conflict list, what min percantage should be the vertex list
    // be reduced, to create the new vertexlist. If it is reduced less than this percantage, use the
    // previous array.

    int min_elements_for_conflictlist;
    // minimum number of elements to create a new conflict list.
    // if current conflict list size is smaller than this number,
    // than we do not need to create a new conflict list.

    bool serial_conflict_resolution;      // perform parallel greedy coloring once, then resolve conflict serially.
    bool tictoc;                          // print time at every step

    bool vb_edge_filtering;               // whether to do edge filtering or not in vertex based algorithms. Swaps on the ad error.

    int vb_chunk_size;                   // the (minimum) size of the consecutive works that a thread will be assigned to.
    int max_number_of_iterations;        // maximum allowed number of phases that

    int eb_num_initial_colors;           // the number of colors to assign at the beginning of the edge-based algorithm


    // STATISTICS
    double overall_coloring_time;             // the overall time that it took to color the graph. In the case of the iterative calls.
    double overall_coloring_time_phase1;      //
    double overall_coloring_time_phase2;      //
    double overall_coloring_time_phase3;      // Some timer accumulators for internal phases.
    double overall_coloring_time_phase4;      //
    double overall_coloring_time_phase5;      //
    double coloring_time;                     // the time that it took to color the graph

    int num_phases;                           // Number of phases used by the coloring algorithm


    size_type                      size_of_edge_list;       // todo not used?

    color_view_t vertex_colors;
    bool         is_coloring_called_before;
    nnz_lno_t    num_colors;



  public:


    /**
     * \brief Default constructor.
     */
    Distance2GraphColoringHandle()
        : GraphColoringType(Distance1)
        , coloring_algorithm_type(COLORING_DEFAULT)
        , conflict_list_type(COLORING_ATOMIC)
        , min_reduction_for_conflictlist(0.35)
        , min_elements_for_conflictlist(1000 /*5000*/)
        , serial_conflict_resolution(true)
        , tictoc(false)
        , vb_edge_filtering(false)
        , vb_chunk_size(8)
        , max_number_of_iterations(200)
        , eb_num_initial_colors(1)
        , overall_coloring_time(0)
        , overall_coloring_time_phase1(0)
        , overall_coloring_time_phase2(0)
        , overall_coloring_time_phase3(0)
        , overall_coloring_time_phase4(0)
        , overall_coloring_time_phase5(0)
        , coloring_time(0)
        , num_phases(0)
        , size_of_edge_list(0)
        , vertex_colors()
        , is_coloring_called_before(false)
        , num_colors(0)
    {
        this->choose_default_algorithm();
        this->set_defaults(this->coloring_algorithm_type);
    }

    /** \brief Sets the graph coloring type. Whether it is distance-1 or distance-2 coloring.
     *  \param col_type: Coloring Type: KokkosKernels::Experimental::Graph::ColoringType which can be
     *        either KokkosKernels::Experimental::Graph::Distance1 or KokkosKernels::Experimental::Graph::Distance2
     */
    void set_coloring_type(const ColoringType &col_type) { this->GraphColoringType = col_type; }

    /** \brief Gets the graph coloring type. Whether it is distance-1 or distance-2 coloring.
     *  returns Coloring Type: KokkosKernels::Experimental::Graph::ColoringType which can be
     *        either KokkosKernels::Experimental::Graph::Distance1 or KokkosKernels::Experimental::Graph::Distance2
     */
    ColoringType get_coloring_type() { return this->GraphColoringType; }


    /** \brief Changes the graph coloring algorithm.
     *  \param col_algo: Coloring algorithm: one of COLORING_VB, COLORING_VBBIT, COLORING_VBCS, COLORING_EB
     *  \param set_default_parameters: whether or not to reset the default parameters for the given algorithm.
     */
    void set_algorithm(const ColoringAlgorithm &col_algo, bool set_default_parameters = true)
    {
        if(col_algo == COLORING_DEFAULT)
        {
            this->choose_default_algorithm();
        }
        else
        {
            this->coloring_algorithm_type = col_algo;
        }
        if(set_default_parameters)
        {
            this->set_defaults(this->coloring_algorithm_type);
        }
    }



    /** \brief Chooses best algorithm based on the execution space. COLORING_EB if cuda, COLORING_VB otherwise.
     */
    void choose_default_algorithm()
    {
        #if defined(KOKKOS_ENABLE_SERIAL)
        if(Kokkos::Impl::is_same<Kokkos::Serial, ExecutionSpace>::value)
        {
            this->coloring_algorithm_type = COLORING_SERIAL;
            #ifdef VERBOSE
            std::cout << "Serial Execution Space, Default Algorithm: COLORING_VB" << std::endl;
            #endif
        }
        #endif

        #if defined(KOKKOS_ENABLE_THREADS)
        if(Kokkos::Impl::is_same<Kokkos::Threads, ExecutionSpace>::value)
        {
            this->coloring_algorithm_type = COLORING_VB;
            #ifdef VERBOSE
            std::cout << "PTHREAD Execution Space, Default Algorithm: COLORING_VB" << std::endl;
            #endif
        }
        #endif

        #if defined(KOKKOS_ENABLE_OPENMP)
        if(Kokkos::Impl::is_same<Kokkos::OpenMP, ExecutionSpace>::value)
        {
            this->coloring_algorithm_type = COLORING_VB;
            #ifdef VERBOSE
            std::cout << "OpenMP Execution Space, Default Algorithm: COLORING_VB" << std::endl;
            #endif
        }
        #endif

        #if defined(KOKKOS_ENABLE_CUDA)
        if(Kokkos::Impl::is_same<Kokkos::Cuda, ExecutionSpace>::value)
        {
            this->coloring_algorithm_type = COLORING_EB;
            #ifdef VERBOSE
            std::cout << "Cuda Execution Space, Default Algorithm: COLORING_VB" << std::endl;
            #endif
        }
        #endif

        #if defined(KOKKOS_ENABLE_QTHREAD)
        if(Kokkos::Impl::is_same<Kokkos::Qthread, ExecutionSpace>::value)
        {
            this->coloring_algorithm_type = COLORING_VB;
            #ifdef VERBOSE
            std::cout << "Qthread Execution Space, Default Algorithm: COLORING_VB" << std::endl;
            #endif
        }
        #endif
    }



    nnz_lno_t get_num_colors()
    {
        if(num_colors == 0)
        {
            typedef typename Kokkos::RangePolicy<ExecutionSpace> my_exec_space;
            Kokkos::parallel_reduce("KokkosKernels::FindMax", my_exec_space(0, vertex_colors.extent(0)), ReduceMaxFunctor(vertex_colors), num_colors);
        }
        return num_colors;
    }



    /** \brief Sets Default Parameter settings for the given algorithm.
     */
    void set_defaults(const ColoringAlgorithm &col_algo)
    {
        switch(col_algo)
        {
            case COLORING_D2_MATRIX_SQUARED:
            case COLORING_D2_SERIAL:
            case COLORING_D2:
            case COLORING_D2_VB:
            case COLORING_D2_VB_BIT:
            case COLORING_D2_VB_BIT_EF:
                this->conflict_list_type             = COLORING_ATOMIC;
                this->min_reduction_for_conflictlist = 0.35;
                this->min_elements_for_conflictlist  = 1000;
                this->serial_conflict_resolution     = false;
                this->tictoc                         = false;
                this->vb_edge_filtering              = false;
                this->vb_chunk_size                  = 8;
                this->max_number_of_iterations       = 200;
                this->eb_num_initial_colors          = 1;
                break;
            default:
                throw std::runtime_error("Unknown Distance-2 Graph Coloring Algorithm\n");
                // break;
        }
    }


    /**
     * \brief Destructor
     */
    virtual ~Distance2GraphColoringHandle(){};

    // getters
    ColoringAlgorithm get_coloring_algo_type() const { return this->coloring_algorithm_type; }
    ConflictList      get_conflict_list_type() const { return this->conflict_list_type; }
    double            get_min_reduction_for_conflictlist() const { return this->min_reduction_for_conflictlist; }
    int               get_min_elements_for_conflictlist() const { return this->min_elements_for_conflictlist; }
    bool              get_serial_conflict_resolution() const { return this->serial_conflict_resolution; }
    bool              get_tictoc() const { return this->tictoc; }
    bool              get_vb_edge_filtering() const { return this->vb_edge_filtering; }
    int               get_vb_chunk_size() const { return this->vb_chunk_size; }
    int               get_max_number_of_iterations() const { return this->max_number_of_iterations; }
    int               get_eb_num_initial_colors() const { return this->eb_num_initial_colors; }

    double       get_overall_coloring_time() const { return this->overall_coloring_time; }
    double       get_overall_coloring_time_phase1() const { return this->overall_coloring_time_phase1; }
    double       get_overall_coloring_time_phase2() const { return this->overall_coloring_time_phase2; }
    double       get_overall_coloring_time_phase3() const { return this->overall_coloring_time_phase3; }
    double       get_overall_coloring_time_phase4() const { return this->overall_coloring_time_phase4; }
    double       get_overall_coloring_time_phase5() const { return this->overall_coloring_time_phase5; }
    double       get_coloring_time() const { return this->coloring_time; }
    int          get_num_phases() const { return this->num_phases; }
    color_view_t get_vertex_colors() const { return this->vertex_colors; }
    bool         is_coloring_called() const { return this->is_coloring_called_before; }

    // setters
    void set_coloring_algo_type(const ColoringAlgorithm &col_algo) { this->coloring_algorithm_type = col_algo; }
    void set_conflict_list_type(const ConflictList &cl) { this->conflict_list_type = cl; }
    void set_min_reduction_for_conflictlist(const double &min_reduction) { this->min_reduction_for_conflictlist = min_reduction; }
    void set_min_elements_for_conflictlist(const int &min_elements) { this->min_elements_for_conflictlist = min_elements; }
    void set_serial_conflict_resolution(const bool &use_serial_conflist_resolution) { this->serial_conflict_resolution = use_serial_conflist_resolution; }
    void set_tictoc(const bool use_tictoc) { this->tictoc = use_tictoc; }
    void set_vb_edge_filtering(const bool &use_vb_edge_filtering) { this->vb_edge_filtering = use_vb_edge_filtering; }
    void set_vb_chunk_size(const int &chunksize) { this->vb_chunk_size = chunksize; }
    void set_max_number_of_iterations(const int &max_phases) { this->max_number_of_iterations = max_phases; }
    void set_eb_num_initial_colors(const int &num_initial_colors) { this->eb_num_initial_colors = num_initial_colors; }
    void add_to_overall_coloring_time(const double &coloring_time_) { this->overall_coloring_time += coloring_time_; }
    void add_to_overall_coloring_time_phase1(const double &coloring_time_) { this->overall_coloring_time_phase1 += coloring_time_; }
    void add_to_overall_coloring_time_phase2(const double &coloring_time_) { this->overall_coloring_time_phase2 += coloring_time_; }
    void add_to_overall_coloring_time_phase3(const double &coloring_time_) { this->overall_coloring_time_phase3 += coloring_time_; }
    void add_to_overall_coloring_time_phase4(const double &coloring_time_) { this->overall_coloring_time_phase4 += coloring_time_; }
    void add_to_overall_coloring_time_phase5(const double &coloring_time_) { this->overall_coloring_time_phase5 += coloring_time_; }
    void set_coloring_time(const double &coloring_time_) { this->coloring_time = coloring_time_; }
    void set_num_phases(const double &num_phases_) { this->num_phases = num_phases_; }

    void set_vertex_colors(const color_view_t vertex_colors_)
    {
        this->vertex_colors             = vertex_colors_;
        this->is_coloring_called_before = true;
        this->num_colors                = 0;
    }


    // Print / write out the graph in a GraphVIZ format.
    // Color "1" will be rendered as a red circle.
    // Color "0" (uncolored) will be a star shape.
    // Other node colors shown as just the color value.
    //
    // @param os use std::cout for output to STDOUT stream, or a ofstream object
    //           (i.e., `std::ofstream os("G.dot", std::ofstream::out);`) to write
    //           to a file.
    template<typename kokkos_view_t, typename index_t, typename rowmap_t, typename entries_t>
    void graphToGraphviz(std::ostream &os, const index_t num_verts, rowmap_t &rowmap, entries_t &entries, kokkos_view_t &colors) const
    {
        typedef typename kokkos_view_t::HostMirror h_colors_t;
        typedef typename rowmap_t::HostMirror      h_rowmap_t;
        typedef typename entries_t::HostMirror     h_entries_t;

        h_colors_t h_colors = Kokkos::create_mirror_view(colors);
        Kokkos::deep_copy(h_colors, colors);

        h_rowmap_t h_rowmap = Kokkos::create_mirror_view(rowmap);
        Kokkos::deep_copy(h_rowmap, rowmap);

        h_entries_t h_entries = Kokkos::create_mirror_view(entries);
        Kokkos::deep_copy(h_entries, entries);


        os << "Graph G" << std::endl
           << "{" << std::endl
           << "    rankdir=LR;" << std::endl
           << "    overlap=false;" << std::endl
           << "    splines=true;" << std::endl
           << "    maxiter=2000;" << std::endl
           << "    node [shape=Mrecord];" << std::endl
           << "    edge [len=2.0];" << std::endl
           << std::endl;

        for(index_t vid = 0; vid < num_verts; vid++)
        {
            if(1 == h_colors(vid))
            {
                os << "    " << vid << " [ label=\"\", shape=circle, style=filled, color=red, fillcolor=red];" << std::endl;
            }
            else if(0 == h_colors(vid))
            {
                os << "    " << vid << " [ label=\"" << vid << "\", shape=star, style=filled, color=black];" << std::endl;
            }
            else
            {
                os << "    " << vid << " [ label=\"" << vid << "|" << h_colors(vid) << "\"];" << std::endl;
            }
            for(index_t iadj = h_rowmap(vid); iadj < h_rowmap(vid + 1); iadj++)
            {
                index_t vadj = h_entries(iadj);
                if(vadj >= vid)
                {
                    os << "    " << vid << " -- " << vadj << ";" << std::endl;
                }
            }
            os << std::endl;
        }
        os << "}" << std::endl;
    }      // graphToGraphviz (end)
};

}      // namespace KokkosGraph

#endif      // _GRAPHCOLORHANDLE_HPP
