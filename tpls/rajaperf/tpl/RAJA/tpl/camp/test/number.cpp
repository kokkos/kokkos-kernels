#include <camp/camp.hpp>

using namespace camp;
CAMP_CHECK_TSAME((make_idx_seq_t<3>), (idx_seq<0, 1, 2>));
CAMP_CHECK_TSAME((make_idx_seq_t<2>), (idx_seq<0, 1>));
CAMP_CHECK_TSAME((make_idx_seq_t<1>), (idx_seq<0>));
CAMP_CHECK_TSAME((make_idx_seq_t<0>), (idx_seq<>));
