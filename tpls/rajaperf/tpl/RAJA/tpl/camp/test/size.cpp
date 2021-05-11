#include <camp/camp.hpp>

using namespace camp;
CAMP_CHECK_IEQ((size<list<int>>), (1));
CAMP_CHECK_IEQ((size<list<int, int>>), (2));
CAMP_CHECK_IEQ((size<list<int, int, int>>), (3));


CAMP_CHECK_IEQ((size<idx_seq<0>>), (1));
CAMP_CHECK_IEQ((size<idx_seq<0, 0>>), (2));
CAMP_CHECK_IEQ((size<idx_seq<0, 0, 0>>), (3));
