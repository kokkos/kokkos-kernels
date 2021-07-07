#include <camp/camp.hpp>

using namespace camp;
CAMP_CHECK_TSAME((index_of<int, list<>>), (nil));
CAMP_CHECK_TSAME((index_of<int, list<float, double, int>>), (num<2>));
CAMP_CHECK_TSAME((index_of<int, list<float, double, int, int, int, int>>),
                 (num<2>));
