#include <camp/camp.hpp>

using namespace camp;
using tl1 = list<list<int, num<0>>, list<char, num<1>>>;
CAMP_CHECK_TSAME((at_key<tl1, int>), (num<0>));
CAMP_CHECK_TSAME((at_key<tl1, char>), (num<1>));
CAMP_CHECK_TSAME((at_key<tl1, bool>), (nil));
