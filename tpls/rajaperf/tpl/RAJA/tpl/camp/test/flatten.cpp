#include <camp/camp.hpp>

using namespace camp;
CAMP_CHECK_TSAME((flatten<list<>>), (list<>));
CAMP_CHECK_TSAME((flatten<list<int>>), (list<int>));
CAMP_CHECK_TSAME((flatten<list<list<int>>>), (list<int>));
CAMP_CHECK_TSAME((flatten<list<list<list<int>>>>), (list<int>));
CAMP_CHECK_TSAME((flatten<list<float, list<int, double>, list<list<int>>>>),
    (list<float, int, double, int>));

