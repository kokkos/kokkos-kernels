#include <camp/camp.hpp>

using namespace camp;
CAMP_CHECK_TSAME((invoke_l<bind<list, _1, int, _2>, float, double>),
                 (list<float, int, double>));
CAMP_CHECK_TSAME((invoke_l<bind_front<list, int>, float, double>),
                 (list<int, float, double>));
