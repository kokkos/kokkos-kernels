#include <camp/camp.hpp>

using namespace camp;
CAMP_CHECK_TSAME((filter<std::is_pointer, list<int, float*, double, short*>>),
                 (list<float*, short*>));
