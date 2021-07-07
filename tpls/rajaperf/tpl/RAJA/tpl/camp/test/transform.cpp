#include <camp/camp.hpp>

using namespace camp;
CAMP_CHECK_TSAME((transform<std::add_cv, list<int>>),
                 (list<const volatile int>));
CAMP_CHECK_TSAME((transform<std::remove_reference, list<int&, int&>>),
                 (list<int, int>));
