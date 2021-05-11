#include <camp/camp.hpp>

using namespace camp;
CAMP_CHECK_TSAME((accumulate<append, list<>, list<int, float, double>>),
                 (list<int, float, double>));
CAMP_CHECK_TSAME((cartesian_product<list<int>, list<float>>),
                 (list<list<int, float>>));
struct a;
struct b;
struct c;
struct d;
struct e;
struct f;
struct g;
CAMP_CHECK_TSAME((cartesian_product<list<a, b>, list<c, d, e>>),
                 (list<list<a, c>,
                       list<a, d>,
                       list<a, e>,
                       list<b, c>,
                       list<b, d>,
                       list<b, e>>));
CAMP_CHECK_TSAME((cartesian_product<list<a, b>, list<c, d, e>, list<f, g>>),
                 (camp::list<camp::list<a, c, f>,
                             camp::list<a, c, g>,
                             camp::list<a, d, f>,
                             camp::list<a, d, g>,
                             camp::list<a, e, f>,
                             camp::list<a, e, g>,
                             camp::list<b, c, f>,
                             camp::list<b, c, g>,
                             camp::list<b, d, f>,
                             camp::list<b, d, g>,
                             camp::list<b, e, f>,
                             camp::list<b, e, g>>));
