#include <camp/camp.hpp>

int main(int, char *[])
{
  auto t = camp::tuple<int, double>{};
  // PASS_REGEX: index out of range
  camp::get<5>(t) = 5;
  return 0;
}
