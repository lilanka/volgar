#include <random>

namespace Falcon {

int randomInt(int min, int max) {
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<std::mt19937::result_type> dist6(min, max);
  return dist6(rng);
}

} // namespace falcon
