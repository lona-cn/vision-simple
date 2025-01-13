#include <cassert>
#include <magic_enum.hpp>

#include "VisionSimpleConfig.h"

using namespace vision_simple;

int main(int argc, char* argv[]) {
  const auto config = Config::Instance();
  if (!config) assert(config.error().message.c_str());
  return 0;
}
