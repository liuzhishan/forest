#pragma once

#include <unistd.h>

namespace sniper {

#define SNIPER_NOT_COPYABLE_AND_MOVABLE(ClassName) \
  ClassName(const ClassName&) = delete;            \
  void operator=(const ClassName&) = delete;       \
  ClassName(ClassName&&) = delete;                 \
  void operator=(ClassName&&) = delete;

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

}  // namespace sniper
