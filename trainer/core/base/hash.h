#pragma once

#include <unistd.h>

namespace sniper {

static uint64_t djb2_hash64(uint64_t key) {
  const unsigned char *p = reinterpret_cast<unsigned char *>(&key);
  uint64_t hash = 5381;
  for (size_t i = 0; i < sizeof(uint64_t); ++i) {
    hash = ((hash << 5ULL) + hash) + p[i];
  }
  return hash;
}

static uint64_t murmur3_hash64(uint64_t key) {
  key ^= key >> 33;
  key *= 0xff51afd7ed558ccd;
  key ^= key >> 33;
  key *= 0xc4ceb9fe1a85ec53;
  key ^= key >> 33;
  return key;
}

}  // namespace sniper
