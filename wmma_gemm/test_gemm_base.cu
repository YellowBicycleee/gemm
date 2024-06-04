#include <mma.h>

#include <cmath>
#include <cstdio>

using namespace nvcuda;

constexpr int WARP_SIZE = 32;

namespace fp64 {
constexpr int WMMA_M = 8;
constexpr int WMMA_N = 8;
constexpr int WMMA_K = 4;

constexpr int BLOCK_ROW_TILES = 1;
constexpr int BLOCK_COL_TILES = 1;
}  // namespace fp64