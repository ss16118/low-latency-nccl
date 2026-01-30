/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "nccl_device/ll_buffer.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>

/* --------------------------------------------------------------------------
 * Datatype -> size in bytes
 * -------------------------------------------------------------------------- */

 static inline size_t ncclDataTypeSize(ncclDataType_t type) {
  switch (type) {
      /* 1-byte types */
      case ncclInt8:      // ncclChar is an alias
      case ncclUint8:
      case ncclFloat8e4m3:
      case ncclFloat8e5m2:
          return 1;

      /* 2-byte types */
      case ncclFloat16:   // ncclHalf is an alias
      case ncclBfloat16:
          return 2;

      /* 4-byte types */
      case ncclInt32:     // ncclInt is an alias
      case ncclUint32:
      case ncclFloat32:   // ncclFloat is an alias
          return 4;

      /* 8-byte types */
      case ncclInt64:
      case ncclUint64:
      case ncclFloat64:   // ncclDouble is an alias
          return 8;

      default:
          return 0;  // unknown / future type
  }
}


// ==================== Host-Side Poison Value Table ====================
// Maps NCCL data types to their 32-bit poison patterns

static uint32_t getPoisonBits32(ncclDataType_t dtype) {
  size_t type_size = ncclDataTypeSize(dtype);
  if (type_size == 1) {
    return ncclLLBufferPoisonTraits<int8_t>::poisonBits32;
  } else if (type_size == 2) {
    return ncclLLBufferPoisonTraits<int16_t>::poisonBits32;
  } else if (type_size == 4) {
    return ncclLLBufferPoisonTraits<int32_t>::poisonBits32;
  } else if (type_size == 8) {
    return ncclLLBufferPoisonTraits<int64_t>::poisonBits32;
  } else {
    // Use the 32 bit poison value by default
    return ncclLLBufferPoisonTraits<int32_t>::poisonBits32;
  }
}

// ==================== Host-Side API Implementation ====================

NCCL_API(size_t, ncclCalcMinReductionBufferSize, ncclLLSyncMode_t mode, int nElts, int eltSize, int nRanks, int nBlocks, int roundRobinFactor);
size_t ncclCalcMinReductionBufferSize(ncclLLSyncMode_t mode, int nElts, int eltSize, int nRanks, int nBlocks, int roundRobinFactor) {
  // Calculate slot size based on mode
  size_t singleBufferSize = nElts * eltSize * nRanks;
  singleBufferSize = mode == ncclLL ? singleBufferSize * 2 : singleBufferSize;
  return singleBufferSize * roundRobinFactor * nBlocks;
}

NCCL_API(uint32_t, ncclLLGetPoisonValue, ncclDataType_t dtype);
uint32_t ncclLLGetPoisonValue(ncclDataType_t dtype) {
  return getPoisonBits32(dtype);
}

// ==================== CUDA Kernel for Poisoning ====================

__global__ void ncclLLPoisonBufferKernel(uint32_t* buffer, size_t nWords, uint32_t poisonValue) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = idx; i < nWords; i += stride) {
    buffer[i] = poisonValue;
  }
}

NCCL_API(ncclResult_t, ncclLLPoisonBuffer, void* buffer, size_t sizeBytes, ncclDataType_t dtype);
ncclResult_t ncclLLPoisonBuffer(
    void* buffer,
    size_t sizeBytes,
    ncclDataType_t dtype
) {
  if (buffer == nullptr || sizeBytes == 0) {
    return ncclSuccess;
  }

  // Size must be a multiple of 4 bytes for 32-bit writes
  if (sizeBytes % 4 != 0) {
    return ncclInvalidArgument;
  }

  uint32_t poisonValue = getPoisonBits32(dtype);
  size_t nWords = sizeBytes / 4;

  // Launch kernel with reasonable block/grid configuration
  int blockSize = 256;
  int numBlocks = (int)((nWords + blockSize - 1) / blockSize);
  numBlocks = std::min(numBlocks, 256);  // Cap at 256 blocks

  ncclLLPoisonBufferKernel<<<numBlocks, blockSize>>>(
    (uint32_t*)buffer, nWords, poisonValue);

  return ncclSuccess;
}

// ==================== Stream-aware version ====================
// Users can call this version directly for stream control

NCCL_API(ncclResult_t, ncclLLPoisonBufferAsync, void* buffer, size_t sizeBytes, ncclDataType_t dtype, cudaStream_t stream);
ncclResult_t ncclLLPoisonBufferAsync(
    void* buffer,
    size_t sizeBytes,
    ncclDataType_t dtype,
    cudaStream_t stream
) {
  if (buffer == nullptr || sizeBytes == 0) {
    return ncclSuccess;
  }

  // Size must be a multiple of 4 bytes for 32-bit writes
  if (sizeBytes % 4 != 0) {
    return ncclInvalidArgument;
  }

  uint32_t poisonValue = getPoisonBits32(dtype);
  size_t nWords = sizeBytes / 4;

  // Launch kernel with reasonable block/grid configuration
  int blockSize = 256;
  int numBlocks = (int)((nWords + blockSize - 1) / blockSize);
  numBlocks = std::min(numBlocks, 256);  // Cap at 256 blocks

  ncclLLPoisonBufferKernel<<<numBlocks, blockSize, 0, stream>>>(
    (uint32_t*)buffer, nWords, poisonValue);

  return ncclSuccess;
}
