/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_SYM_KERNELS_H_
#define NCCL_SYM_KERNELS_H_
#include "nccl.h"
#include "nccl_device.h"
#include "nccl_common.h"
#include "device.h"
#include "../device/symmetric/gin_scratch__types.h"

////////////////////////////////////////////////////////////////////////////////
// ncclSymk[Foo]: Kernels built on the device API

#define NCCL_SYM_KERNEL_CELL_SIZE 1024 // no less than 16 bytes minimal cell size
// Sets the default accumulation buffer size to 64MiB which should be enough for most operations.
#define REDUCTION_BUFFER_SIZE (64 * 1024 * 1024)

#define NCCL_LAMPORT_INT 0xFFFAFFFA

// Maximum number of slots supported is 32
constexpr int ncclSymkLamportAccumSlots = 3;

constexpr int ncclSymkMaxBlocks = 64;
constexpr int ncclSymkMaxThreads = 512;
constexpr int ncclSymkLLMaxEltSize = 8;

constexpr __host__ __device__ int ncclSymkLLMaxSlots(int eltSize = ncclSymkLLMaxEltSize) {
  return ncclSymkMaxThreads*ncclSymkLLMaxEltSize/eltSize;
}

enum ncclSymkKernelId {
  ncclSymkKernelId_AllReduce_AGxLL_R,
  ncclSymkKernelId_AllReduce_AGxLLMC_R,
  ncclSymkKernelId_AllReduce_RSxLD_AGxST,
  ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC,
  ncclSymkKernelId_AllReduce_RSxNet_ARxMC_AGxNet,
  ncclSymkKernelId_AllReduce_Lamport2Shot,
  ncclSymkKernelId_AllReduce_Lamport2ShotMC,
  ncclSymkKernelId_AllReduce_Lamport2ShotPoison,
  ncclSymkKernelId_AllReduce_Lamport1ShotV2,
  ncclSymkKernelId_AllReduce_Lamport1ShotPoison,
  ncclSymkKernelId_AllReduce_Lamport1ShotPoisonMC,
  // ncclLLBuffer-based 1-shot allreduce.
  // Selected via NCCL_SYM_LLBUFFER_SYNC:
  //   0: Poison  -> kernel id AllReduce_LLBuffer* (default)
  //   1: LL      -> kernel id AllReduce_LLBuffer_LL16*
  // Rank-specialized variants (_R4, _R8, _R16, _R32) are selected based on nRanks.
  // Base variants (no _Rn suffix) are used for other rank counts.
  ncclSymkKernelId_AllReduce_LLBuffer,
  ncclSymkKernelId_AllReduce_LLBuffer_R4,
  ncclSymkKernelId_AllReduce_LLBuffer_R8,
  ncclSymkKernelId_AllReduce_LLBuffer_R16,
  ncclSymkKernelId_AllReduce_LLBuffer_R32,
  ncclSymkKernelId_AllReduce_LLBuffer_LL16,
  ncclSymkKernelId_AllReduce_LLBuffer_LL16_R4,
  ncclSymkKernelId_AllReduce_LLBuffer_LL16_R8,
  ncclSymkKernelId_AllReduce_LLBuffer_LL16_R16,
  ncclSymkKernelId_AllReduce_LLBuffer_LL16_R32,
  // Multimem version of AllReduce_LLBuffer (uses multicast for broadcast)
  ncclSymkKernelId_AllReduce_LLBufferMC,
  // Two-shot AllReduce using ncclLLBuffer (reduce-scatter + all-gather pattern)
  // Rank-specialized variants (_R4, _R8, _R16, _R32) are selected based on nRanks.
  ncclSymkKernelId_AllReduce_LLBuffer_Twoshot,
  ncclSymkKernelId_AllReduce_LLBuffer_Twoshot_R4,
  ncclSymkKernelId_AllReduce_LLBuffer_Twoshot_R8,
  ncclSymkKernelId_AllReduce_LLBuffer_Twoshot_R16,
  ncclSymkKernelId_AllReduce_LLBuffer_Twoshot_R32,
  ncclSymkKernelId_AllReduce_Lamport1Shot,
  ncclSymkKernelId_AllReduce_Lamport1ShotMC,
  ncclSymkKernelId_AllReduce_SOL,

  ncclSymkKernelId_AllGather_LL,
  ncclSymkKernelId_AllGather_LLMC,
  ncclSymkKernelId_AllGather_ST,
  ncclSymkKernelId_AllGather_STMC,

  ncclSymkKernelId_ReduceScatter_LL,
  ncclSymkKernelId_ReduceScatter_LD,
  ncclSymkKernelId_ReduceScatter_LDMC,

  // ncclLLBuffer-based reduce-scatter.
  // Rank-specialized variants selected based on nRanks.
  ncclSymkKernelId_ReduceScatter_LLBuffer,
  ncclSymkKernelId_ReduceScatter_LLBuffer_R4,
  ncclSymkKernelId_ReduceScatter_LLBuffer_R8,
  ncclSymkKernelId_ReduceScatter_LLBuffer_R16,
  ncclSymkKernelId_ReduceScatter_LLBuffer_R32,
  ncclSymkKernelId_ReduceScatter_LLBuffer_LL16,
  ncclSymkKernelId_ReduceScatter_LLBuffer_LL16_R4,
  ncclSymkKernelId_ReduceScatter_LLBuffer_LL16_R8,
  ncclSymkKernelId_ReduceScatter_LLBuffer_LL16_R16,
  ncclSymkKernelId_ReduceScatter_LLBuffer_LL16_R32,
  // Multimem version of ReduceScatter_LLBuffer
  ncclSymkKernelId_ReduceScatter_LLBufferMC,

  // ncclLLBuffer-based all-gather.
  // Rank-specialized variants selected based on nRanks.
  ncclSymkKernelId_AllGather_LLBuffer,
  ncclSymkKernelId_AllGather_LLBuffer_R4,
  ncclSymkKernelId_AllGather_LLBuffer_R8,
  ncclSymkKernelId_AllGather_LLBuffer_R16,
  ncclSymkKernelId_AllGather_LLBuffer_R32,
  ncclSymkKernelId_AllGather_LLBuffer_LL16,
  ncclSymkKernelId_AllGather_LLBuffer_LL16_R4,
  ncclSymkKernelId_AllGather_LLBuffer_LL16_R8,
  ncclSymkKernelId_AllGather_LLBuffer_LL16_R16,
  ncclSymkKernelId_AllGather_LLBuffer_LL16_R32,
  // Multimem version of AllGather_LLBuffer
  ncclSymkKernelId_AllGather_LLBufferMC,

  ncclSymkKernelId_AllGather_GinHier_MCRing,

  ncclSymkKernelId_Count
};

struct ncclSymkDevComm {
  struct ncclDevComm devComm;
  struct ncclLLA2AHandle lsaLLA2A;
  struct ncclGinSyncHandle ginSyncHandle;
  // Lamport 2-shot accumulation buffer
  ncclWindow_t accumBuffer;
  size_t lamportAccumOffset;
  size_t lamportAccumStrideBytes;
  uint32_t lamportAccumSlotCount;
};

struct ncclSymkState {
  bool initialized;
  struct ncclSymkDevComm kcomm;
  bool lamportPoisonInit;
  uint32_t lamportLastSlot;
  uint32_t lamportSlotCount;
  size_t lamportSlotStrideBytes;
  // Host-visible base pointer of the accumulation buffer for per-launch zeroing
  void* lamportAccumDevBase;
};

struct ncclSymkChannelWorkRange {
  uint16_t workHi; // inclusive index of my ending work
  uint16_t fracHi; // 16-bit fraction in (0.0, 1.0] indicating where my part ends
};

// 16 bytes aligned
struct alignas(16) ncclSymkDevWork {
  uint64_t redOpArg; // must be collectively uniform
  size_t nElts;
  struct ncclWindow_vidmem* inputWin, *outputWin;
  size_t inputOff, outputOff; // these = origUserOffset + cbdPartOffset
  uint64_t rootRank;
  uint64_t sChannelId:16, nChannels:16, padding:32;
};

struct alignas(16) ncclSymkDevWorkArgs {
  struct ncclSymkDevComm kcomm;
  int nMaxChannels;
  // starting of channelWorkRange will be aligned to 16 bytes
  // channelWorkRange[nChannels];
  // ncclSymDevWork[nWorks];
  // aux functions
  __host__ static constexpr size_t calcArgsSize(int nChannels, int nWorks) {
    return alignUp(sizeof(struct ncclSymkDevWorkArgs), 16) + alignUp(nChannels * sizeof(struct ncclSymkChannelWorkRange), 16) + nWorks * sizeof(struct ncclSymkDevWork);
  }
  __host__ __device__ struct ncclSymkChannelWorkRange* getWorkRange() const {
    return (struct ncclSymkChannelWorkRange*)((uint8_t*)this + alignUp(sizeof(struct ncclSymkDevWorkArgs), 16));
  }
  __host__ __device__ struct ncclSymkDevWork* getWorks(int nChannels) const {
    return (struct ncclSymkDevWork*)((uint8_t*)this->getWorkRange() + alignUp(nChannels * sizeof(struct ncclSymkChannelWorkRange), 16));
  }
};

union ncclSymkDevWorkArgs4K {
  struct ncclSymkDevWorkArgs args;
  char buf4K[4096];
};

typedef enum {
  ncclSymSendNonregRecvNonreg = 0,
  ncclSymSendNonregRecvReg = 1,
  ncclSymSendRegRecvNonreg = 2,
  ncclSymSendRegRecvReg = 3,
  ncclNumSymRegTypes = 4
} ncclSymRegType_t;

// We assume ncclComm contains a field: `ncclSymkState symkState`
ncclResult_t ncclSymkInitOnce(struct ncclComm* comm);
ncclResult_t ncclSymkFinalize(struct ncclComm* comm);

bool ncclSymkAvailable(struct ncclComm* comm, ncclFunc_t coll, int/*ncclDevRedOp_t*/ red,
                       ncclDataType_t ty, size_t nElts);
ncclResult_t ncclSymkPickKernel(struct ncclComm* comm, ncclFunc_t coll, int/*ncclDevRedOp_t*/ red, ncclDataType_t ty,
                                size_t nEltsTotal, size_t nEltsMax, int nWorks, ncclSymRegType_t winRegType,
                                float* estTimeUs, ncclSymkKernelId* kernelId, int* nBlocks, int* nWarps, int* gridDimY, bool* forced);

ncclResult_t ncclSymkMakeDevWork(struct ncclComm* comm, struct ncclTaskColl* task, struct ncclSymkDevWork* outDevWork);

// Generated by src/device/symmetric/generate.py
extern int const ncclSymkKernelCount;
extern int ncclSymkKernelRequirements[/*ncclSymkKernelCount*/];
extern void* const ncclSymkKernelList[];
void* ncclSymkGetKernelPtr(ncclSymkKernelId kernelId, int/*ncclDevRedOp_t*/ red, ncclDataType_t ty);
const char* ncclSymkKernelIdToString(int kernelId);
ncclResult_t ncclGetSymRegType(struct ncclDevrWindow* sendWin, struct ncclDevrWindow* recvWin, ncclSymRegType_t* winRegType);

uint64_t ncclSymkLLKernelMask();

constexpr int ncclSymkGinWorldBufSize = 16<<10;
constexpr int ncclSymkGinRailBufSize = 4<<20;
#endif
