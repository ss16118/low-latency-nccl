/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "sym_kernels.h"
#include "comm.h"
#include "device.h"
#include "transport.h"
#include "dev_runtime.h"
#include "alloc.h"
#include "nccl_device/ll_buffer.h"
#include <cmath>
#include <cfloat>
#include <algorithm>

#define NCCL_ONESHOT_LLBUFFER_KERNEL_THRESHOLD 1048576 // 1MiB
#define NCCL_TWOSHOT_LLBUFFER_KERNEL_THRESHOLD 8388608 // 8MiB

constexpr char const* kernelName[] = {
  // Must align with enum ncclSymkKernelId definition in src/include/sym_kernels.h
  "AllReduce_AGxLL_R",
  "AllReduce_AGxLLMC_R",
  "AllReduce_RSxLD_AGxST",
  "AllReduce_RSxLDMC_AGxSTMC",
  "AllReduce_RSxNet_ARxMC_AGxNet",
  "AllReduce_Lamport2Shot",
  "AllReduce_Lamport2ShotMC",
  // AllReduce LLBuffer with rank specialization
  "AllReduce_LLBuffer",
  "AllReduce_LLBuffer_R4",
  "AllReduce_LLBuffer_R8",
  "AllReduce_LLBuffer_R16",
  "AllReduce_LLBuffer_R32",
  "AllReduce_LLBuffer_LL16",
  "AllReduce_LLBuffer_LL16_R4",
  "AllReduce_LLBuffer_LL16_R8",
  "AllReduce_LLBuffer_LL16_R16",
  "AllReduce_LLBuffer_LL16_R32",
  "AllReduce_LLBufferMC",
  "AllReduce_LLBuffer_LL16MC",
  // AllReduce LLBuffer_Twoshot with rank specialization
  "AllReduce_LLBuffer_Twoshot",
  // "AllReduce_LLBuffer_Twoshot_R4",
  "AllReduce_LLBuffer_Twoshot_R8",
  "AllReduce_LLBuffer_Twoshot_LL16",
  "AllReduce_LLBuffer_Twoshot_LL16_R8",
  "AllReduce_LLBuffer_TwoshotMC",
  "AllReduce_LLBuffer_Twoshot_LL16MC",
  "AllReduce_Lamport1Shot",
  "AllReduce_Lamport1ShotMC",
  "AllReduce_SOL",
  "AllGather_LL",
  "AllGather_LLMC",
  "AllGather_ST",
  "AllGather_STMC",
  "ReduceScatter_LL",
  "ReduceScatter_LD",
  "ReduceScatter_LDMC",
  // ReduceScatter LLBuffer with rank specialization
  "ReduceScatter_LLBuffer",
  // "ReduceScatter_LLBuffer_R4",
  "ReduceScatter_LLBuffer_R8",
  // "ReduceScatter_LLBuffer_R16",
  // "ReduceScatter_LLBuffer_R32",
  "ReduceScatter_LLBuffer_LL16",
  // "ReduceScatter_LLBuffer_LL16_R4",
  "ReduceScatter_LLBuffer_LL16_R8",
  // "ReduceScatter_LLBuffer_LL16_R16",
  // "ReduceScatter_LLBuffer_LL16_R32",
  "ReduceScatter_LLBufferMC",
  // AllGather LLBuffer with rank specialization
  "AllGather_LLBuffer",
  // "AllGather_LLBuffer_R4",
  "AllGather_LLBuffer_R8",
  // "AllGather_LLBuffer_R16",
  // "AllGather_LLBuffer_R32",
  // "AllGather_LLBuffer_LL16",
  "AllGather_LLBuffer_LL16_R4",
  "AllGather_LLBuffer_LL16_R8",
  // "AllGather_LLBuffer_LL16_R16",
  // "AllGather_LLBuffer_LL16_R32",
  "AllGather_LLBufferMC",
  "AllGather_GinHier_MCRing",
  // Reduce LLBuffer with rank specialization
  "Reduce_LLBuffer",
  "Reduce_LLBuffer_R8",
  "Reduce_LLBuffer_LL16",
  "Reduce_LLBuffer_LL16_R8",
  // Broadcast LLBuffer with rank specialization
  "Broadcast_LLBuffer",
  "Broadcast_LLBuffer_R8",
  "Broadcast_LLBuffer_LL16",
  "Broadcast_LLBuffer_LL16_R8",
  "Broadcast_LLBufferMC"
};

// Helper to get rank-specialized LLBuffer kernel ID based on nRanks.
// Returns the base kernel ID if nRanks is not in {4,8,16,32}.
static ncclSymkKernelId getLLBufferRankKernel(ncclSymkKernelId baseKernel, int nRanks) {
  // Map nRanks to offset (0 for base, 1-4 for R4/R8/R16/R32)
  int offset = 0;
  if (baseKernel == ncclSymkKernelId_AllReduce_LLBuffer_Twoshot ||
    baseKernel == ncclSymkKernelId_AllReduce_LLBuffer_Twoshot_LL16 ||
    baseKernel == ncclSymkKernelId_AllGather_LLBuffer ||
    baseKernel == ncclSymkKernelId_AllGather_LLBuffer_LL16 ||
    baseKernel == ncclSymkKernelId_ReduceScatter_LLBuffer ||
    baseKernel == ncclSymkKernelId_ReduceScatter_LLBuffer_LL16 ||
    baseKernel == ncclSymkKernelId_Reduce_LLBuffer ||
    baseKernel == ncclSymkKernelId_Reduce_LLBuffer_LL16 ||
    baseKernel == ncclSymkKernelId_Broadcast_LLBuffer ||
    baseKernel == ncclSymkKernelId_Broadcast_LLBuffer_LL16) {
    offset = nRanks == 8 ? 1 : 0;
  } else {
    switch (nRanks) {
      case 4:  offset = 1; break;
      case 8:  offset = 2; break;
      case 16: offset = 3; break;
      case 32: offset = 4; break;
      default: offset = 0; break; // Use base for other rank counts
    }
  }
  // The rank-specialized IDs follow immediately after the base ID
  return (ncclSymkKernelId)(baseKernel + offset);
}

// Check if a kernel ID is an LLBuffer kernel that has rank specialization
static bool isLLBufferRankSpecialized(ncclSymkKernelId k) {
  return k == ncclSymkKernelId_AllReduce_LLBuffer ||
         k == ncclSymkKernelId_AllReduce_LLBuffer_LL16 ||
         k == ncclSymkKernelId_AllReduce_LLBuffer_Twoshot ||
         k == ncclSymkKernelId_AllReduce_LLBuffer_Twoshot_LL16 ||
         k == ncclSymkKernelId_ReduceScatter_LLBuffer ||
         k == ncclSymkKernelId_ReduceScatter_LLBuffer_LL16 ||
         k == ncclSymkKernelId_AllGather_LLBuffer ||
         k == ncclSymkKernelId_AllGather_LLBuffer_LL16 ||
         k == ncclSymkKernelId_Reduce_LLBuffer ||
         k == ncclSymkKernelId_Reduce_LLBuffer_LL16 ||
         k == ncclSymkKernelId_Broadcast_LLBuffer ||
         k == ncclSymkKernelId_Broadcast_LLBuffer_LL16;
}

constexpr uint64_t kernelMask_STMC = 1ull<<ncclSymkKernelId_AllGather_LLMC |
                                     1ull<<ncclSymkKernelId_AllGather_STMC |
                                     1ull<<ncclSymkKernelId_AllReduce_AGxLLMC_R |
                                     1ull<<ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC |
                                     1ull<<ncclSymkKernelId_AllReduce_Lamport2ShotMC |
                                    //  1ull<<ncclSymkKernelId_AllReduce_Lamport1ShotPoisonMC |
                                     1ull<<ncclSymkKernelId_AllReduce_LLBufferMC |
                                     1ull<<ncclSymkKernelId_AllReduce_LLBuffer_LL16MC |
                                     1ull<<ncclSymkKernelId_AllReduce_Lamport1ShotMC |
                                     1ull<<ncclSymkKernelId_ReduceScatter_LDMC |
                                     1ull<<ncclSymkKernelId_ReduceScatter_LLBufferMC |
                                     1ull<<ncclSymkKernelId_AllGather_LLBufferMC |
                                    //  1ull<<ncclSymkKernelId_Reduce_LLBufferMC |
                                     1ull<<ncclSymkKernelId_Broadcast_LLBufferMC;

constexpr uint64_t kernelMask_AG = 1ull<<ncclSymkKernelId_AllGather_LL |
                                   1ull<<ncclSymkKernelId_AllGather_LLMC |
                                   1ull<<ncclSymkKernelId_AllGather_ST |
                                   1ull<<ncclSymkKernelId_AllGather_STMC |
                                   1ull<<ncclSymkKernelId_AllGather_GinHier_MCRing |
                                   1ull<<ncclSymkKernelId_AllGather_LLBuffer |
                                   1ull<<ncclSymkKernelId_AllGather_LLBuffer_LL16 |
                                   1ull<<ncclSymkKernelId_AllGather_LLBufferMC;

constexpr uint64_t kernelMask_LDMC = 1ull<<ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC |
                                     1ull<<ncclSymkKernelId_ReduceScatter_LDMC;

// Multimem-enabled LLBuffer kernels
constexpr uint64_t kernelMask_LLBufferMC = 1ull<<ncclSymkKernelId_AllReduce_LLBufferMC |
                                           1ull<<ncclSymkKernelId_AllReduce_LLBuffer_LL16MC |
                                           1ull<<ncclSymkKernelId_ReduceScatter_LLBufferMC |
                                           1ull<<ncclSymkKernelId_AllGather_LLBufferMC |
                                          //  1ull<<ncclSymkKernelId_Reduce_LLBufferMC |
                                           1ull<<ncclSymkKernelId_Broadcast_LLBufferMC |
                                           1ull<<ncclSymkKernelId_AllReduce_LLBuffer_TwoshotMC |
                                           1ull<<ncclSymkKernelId_AllReduce_LLBuffer_Twoshot_LL16MC;

constexpr uint64_t kernelMask_1Shot_LL = 1ull<<ncclSymkKernelId_AllReduce_AGxLL_R |
                                   1ull<<ncclSymkKernelId_AllReduce_AGxLLMC_R |
                                   1ull<<ncclSymkKernelId_AllReduce_LLBuffer |
                                   1ull<<ncclSymkKernelId_AllReduce_LLBuffer_R4 |
                                   1ull<<ncclSymkKernelId_AllReduce_LLBuffer_R8 |
                                   1ull<<ncclSymkKernelId_AllReduce_LLBuffer_R16 |
                                   1ull<<ncclSymkKernelId_AllReduce_LLBuffer_R32 |
                                   1ull<<ncclSymkKernelId_AllReduce_LLBuffer_LL16 |
                                   1ull<<ncclSymkKernelId_AllReduce_LLBuffer_LL16_R4 |
                                   1ull<<ncclSymkKernelId_AllReduce_LLBuffer_LL16_R8 |
                                   1ull<<ncclSymkKernelId_AllReduce_LLBuffer_LL16_R16 |
                                   1ull<<ncclSymkKernelId_AllReduce_LLBuffer_LL16_R32 |
                                   1ull<<ncclSymkKernelId_AllReduce_LLBufferMC |
                                   1ull<<ncclSymkKernelId_AllReduce_LLBuffer_LL16MC |
                                   1ull<<ncclSymkKernelId_AllGather_LL |
                                   1ull<<ncclSymkKernelId_AllGather_LLMC |
                                   1ull<<ncclSymkKernelId_AllGather_LLBuffer |
                                   1ull<<ncclSymkKernelId_AllGather_LLBuffer_R8 |
                                   1ull<<ncclSymkKernelId_AllGather_LLBuffer_LL16 |
                                   1ull<<ncclSymkKernelId_AllGather_LLBuffer_LL16_R8 |
                                   1ull<<ncclSymkKernelId_AllGather_LLBufferMC |
                                   1ull<<ncclSymkKernelId_ReduceScatter_LL |
                                   1ull<<ncclSymkKernelId_ReduceScatter_LLBuffer |
                                   1ull<<ncclSymkKernelId_ReduceScatter_LLBuffer_R8 |
                                   1ull<<ncclSymkKernelId_ReduceScatter_LLBuffer_LL16 |
                                   1ull<<ncclSymkKernelId_ReduceScatter_LLBuffer_LL16_R8 |
                                   1ull<<ncclSymkKernelId_ReduceScatter_LLBufferMC |
                                   1ull<<ncclSymkKernelId_Reduce_LLBuffer |
                                   1ull<<ncclSymkKernelId_Reduce_LLBuffer_R8 |
                                   1ull<<ncclSymkKernelId_Reduce_LLBuffer_LL16 |
                                   1ull<<ncclSymkKernelId_Reduce_LLBuffer_LL16_R8 |
                                  //  1ull<<ncclSymkKernelId_Reduce_LLBufferMC |
                                   1ull<<ncclSymkKernelId_Broadcast_LLBuffer |
                                   1ull<<ncclSymkKernelId_Broadcast_LLBuffer_R8 |
                                   1ull<<ncclSymkKernelId_Broadcast_LLBuffer_LL16 |
                                   1ull<<ncclSymkKernelId_Broadcast_LLBuffer_LL16_R8 |
                                   1ull<<ncclSymkKernelId_Broadcast_LLBufferMC;


constexpr uint64_t kernelMask_2Shot_LL = 1ull<<ncclSymkKernelId_AllReduce_LLBuffer_Twoshot |
                                         1ull<<ncclSymkKernelId_AllReduce_LLBuffer_Twoshot_R8 |
                                         1ull<<ncclSymkKernelId_AllReduce_LLBuffer_Twoshot_LL16 |
                                         1ull<<ncclSymkKernelId_AllReduce_LLBuffer_Twoshot_LL16_R8 |
                                         1ull<<ncclSymkKernelId_AllReduce_LLBuffer_TwoshotMC |
                                         1ull<<ncclSymkKernelId_AllReduce_LLBuffer_Twoshot_LL16MC;

constexpr uint64_t kernelMask_LSA = 1ull<<ncclSymkKernelId_AllReduce_AGxLL_R |
                                    1ull<<ncclSymkKernelId_AllReduce_AGxLLMC_R |
                                    1ull<<ncclSymkKernelId_AllReduce_RSxLD_AGxST |
                                    1ull<<ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC |
                                    1ull<<ncclSymkKernelId_AllReduce_Lamport2Shot |
                                    1ull<<ncclSymkKernelId_AllReduce_Lamport2ShotMC |
                                    // 1ull<<ncclSymkKernelId_AllReduce_Lamport2ShotPoison |
                                    // 1ull<<ncclSymkKernelId_AllReduce_Lamport1ShotV2 |
                                    // 1ull<<ncclSymkKernelId_AllReduce_Lamport1ShotPoison |
                                    // 1ull<<ncclSymkKernelId_AllReduce_Lamport1ShotPoisonMC |
                                    1ull<<ncclSymkKernelId_AllReduce_Lamport1Shot |
                                    1ull<<ncclSymkKernelId_AllReduce_Lamport1ShotMC |
                                    1ull<<ncclSymkKernelId_AllReduce_LLBuffer |
                                    1ull<<ncclSymkKernelId_AllReduce_LLBuffer_LL16 |
                                    1ull<<ncclSymkKernelId_AllReduce_LLBufferMC |
                                    1ull<<ncclSymkKernelId_AllReduce_LLBuffer_LL16MC |
                                    1ull<<ncclSymkKernelId_AllReduce_LLBuffer_Twoshot |
                                    1ull<<ncclSymkKernelId_AllReduce_LLBuffer_Twoshot_LL16 |
                                    1ull<<ncclSymkKernelId_AllReduce_LLBuffer_TwoshotMC |
                                    1ull<<ncclSymkKernelId_AllReduce_LLBuffer_Twoshot_LL16MC |
                                    1ull<<ncclSymkKernelId_AllReduce_SOL |
                                    1ull<<ncclSymkKernelId_AllGather_LL |
                                    1ull<<ncclSymkKernelId_AllGather_LLMC |
                                    1ull<<ncclSymkKernelId_AllGather_ST |
                                    1ull<<ncclSymkKernelId_AllGather_STMC |
                                    1ull<<ncclSymkKernelId_AllGather_LLBuffer |
                                    1ull<<ncclSymkKernelId_AllGather_LLBuffer_LL16 |
                                    1ull<<ncclSymkKernelId_AllGather_LLBufferMC |
                                    1ull<<ncclSymkKernelId_ReduceScatter_LL |
                                    1ull<<ncclSymkKernelId_ReduceScatter_LD |
                                    1ull<<ncclSymkKernelId_ReduceScatter_LDMC |
                                    1ull<<ncclSymkKernelId_ReduceScatter_LLBuffer |
                                    1ull<<ncclSymkKernelId_ReduceScatter_LLBuffer_LL16 |
                                    1ull<<ncclSymkKernelId_ReduceScatter_LLBufferMC |
                                    1ull<<ncclSymkKernelId_Reduce_LLBuffer |
                                    1ull<<ncclSymkKernelId_Reduce_LLBuffer_LL16 |
                                    // 1ull<<ncclSymkKernelId_Reduce_LLBufferMC |
                                    1ull<<ncclSymkKernelId_Broadcast_LLBuffer |
                                    1ull<<ncclSymkKernelId_Broadcast_LLBuffer_LL16 |
                                    1ull<<ncclSymkKernelId_Broadcast_LLBufferMC;


constexpr uint64_t kernelMask_Gin = 1ull<<ncclSymkKernelId_AllGather_GinHier_MCRing;

uint64_t ncclSymkLLKernelMask() {
  return kernelMask_1Shot_LL | kernelMask_2Shot_LL;
}

constexpr uint64_t kernelMask_AR = 1ull<<ncclSymkKernelId_AllReduce_AGxLLMC_R |
                                   1ull<<ncclSymkKernelId_AllReduce_AGxLL_R |
                                   1ull<<ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC |
                                   1ull<<ncclSymkKernelId_AllReduce_RSxLD_AGxST |
                                   1ull<<ncclSymkKernelId_AllReduce_Lamport2Shot |
                                   1ull<<ncclSymkKernelId_AllReduce_Lamport2ShotMC |
                                  //  1ull<<ncclSymkKernelId_AllReduce_Lamport2ShotPoison |
                                  //  1ull<<ncclSymkKernelId_AllReduce_Lamport1ShotV2 |
                                  //  1ull<<ncclSymkKernelId_AllReduce_Lamport1ShotPoison |
                                  //  1ull<<ncclSymkKernelId_AllReduce_Lamport1ShotPoisonMC |
                                   1ull<<ncclSymkKernelId_AllReduce_LLBuffer |
                                   1ull<<ncclSymkKernelId_AllReduce_LLBuffer_LL16 |
                                   1ull<<ncclSymkKernelId_AllReduce_LLBufferMC |
                                   1ull<<ncclSymkKernelId_AllReduce_LLBuffer_LL16MC |
                                   1ull<<ncclSymkKernelId_AllReduce_LLBuffer_Twoshot |
                                   1ull<<ncclSymkKernelId_AllReduce_LLBuffer_Twoshot_LL16 |
                                   1ull<<ncclSymkKernelId_AllReduce_LLBuffer_TwoshotMC |
                                   1ull<<ncclSymkKernelId_AllReduce_LLBuffer_Twoshot_LL16MC |
                                   1ull<<ncclSymkKernelId_AllReduce_Lamport1Shot |
                                   1ull<<ncclSymkKernelId_AllReduce_Lamport1ShotMC |
                                   1ull<<ncclSymkKernelId_AllReduce_SOL;

constexpr uint64_t kernelMask_RS = 1ull<<ncclSymkKernelId_ReduceScatter_LD |
                                   1ull<<ncclSymkKernelId_ReduceScatter_LDMC |
                                   1ull<<ncclSymkKernelId_ReduceScatter_LL |
                                   1ull<<ncclSymkKernelId_ReduceScatter_LLBuffer |
                                   1ull<<ncclSymkKernelId_ReduceScatter_LLBuffer_LL16 |
                                   1ull<<ncclSymkKernelId_ReduceScatter_LLBufferMC;

constexpr uint64_t kernelMask_RD = 1ull<<ncclSymkKernelId_Reduce_LLBuffer |
                                   1ull<<ncclSymkKernelId_Reduce_LLBuffer_LL16;

constexpr uint64_t kernelMask_BC = 1ull<<ncclSymkKernelId_Broadcast_LLBuffer |
                                   1ull<<ncclSymkKernelId_Broadcast_LLBuffer_LL16 |
                                   1ull<<ncclSymkKernelId_Broadcast_LLBufferMC;


static uint64_t kernelMask_coll(ncclFunc_t coll) {
  switch (coll) {
  case ncclFuncAllGather: return kernelMask_AG;
  case ncclFuncAllReduce: return kernelMask_AR;
  case ncclFuncReduceScatter: return kernelMask_RS;
  case ncclFuncReduce: return kernelMask_RD;
  case ncclFuncBroadcast: return kernelMask_BC;
  default: return 0;
  }
}

// Selects ncclLLBuffer sync mode for all LLBuffer-based kernels (AllReduce, AllGather, ReduceScatter).
// 0: Poison (default, requires SYM_LAMPORT_POISON_INIT=1 for correctness)
// 1: LL (i.e., the *_LL16 variants)
NCCL_PARAM(SymLLBufferSync, "SYM_LLBUFFER_SYNC", 0)

// Returns the effective LLBuffer sync mode: 0=Poison, 1=LL
static int getLLBufferSyncMode() {
  return ncclParamSymLLBufferSync();
}

static uint64_t kernelMask_user() {
  static uint64_t cache = ~0ull;
  uint64_t got = COMPILER_ATOMIC_LOAD(&cache, std::memory_order_relaxed);
  if (got == ~0ull) {
    // TODO: Enhance this to be a pattern match. I like regex's but we also have
    // the parseList() used by NCCL_ALGO/PROTO.
    char const* name = ncclGetEnv("NCCL_SYM_KERNEL");
    if (name == nullptr || strcmp(name, "^") == 0) {
      static_assert((int)ncclSymkKernelId_Count < 64, "Use more than 64 bits");
      got = (1ull<<(int)ncclSymkKernelId_Count)-1;
    } else {
      got = 0;
      int sync = getLLBufferSyncMode(); // 0=Poison, 1=LL
      // Special-case: treat "AllReduce_LLBuffer" as meta-name that enables all rank variants
      // whose ncclLLBuffer sync mode is selected via NCCL_SYM_LLBUFFER_SYNC
      if (strcmp(name, "AllReduce_LLBuffer") == 0) {
        ncclSymkKernelId base = (sync == 0) ? ncclSymkKernelId_AllReduce_LLBuffer : ncclSymkKernelId_AllReduce_LLBuffer_LL16;
        // Enable base + all rank variants (do NOT enable MC variant when user explicitly requests non-MC)
        for (int i = 0; i < 5; i++) got |= 1ull << (base + i);
        __atomic_store_n(&cache, got, __ATOMIC_RELAXED);
      } else if (strcmp(name, "AllReduce_LLBufferMC") == 0) {
        ncclSymkKernelId base = (sync == 0) ? ncclSymkKernelId_AllReduce_LLBufferMC : ncclSymkKernelId_AllReduce_LLBuffer_LL16MC;
        got |= 1ull << base;
        __atomic_store_n(&cache, got, __ATOMIC_RELAXED);
      } else if (strcmp(name, "ReduceScatter_LLBuffer") == 0) {
        ncclSymkKernelId base = (sync == 0) ? ncclSymkKernelId_ReduceScatter_LLBuffer : ncclSymkKernelId_ReduceScatter_LLBuffer_LL16;
        // Enable base + all rank variants (do NOT enable MC variant when user explicitly requests non-MC)
        for (int i = 0; i < 2; i++) got |= 1ull << (base + i);
        __atomic_store_n(&cache, got, __ATOMIC_RELAXED);
      } else if (strcmp(name, "AllGather_LLBuffer") == 0) {
        ncclSymkKernelId base = (sync == 0) ? ncclSymkKernelId_AllGather_LLBuffer : ncclSymkKernelId_AllGather_LLBuffer_LL16;
        // Enable base + all rank variants (do NOT enable MC variant when user explicitly requests non-MC)
        for (int i = 0; i < 2; i++) got |= 1ull << (base + i);
        __atomic_store_n(&cache, got, __ATOMIC_RELAXED);
      } else if (strcmp(name, "AllReduce_LLBuffer_Twoshot") == 0) {
        // LLBuffer_Twoshot always uses ncclPoison mode, enable base + R8 variant
        // ncclSymkKernelId base = ncclSymkKernelId_AllReduce_LLBuffer_Twoshot;
        ncclSymkKernelId base = (sync == 0) ? ncclSymkKernelId_AllReduce_LLBuffer_Twoshot : ncclSymkKernelId_AllReduce_LLBuffer_Twoshot_LL16;
        for (int i = 0; i < 2; i++) got |= 1ull << (base + i);
        __atomic_store_n(&cache, got, __ATOMIC_RELAXED);
      } else if (strcmp(name, "AllReduce_LLBuffer_TwoshotMC") == 0) {
        ncclSymkKernelId base = (sync == 0) ? ncclSymkKernelId_AllReduce_LLBuffer_TwoshotMC : ncclSymkKernelId_AllReduce_LLBuffer_Twoshot_LL16MC;
        got |= 1ull << base;
        __atomic_store_n(&cache, got, __ATOMIC_RELAXED);
      } else if (strcmp(name, "Reduce_LLBuffer") == 0) {
        ncclSymkKernelId base = (sync == 0) ? ncclSymkKernelId_Reduce_LLBuffer : ncclSymkKernelId_Reduce_LLBuffer_LL16;
        // Enable base + R8 variant (do NOT enable MC variant when user explicitly requests non-MC)
        for (int i = 0; i < 2; i++) got |= 1ull << (base + i);
        __atomic_store_n(&cache, got, __ATOMIC_RELAXED);
      } else if (strcmp(name, "Broadcast_LLBuffer") == 0) {
        ncclSymkKernelId base = (sync == 0) ? ncclSymkKernelId_Broadcast_LLBuffer : ncclSymkKernelId_Broadcast_LLBuffer_LL16;
        // Enable base + R8 variant (do NOT enable MC variant when user explicitly requests non-MC)
        for (int i = 0; i < 2; i++) got |= 1ull << (base + i);
        __atomic_store_n(&cache, got, __ATOMIC_RELAXED);
      } else {
        for (int k=0; k < (int)ncclSymkKernelId_Count; k++) {
          if (strcmp(kernelName[k], name) == 0) {
            __atomic_store_n(&cache, 1ull<<k, __ATOMIC_RELAXED);
            got = 1ull<<k;
            break;
          }
        }
      }
    }
    COMPILER_ATOMIC_STORE(&cache, got, std::memory_order_relaxed);
  }
  return got;
}

NCCL_PARAM(SymCTAs, "SYM_CTAS", 0)
NCCL_PARAM(SymLamportPoisonInit, "SYM_LAMPORT_POISON_INIT", 1)
NCCL_PARAM(SymMaxConcurrentEpochs, "SYM_MAX_CONCURRENT_EPOCHS", 0)
NCCL_PARAM(SymKernelMaxWarps, "SYM_KERNEL_MAX_WARPS", 16)
// Poison dtype selection for SYM_LAMPORT_POISON_INIT=1.
// Numeric mapping (preferred):
//   0:f32, 1:f16, 2:bf16, 3:fp8e4m3, 4:fp8e5m2, 5:int8, 6:int32, 7:int64, 8:f64
NCCL_PARAM(SymLamportPoisonDtype, "SYM_LAMPORT_POISON_DTYPE", 0)

// Parse data type from NCCL_SYM_LAMPORT_POISON_DTYPE (numeric via NCCL_PARAM).
static ncclDataType_t getLamportPoisonDataType() {
  static ncclDataType_t cache = ncclNumTypes; // sentinel for "not yet parsed"
  ncclDataType_t got = __atomic_load_n(&cache, __ATOMIC_RELAXED);
  if (got == ncclNumTypes) {
    got = ncclFloat32; // default
    int v = ncclParamSymLamportPoisonDtype();
    switch (v) {
    default:
      WARN("Unknown NCCL_SYM_LAMPORT_POISON_DTYPE=%d, using float32", v);
      got = ncclFloat32;
      break;
    case 0: got = ncclFloat32; break;
    case 1: got = ncclFloat16; break;
    case 2: got = ncclBfloat16; break;
    case 3: got = ncclFloat8e4m3; break;
    case 4: got = ncclFloat8e5m2; break;
    case 5: got = ncclInt8; break;
    case 6: got = ncclInt32; break;
    case 7: got = ncclInt64; break;
    case 8: got = ncclFloat64; break;
    }
    __atomic_store_n(&cache, got, __ATOMIC_RELAXED);
  }
  return got;
}

static double softmin(double x, double ceiling, double softness) {
  // looks like a smooth version of: min(x, ceiling)
  return ceiling - softness*std::log1p((std::exp(ceiling/softness) - 1)*std::exp(-x/softness));
}

static double softplus(double x, double softness) {
  // looks like a smooth version of: max(0, x)
  double z = x/softness;
  return 100.0 <= z ? x : softness*std::log1p(std::exp(z));
}

static double model(double busBytes, double baseLat, int nSMs, double smBw, double busMultiplier, double peakBw) {
  double bw = softmin(nSMs*smBw*busMultiplier, peakBw, smBw);
  return baseLat + softplus(busBytes/bw - 1, 1);
}

// Check if kernel is an LL-style kernel (LLBuffer, LL, or Lamport poison) that uses 8 bytes per thread
static bool isLLStyleKernel(ncclSymkKernelId k) {
  // LLBuffer kernels (all variants)
  if (k >= ncclSymkKernelId_AllReduce_LLBuffer && k <= ncclSymkKernelId_AllReduce_LLBuffer_R32) return true;
  if (k >= ncclSymkKernelId_AllReduce_LLBuffer_LL16 && k <= ncclSymkKernelId_AllReduce_LLBuffer_LL16_R32) return true;
  if (k == ncclSymkKernelId_AllReduce_LLBufferMC) return true;
  if (k == ncclSymkKernelId_AllReduce_LLBuffer_LL16MC) return true;
  if (k >= ncclSymkKernelId_AllReduce_LLBuffer_Twoshot && k <= ncclSymkKernelId_AllReduce_LLBuffer_Twoshot_R8) return true;
  if (k >= ncclSymkKernelId_AllReduce_LLBuffer_Twoshot_LL16 && k <= ncclSymkKernelId_AllReduce_LLBuffer_Twoshot_LL16_R8) return true;
  if (k == ncclSymkKernelId_AllReduce_LLBuffer_TwoshotMC) return true;
  if (k == ncclSymkKernelId_AllReduce_LLBuffer_Twoshot_LL16MC) return true;
  if (k >= ncclSymkKernelId_ReduceScatter_LLBuffer && k <= ncclSymkKernelId_ReduceScatter_LLBuffer_R8) return true;
  if (k >= ncclSymkKernelId_ReduceScatter_LLBuffer_LL16 && k <= ncclSymkKernelId_ReduceScatter_LLBuffer_LL16_R8) return true;
  if (k == ncclSymkKernelId_ReduceScatter_LLBufferMC) return true;
  if (k >= ncclSymkKernelId_AllGather_LLBuffer && k <= ncclSymkKernelId_AllGather_LLBuffer_R8) return true;
  if (k >= ncclSymkKernelId_AllGather_LLBuffer_LL16 && k <= ncclSymkKernelId_AllGather_LLBuffer_LL16_R8) return true;
  if (k == ncclSymkKernelId_AllGather_LLBufferMC) return true;
  // Reduce LLBuffer kernels
  if (k >= ncclSymkKernelId_Reduce_LLBuffer && k <= ncclSymkKernelId_Reduce_LLBuffer_R8) return true;
  if (k >= ncclSymkKernelId_Reduce_LLBuffer_LL16 && k <= ncclSymkKernelId_Reduce_LLBuffer_LL16_R8) return true;
  // Broadcast LLBuffer kernels
  if (k >= ncclSymkKernelId_Broadcast_LLBuffer && k <= ncclSymkKernelId_Broadcast_LLBuffer_R8) return true;
  if (k >= ncclSymkKernelId_Broadcast_LLBuffer_LL16 && k <= ncclSymkKernelId_Broadcast_LLBuffer_LL16_R8) return true;
  if (k == ncclSymkKernelId_Broadcast_LLBufferMC) return true;
  // LL kernels
  if (k == ncclSymkKernelId_AllReduce_AGxLL_R) return true;
  if (k == ncclSymkKernelId_AllReduce_AGxLLMC_R) return true;
  if (k == ncclSymkKernelId_AllReduce_RSxLD_AGxST) return true;
  if (k == ncclSymkKernelId_AllGather_LL) return true;
  if (k == ncclSymkKernelId_AllGather_LLMC) return true;
  if (k == ncclSymkKernelId_ReduceScatter_LL) return true;
  // Lamport poison kernels
  // if (k == ncclSymkKernelId_AllReduce_Lamport1ShotPoison) return true;
  // if (k == ncclSymkKernelId_AllReduce_Lamport1ShotPoisonMC) return true;
  // Lamport 1-shot non-poison also uses 8 bytes per thread pattern
  if (k == ncclSymkKernelId_AllReduce_Lamport1Shot) return true;
  if (k == ncclSymkKernelId_AllReduce_Lamport1ShotMC) return true;
  // if (k == ncclSymkKernelId_AllReduce_Lamport1ShotV2) return true;
  if (k == ncclSymkKernelId_AllReduce_SOL) return true;
  return false;
}

static bool isLLBufferKernel(ncclSymkKernelId k) {
  if (k >= ncclSymkKernelId_AllReduce_LLBuffer && k <= ncclSymkKernelId_AllReduce_LLBuffer_R32) return true;
  if (k >= ncclSymkKernelId_AllReduce_LLBuffer_LL16 && k <= ncclSymkKernelId_AllReduce_LLBuffer_LL16_R32) return true;
  if (k == ncclSymkKernelId_AllReduce_LLBufferMC) return true;
  if (k == ncclSymkKernelId_AllReduce_LLBuffer_LL16MC) return true;
  if (k >= ncclSymkKernelId_AllReduce_LLBuffer_Twoshot && k <= ncclSymkKernelId_AllReduce_LLBuffer_Twoshot_R8) return true;
  if (k >= ncclSymkKernelId_AllReduce_LLBuffer_Twoshot_LL16 && k <= ncclSymkKernelId_AllReduce_LLBuffer_Twoshot_LL16_R8) return true;
  if (k == ncclSymkKernelId_AllReduce_LLBuffer_TwoshotMC) return true;
  if (k == ncclSymkKernelId_AllReduce_LLBuffer_Twoshot_LL16MC) return true;
  if (k >= ncclSymkKernelId_ReduceScatter_LLBuffer && k <= ncclSymkKernelId_ReduceScatter_LLBuffer_R8) return true;
  if (k >= ncclSymkKernelId_ReduceScatter_LLBuffer_LL16 && k <= ncclSymkKernelId_ReduceScatter_LLBuffer_LL16_R8) return true;
  if (k == ncclSymkKernelId_ReduceScatter_LLBufferMC) return true;
  if (k >= ncclSymkKernelId_AllGather_LLBuffer && k <= ncclSymkKernelId_AllGather_LLBuffer_R8) return true;
  if (k >= ncclSymkKernelId_AllGather_LLBuffer_LL16 && k <= ncclSymkKernelId_AllGather_LLBuffer_LL16_R8) return true;
  if (k == ncclSymkKernelId_AllGather_LLBufferMC) return true;
  // Reduce LLBuffer kernels
  if (k >= ncclSymkKernelId_Reduce_LLBuffer && k <= ncclSymkKernelId_Reduce_LLBuffer_R8) return true;
  if (k >= ncclSymkKernelId_Reduce_LLBuffer_LL16 && k <= ncclSymkKernelId_Reduce_LLBuffer_LL16_R8) return true;
  // Broadcast LLBuffer kernels
  if (k >= ncclSymkKernelId_Broadcast_LLBuffer && k <= ncclSymkKernelId_Broadcast_LLBuffer_R8) return true;
  if (k >= ncclSymkKernelId_Broadcast_LLBuffer_LL16 && k <= ncclSymkKernelId_Broadcast_LLBuffer_LL16_R8) return true;
  if (k == ncclSymkKernelId_Broadcast_LLBufferMC) return true;
  return false;
}


static bool isLLBufferMCKernel(ncclSymkKernelId k) {
  return k == ncclSymkKernelId_AllReduce_LLBufferMC ||
         k == ncclSymkKernelId_AllReduce_LLBuffer_LL16MC ||
         k == ncclSymkKernelId_ReduceScatter_LLBufferMC ||
         k == ncclSymkKernelId_AllGather_LLBufferMC ||
         k == ncclSymkKernelId_Broadcast_LLBufferMC ||
         k == ncclSymkKernelId_AllReduce_LLBuffer_TwoshotMC ||
         k == ncclSymkKernelId_AllReduce_LLBuffer_Twoshot_LL16MC;
}

static bool isReduceScatterKernel(ncclSymkKernelId k) {
  if (k >= ncclSymkKernelId_ReduceScatter_LLBuffer && k <= ncclSymkKernelId_ReduceScatter_LLBuffer_R8) return true;
  if (k >= ncclSymkKernelId_ReduceScatter_LLBuffer_LL16 && k <= ncclSymkKernelId_ReduceScatter_LLBuffer_LL16_R8) return true;
  if (k == ncclSymkKernelId_ReduceScatter_LLBufferMC) return true;
  if (k == ncclSymkKernelId_ReduceScatter_LD) return true;
  if (k == ncclSymkKernelId_ReduceScatter_LDMC) return true;
  if (k == ncclSymkKernelId_ReduceScatter_LL) return true;
  return false;
}

static bool isReduceKernel(ncclSymkKernelId k) {
  if (k >= ncclSymkKernelId_Reduce_LLBuffer && k <= ncclSymkKernelId_Reduce_LLBuffer_R8) return true;
  if (k >= ncclSymkKernelId_Reduce_LLBuffer_LL16 && k <= ncclSymkKernelId_Reduce_LLBuffer_LL16_R8) return true;
  return false;
}

static bool isBroadcastKernel(ncclSymkKernelId k) {
  if (k >= ncclSymkKernelId_Broadcast_LLBuffer && k <= ncclSymkKernelId_Broadcast_LLBuffer_R8) return true;
  if (k >= ncclSymkKernelId_Broadcast_LLBuffer_LL16 && k <= ncclSymkKernelId_Broadcast_LLBuffer_LL16_R8) return true;
  if (k == ncclSymkKernelId_Broadcast_LLBufferMC) return true;
  return false;
}


// Check if kernel is a Lamport 2-shot kernel (uses LL2lines calculation with gridDim.y = nRanks)
static bool isLamport2ShotL2Kernel(ncclSymkKernelId k) {
  return k == ncclSymkKernelId_AllReduce_Lamport2Shot || // k == ncclSymkKernelId_AllReduce_Lamport2ShotPoison ||
         k == ncclSymkKernelId_AllReduce_Lamport2ShotMC;
}

static bool isLLBufferTwoshotKernel(ncclSymkKernelId k) {
  return k == ncclSymkKernelId_AllReduce_LLBuffer_Twoshot ||
         k == ncclSymkKernelId_AllReduce_LLBuffer_Twoshot_R8 || 
         k == ncclSymkKernelId_AllReduce_LLBuffer_Twoshot_LL16 ||
         k == ncclSymkKernelId_AllReduce_LLBuffer_Twoshot_LL16_R8;
}

static bool isLLBufferOneShotKernel(ncclSymkKernelId k) {
  return k == ncclSymkKernelId_AllReduce_LLBuffer ||
         k == ncclSymkKernelId_AllReduce_LLBuffer_LL16 ||
         k == ncclSymkKernelId_AllReduce_LLBuffer_R4 ||
         k == ncclSymkKernelId_AllReduce_LLBuffer_R8 ||
         k == ncclSymkKernelId_AllReduce_LLBuffer_R16 ||
         k == ncclSymkKernelId_AllReduce_LLBuffer_R32 ||
         k == ncclSymkKernelId_AllReduce_LLBuffer_LL16_R4 ||
         k == ncclSymkKernelId_AllReduce_LLBuffer_LL16_R8 ||
         k == ncclSymkKernelId_AllReduce_LLBuffer_LL16_R16 ||
         k == ncclSymkKernelId_AllReduce_LLBuffer_LL16_R32 ||
         k == ncclSymkKernelId_AllReduce_LLBufferMC ||
         k == ncclSymkKernelId_AllReduce_LLBuffer_LL16MC;
}

// Given the kernel and bytes, return the minimum number of blocks to run on such that
// perf is 99% of running at max blocks, and return the estimate runtime for that
// block count.
static void queryModel_gin(struct ncclComm* comm, ncclSymkKernelId k, size_t nBytes, float* timeUs, int* nBlocks, int* nWarps, int* gridDimY);
static void queryModel_lsa(struct ncclComm* comm, ncclSymkKernelId k, size_t nBytes, float* timeUs, int* nBlocks, int* nWarps, int* gridDimY);

static void queryModel(struct ncclComm* comm, ncclSymkKernelId k, size_t nBytes, float* timeUs, int* nBlocks, int* nWarps, int* gridDimY) {
  if (kernelMask_Gin>>k & 1) {
    queryModel_gin(comm, k, nBytes, timeUs, nBlocks, nWarps, gridDimY);
  } else {
    queryModel_lsa(comm, k, nBytes, timeUs, nBlocks, nWarps, gridDimY);
  }
}

#define NCCL_NVLINK_BW_IDX_HOPPER 0
#define NCCL_NVLINK_BW_IDX_BLACKWELL 1
#define NCCL_NVLINK_BW_IDX_NUM 2

// NVLS max bws NCCL can achieve
static const float nvlinkBws[NCCL_NVLINK_BW_IDX_NUM] = {
  360.0f, // Hopper
  720.0f, // Blackwell
};

static void queryModel_gin(struct ncclComm* comm, ncclSymkKernelId k, size_t nBytes, float* timeUs, int* nBlocks, int* nWarps, int* gridDimY) {
  int compCapIndex = comm->minCompCap >= 100 ? NCCL_NVLINK_BW_IDX_BLACKWELL : NCCL_NVLINK_BW_IDX_HOPPER;
  ncclTeam rail = ncclTeamRail(comm);
  const size_t railChunkSize = ncclSymkGinRailBufSize;
  float netLatency = comm->tunerConstants.hwLatencies[NCCL_HW_NET][NCCL_ALGO_RING][NCCL_PROTO_SIMPLE];
  *timeUs = FLT_MAX;
  *nBlocks = 0;
  *nWarps = 16;  // 512 threads = 16 warps
  *gridDimY = 1;
  switch (k) {
    case ncclSymkKernelId_AllGather_GinHier_MCRing: {
        int requiredBlocks = (int)std::min(DIVUP(nBytes, railChunkSize), (size_t)ncclSymkMaxBlocks);
        int factor = comm->compCap >= 100 ? 32 : 16;
        int maxBlocks = DIVUP(factor, comm->nvlsResources->nHeads);
        float intraBw = nvlinkBws[compCapIndex];
        float interBw = comm->minNetBw;
        float intraTime = (float)(nBytes * comm->nRanks) / intraBw;
        float interTime = (float)(nBytes * (rail.nRanks - 1)) / interBw;
        uint32_t steps = DIVUP(nBytes, railChunkSize) * (rail.nRanks - 1);
        *timeUs = steps * netLatency + std::max(intraTime, interTime);
        *nBlocks = std::max(comm->config.minCTAs, std::min(comm->config.maxCTAs, std::min(requiredBlocks, maxBlocks)));
        break;
      }
  default: break;
  }
}

static void queryModel_lsa(struct ncclComm* comm, ncclSymkKernelId k, size_t nBytes, float* timeUs, int* nBlocks, int* nWarps, int* gridDimY) {
  constexpr double LL_BusFactor = 9; // 2X the bytes, plus some processing, plus no unrolling

  int nRanks = comm->nRanks;
  int nMaxBlocks = ncclSymkMaxBlocks;
  int nMaxBlocksNvls = divUp((comm->cudaArch < 1000 ? 16 : 32), nRanks);
  // int nMaxBlocksNvls = 64;
  size_t busBytes; // max(bytes sent, bytes received)
  double busMultiplier = 1;

  switch (k) {
  default:
    busBytes = size_t(1)<<50;
    break;

  case ncclSymkKernelId_AllReduce_AGxLL_R:
    busBytes = nRanks*nBytes*LL_BusFactor;
    break;
  case ncclSymkKernelId_AllReduce_AGxLLMC_R:
    busBytes = nRanks*nBytes*LL_BusFactor;
    busMultiplier = 1.1; // To beat non-MC LL
    break;
  case ncclSymkKernelId_AllReduce_RSxLD_AGxST:
    busBytes = 2*nBytes*(nRanks-1)/nRanks;
    break;
  case ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC:
    busBytes = nBytes/nRanks + nBytes;
    busMultiplier = nRanks;
    nMaxBlocks = nMaxBlocksNvls;
    break;
  // Needs to be changed to match the actual kernel
  case ncclSymkKernelId_AllReduce_Lamport2Shot:
    busBytes = nRanks*nBytes*LL_BusFactor;
    break;
  case ncclSymkKernelId_AllReduce_LLBuffer_Twoshot:
  case ncclSymkKernelId_AllReduce_LLBuffer_Twoshot_R8:
  case ncclSymkKernelId_AllReduce_LLBuffer_Twoshot_LL16:
  case ncclSymkKernelId_AllReduce_LLBuffer_Twoshot_LL16_R8:
    busBytes = 2*nBytes*(nRanks-1)/nRanks;
    break;
  case ncclSymkKernelId_AllReduce_LLBuffer:
  case ncclSymkKernelId_AllReduce_LLBuffer_LL16:
    // ncclLLBuffer-based AllReduce (Poison or LL sync) moves O(nRanks) traffic through
    // symmetric memory similarly to other LL-style collectives.
    busBytes = nRanks*nBytes*LL_BusFactor;
    break;
  case ncclSymkKernelId_AllReduce_LLBufferMC:
  case ncclSymkKernelId_AllReduce_LLBuffer_LL16MC:
    busBytes = nRanks*nBytes*LL_BusFactor;
    busMultiplier = 1.1; // To beat non-MC version
    break;
  case ncclSymkKernelId_AllReduce_LLBuffer_TwoshotMC:
  case ncclSymkKernelId_AllReduce_LLBuffer_Twoshot_LL16MC:
    busBytes = nRanks*nBytes*LL_BusFactor;
    busMultiplier = 1.1; // To beat non-MC version
    break;
  case ncclSymkKernelId_AllGather_LL:
    busBytes = nRanks*nBytes*LL_BusFactor;
    break;
  case ncclSymkKernelId_AllGather_LLMC:
    busBytes = nRanks*nBytes*LL_BusFactor;
    busMultiplier = 1.1; // To beat non-MC LL
    break;
  case ncclSymkKernelId_AllGather_ST:
    busBytes = (nRanks-1)*nBytes;
    break;
  case ncclSymkKernelId_AllGather_STMC:
    busBytes = (nRanks-1)*nBytes; // Wrong. Should be nRanks*nBytes but we want to beat non-MC.
    busMultiplier = 0.55*nRanks;
    nMaxBlocks = nMaxBlocksNvls;
    break;
  case ncclSymkKernelId_AllGather_LLBuffer:
  case ncclSymkKernelId_AllGather_LLBuffer_LL16:
    busBytes = nRanks*nBytes*LL_BusFactor;
    break;
  case ncclSymkKernelId_AllGather_LLBufferMC:
    busBytes = nRanks*nBytes*LL_BusFactor;
    busMultiplier = 1.1; // To beat non-MC version
    break;

  case ncclSymkKernelId_ReduceScatter_LL:
    busBytes = nRanks*nBytes*LL_BusFactor;
    break;
  case ncclSymkKernelId_ReduceScatter_LD:
    busBytes = (nRanks-1)*nBytes;
    break;
  case ncclSymkKernelId_ReduceScatter_LDMC:
    busBytes = (nRanks-1)*nBytes; // Wrong. Should be nRanks*nBytes but we want to beat non-MC.
    busMultiplier = 0.55*nRanks;
    nMaxBlocks = nMaxBlocksNvls;
    break;
  case ncclSymkKernelId_ReduceScatter_LLBuffer:
  case ncclSymkKernelId_ReduceScatter_LLBuffer_LL16:
    busBytes = nRanks*nBytes*LL_BusFactor;
    break;
  case ncclSymkKernelId_ReduceScatter_LLBufferMC:
    busBytes = nRanks*nBytes*LL_BusFactor;
    busMultiplier = 1.1; // To beat non-MC version
    break;

  // Reduce: all ranks send to root, root reduces
  // Traffic pattern similar to ReduceScatter but all to one rank
  case ncclSymkKernelId_Reduce_LLBuffer:
  case ncclSymkKernelId_Reduce_LLBuffer_LL16:
    busBytes = nRanks*nBytes*LL_BusFactor;
    break;

  // Broadcast: root sends to all ranks
  // Traffic pattern similar to AllGather but from one rank
  case ncclSymkKernelId_Broadcast_LLBuffer:
  case ncclSymkKernelId_Broadcast_LLBuffer_LL16:
    busBytes = nBytes*LL_BusFactor; // Root sends nBytes to all peers
    break;
  case ncclSymkKernelId_Broadcast_LLBufferMC:
    busBytes = nBytes*LL_BusFactor;
    busMultiplier = 1.1; // To beat non-MC version
    break;
  }

  nMaxBlocks = std::min<int>(nMaxBlocks, comm->config.maxCTAs);
  int nMinBlocks = comm->config.minCTAs;

  int nUserCTAs = std::min<int>(ncclSymkMaxBlocks, ncclParamSymCTAs());
  if (nUserCTAs > 0) nMinBlocks = nMaxBlocks = nUserCTAs;

  bool isLL = kernelMask_1Shot_LL>>k & 1 || kernelMask_2Shot_LL>>k & 1;
  bool isAG = kernelMask_AG>>k & 1;
  bool isAR = kernelMask_AR>>k & 1;
  bool isRD = kernelMask_RD>>k & 1;
  bool isBC = kernelMask_BC>>k & 1;
  constexpr double GBps = (1<<30)/1.e6;
  double baseLat, smBw, peakBw;
  if (comm->cudaArch < 1000) {
    baseLat = isLL ? 4.5 : 7.8;
    smBw = (isAR || isRD) ? 65*GBps : 44*GBps;
    peakBw = k == ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC ? 480*GBps : 320*GBps;
  } else {
    baseLat = isLL ? ((isAG || isBC) ? 8.5 : 11) : ((isAR || isRD) ? 19.5 : 13.0);
    smBw = 55*GBps;
    peakBw = k == ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC ? 1000*GBps : 600*GBps;
  }
  *nBlocks = nMaxBlocks;
  *timeUs = model(busBytes, baseLat, nMaxBlocks, smBw, busMultiplier, peakBw);
  // Use least number of blocks that puts us within a tolerance of peak performance.
  for (int bn = nMinBlocks; bn < nMaxBlocks; bn++) {
    double time = model(busBytes, baseLat, bn, smBw, busMultiplier, peakBw);
    if (time <= 1.025*(*timeUs)) {
      *nBlocks = bn;
      *timeUs = time;
      break;
    }
  }

  // Compute nWarps and gridDimY based on kernel type.
  // 1 warp = 32 threads, clamp user setting to [1, 16] (32..512 threads).
  constexpr int maxWarpsCap = 16;
  constexpr int minWarps = 1;
  int maxWarps = ncclParamSymKernelMaxWarps();
  if (maxWarps < minWarps) maxWarps = minWarps;
  if (maxWarps > maxWarpsCap) maxWarps = maxWarpsCap;

  if (isLamport2ShotL2Kernel(k)) {
    constexpr int bytesPerThread = 16;
    // Lamport 2-shot non-poison: use LL2lines calculation, gridDimY = nRanks
    // Constants from userbuffers.cu
    constexpr int LL2lines = 62;  // Number of cache lines processed per CTA
    size_t elementSize = 4; // Assume FP32 for now (will be refined by caller if needed)
    size_t elementsPerCta = LL2lines * 128 / elementSize;
    size_t elementsPerRank = (nBytes / elementSize) / nRanks;
    int ctasPerRank = (int)((elementsPerRank + elementsPerCta - 1) / elementsPerCta);
    int maxCtasPerRank = std::max(64 / nRanks, 1);
    ctasPerRank = std::min(ctasPerRank, maxCtasPerRank);
    *nBlocks = ctasPerRank;  // This becomes gridDimX
    *gridDimY = nRanks;
    *nWarps = maxWarps;
  } else if (isLLStyleKernel(k)) {

    int bytesPerThread = isLLBufferTwoshotKernel(k) ? 16 : 8;
    if (isLLBufferOneShotKernel(k)) {
      bytesPerThread >>= (nRanks / 16);
    }
    // LLBuffer, LL, and Lamport poison kernels: dynamic warps based on message size
    // Each thread processes 8 bytes, 1 warp = 32 threads
    *gridDimY = 1;

    if (isReduceScatterKernel(k)) nBytes = nRanks * nBytes;
    int totalThreadsNeeded = (int)(nBytes / bytesPerThread);
    if (totalThreadsNeeded <= 0) totalThreadsNeeded = 1;
    // Convert to warps: round up to next warp, clamp to [minWarps, maxWarps]
    int warpsNeeded = (totalThreadsNeeded + 31) / 32;
    int warpsPerBlock = std::min(warpsNeeded, maxWarps);
    warpsPerBlock = std::max(warpsPerBlock, minWarps);
    *nWarps = warpsPerBlock;

    // Compute nBlocks based on total warps needed
    int totalWarpsNeeded = (totalThreadsNeeded + 31) / 32;
    *nBlocks = (totalWarpsNeeded + warpsPerBlock - 1) / warpsPerBlock;
    *nBlocks = std::max(*nBlocks, nMinBlocks);
    *nBlocks = std::min(*nBlocks, nMaxBlocks);
  } else {
    // Other kernels: use default 16 warps (512 threads), gridDimY = 1
    *nWarps = maxWarps;
    *gridDimY = 1;
  }


  // Force the kernel to use the LLBuffer kernel if the message size is less than the threshold
  if (isLLBufferKernel(k) && nBytes <= NCCL_ONESHOT_LLBUFFER_KERNEL_THRESHOLD) {
    *timeUs = 0.1;
  }

  if (isLLBufferMCKernel(k) && nBytes <= NCCL_ONESHOT_LLBUFFER_KERNEL_THRESHOLD) {
    *timeUs = 0;
  }

  if (isLLBufferTwoshotKernel(k) && nBytes <= NCCL_TWOSHOT_LLBUFFER_KERNEL_THRESHOLD) {
    *timeUs = 0.1;
  }
}


ncclResult_t ncclSymkAllocAccumBuffer(struct ncclComm* comm) {
  struct ncclSymkState* symk = &comm->symkState;

  // Only allocate if not already allocated
  if (symk->kcomm.accumBuffer != nullptr || symk->kcomm.lamport2ShotAccumBuffer != nullptr) {
    return ncclSuccess;
  }

  // Sets the accumulation buffer size
  // Align to page boundary for better performance
  size_t slotStrideBytes = alignUp((size_t)REDUCTION_BUFFER_SIZE, (size_t)4096);
  size_t accumBufferSize = slotStrideBytes * ncclSymkLamportAccumSlots;

  uint8_t* accumDevBase;
  uint8_t* lamport2ShotAccumDevBase;
  ncclWindow_vidmem* accumWinDev;
  ncclWindow_vidmem* lamport2ShotAccumWinDev;

  // Ensure symmetric memory runtime is initialized
  NCCLCHECK(ncclDevrInitOnce(comm));

  // Allocate and register memory for the shared LLBuffer/scratch accumulation buffer.
  NCCLCHECK(ncclMemAlloc((void**)&accumDevBase, accumBufferSize));
  if (symk->lamportPoisonInit) {
    // Use ncclLLPoisonBuffer for consistent poisoning with ncclLLBuffer API
    // Data type is configurable via NCCL_SYM_LAMPORT_POISON_DTYPE (numeric NCCL_PARAM mapping)
    ncclDataType_t poisonDtype = getLamportPoisonDataType();
    NCCLCHECK(ncclLLPoisonBuffer(accumDevBase, accumBufferSize, poisonDtype));
    INFO(NCCL_INIT, "Poisoned reduction buffer with dtype %d", (int)poisonDtype);
  } else {
    // Initialize the value in the accumulation buffer to 0
    CUDACHECK(cudaMemset(accumDevBase, 0, accumBufferSize));
  }
  NCCLCHECK(ncclDevrWindowRegisterInGroup(comm, accumDevBase, accumBufferSize,
                                         NCCL_WIN_COLL_SYMMETRIC, &accumWinDev));

  // Allocate and register dedicated Lamport 2-shot accumulation buffer.
  // This buffer must always start at zero, independent of LLBuffer poison mode.
  NCCLCHECK(ncclMemAlloc((void**)&lamport2ShotAccumDevBase, accumBufferSize));
  CUDACHECK(cudaMemset(lamport2ShotAccumDevBase, 0, accumBufferSize));
  NCCLCHECK(ncclDevrWindowRegisterInGroup(comm, lamport2ShotAccumDevBase, accumBufferSize,
                                         NCCL_WIN_COLL_SYMMETRIC, &lamport2ShotAccumWinDev));

  // Store device-side references.
  symk->kcomm.accumBuffer = accumWinDev;
  symk->kcomm.lamport2ShotAccumBuffer = lamport2ShotAccumWinDev;
  symk->kcomm.lamportAccumStrideBytes = slotStrideBytes;
  symk->kcomm.lamportAccumSlotCount = ncclSymkLamportAccumSlots;
  symk->kcomm.lamportAccumOffset = 0;

  // Save host-visible device base pointers
  symk->lamportAccumDevBase = accumDevBase;
  symk->lamport2ShotAccumDevBase = lamport2ShotAccumDevBase;

  symk->lamportSlotStrideBytes = slotStrideBytes;
  symk->lamportSlotCount = ncclSymkLamportAccumSlots;
  symk->lamportLastSlot = (ncclSymkLamportAccumSlots == 0) ? 0 : (ncclSymkLamportAccumSlots - 1);
  symk->lamport2ShotLastSlot = (ncclSymkLamportAccumSlots == 0) ? 0 : (ncclSymkLamportAccumSlots - 1);

  INFO(NCCL_INIT, "Allocated Lamport accumulation buffers: rank %d, slots %d, per-slot %zu MB",
       comm->rank, ncclSymkLamportAccumSlots, slotStrideBytes / (1024 * 1024));

  return ncclSuccess;
}

ncclResult_t ncclSymkInitOnce(struct ncclComm* comm) {
  struct ncclSymkState* symk = &comm->symkState;
  if (!symk->initialized) {
    symk->initialized = true;
    struct ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;

    // Initialize accumulation buffer field
    symk->kcomm.accumBuffer = nullptr;
    symk->kcomm.lamport2ShotAccumBuffer = nullptr;
    symk->kcomm.lamportAccumOffset = 0;
    symk->kcomm.lamportAccumSlotCount = ncclSymkLamportAccumSlots;
    int cfgMaxConcurrentEpochs = ncclParamSymMaxConcurrentEpochs();
    symk->kcomm.maxConcurrentEpochs = (cfgMaxConcurrentEpochs > 0) ? (uint32_t)cfgMaxConcurrentEpochs : 0;
    symk->lamportSlotCount = ncclSymkLamportAccumSlots;
    symk->lamportSlotStrideBytes = alignUp((size_t)REDUCTION_BUFFER_SIZE, (size_t)4096);
    symk->kcomm.lamportAccumStrideBytes = symk->lamportSlotStrideBytes;
    symk->lamportLastSlot = (symk->lamportSlotCount == 0) ? 0 : (symk->lamportSlotCount - 1);
    symk->lamport2ShotLastSlot = (symk->lamportSlotCount == 0) ? 0 : (symk->lamportSlotCount - 1);
    symk->lamportAccumDevBase = nullptr;
    symk->lamport2ShotAccumDevBase = nullptr;
    symk->llBufferEpoch = 2;
    // Poisoning is normally controlled by SYM_LAMPORT_POISON_INIT. Additionally, if the user
    // explicitly forces the ncclLLBuffer-based AllReduce_LL with poison sync, we must poison
    // the accumulation buffer at init for correctness (do NOT do this in enqueue).
    // LLBuffer kernels in Poison sync mode (sync=0) require poisoned accumulation buffer.
    // Always poison if:
    // 1. User explicitly requested an LLBuffer kernel with Poison mode, OR
    // 2. LLBuffer kernels with Poison mode are available for auto-selection
    // This ensures correctness when LLBuffer is auto-selected without explicit NCCL_SYM_KERNEL.
    bool forcePoison = false;
    int sync = getLLBufferSyncMode(); // 0=Poison, 1=LL
    if (sync == 0) {
      // Poison sync mode: always poison the buffer since LLBuffer kernels may be auto-selected
      forcePoison = true;
    }
    symk->lamportPoisonInit = ncclParamSymLamportPoisonInit() && forcePoison;

    // struct ncclDevCommRequirements reqs = {};
    reqs.lsaMultimem = comm->nvlsSupport;
    reqs.barrierCount = ncclSymkMaxBlocks;

    struct ncclDevResourceRequirements lla2aReq;
    ncclLLA2ACreateRequirement(
      ncclSymkMaxBlocks, ncclLLA2ACalcSlots(ncclTeamLsa(comm).nRanks*ncclSymkMaxThreads, ncclSymkLLMaxEltSize),
      &symk->kcomm.lsaLLA2A, &lla2aReq
    );
    lla2aReq.next = reqs.resourceRequirementsList;
    reqs.resourceRequirementsList = &lla2aReq;

    struct ncclDevResourceRequirements railSignalReq = {};
    if (comm->nNodes > 1) {
      uint32_t railSignalCount = ncclTeamRail(comm).nRanks * ncclSymkMaxBlocks;

      railSignalReq.bufferSize = 0;
      railSignalReq.bufferAlign = 0;
      railSignalReq.outBufferHandle = nullptr;
      railSignalReq.ginSignalCount = railSignalCount;
      railSignalReq.ginCounterCount = 0;
      railSignalReq.outGinSignalStart = &symk->kcomm.ginSyncHandle.railSignals;
      railSignalReq.next = reqs.resourceRequirementsList;
      reqs.resourceRequirementsList = &railSignalReq;
    }
    NCCLCHECK(ncclDevrCommCreateInternal(comm, &reqs, &symk->kcomm.devComm));

    // Allocate accumulation buffer for Lamport 2-shot operations
    NCCLCHECK(ncclSymkAllocAccumBuffer(comm));
  }
  return ncclSuccess;
}

ncclResult_t ncclSymkFinalize(struct ncclComm* comm) {
  struct ncclSymkState* symk = &comm->symkState;
  if (symk->initialized) {
    NCCLCHECK(ncclDevCommDestroy(comm, &symk->kcomm.devComm));

    // Cleanup accumulation buffers
    if (symk->kcomm.accumBuffer || symk->kcomm.lamport2ShotAccumBuffer) {
      INFO(NCCL_INIT, "Cleaning up Lamport accumulation buffers: rank %d", comm->rank);
      // The symmetric memory system handles cleanup automatically when the communicator is destroyed
      symk->kcomm.accumBuffer = nullptr;
      symk->kcomm.lamport2ShotAccumBuffer = nullptr;
      symk->kcomm.lamportAccumOffset = 0;
      symk->lamportLastSlot = 0;
      symk->lamportSlotCount = 0;
      symk->lamportSlotStrideBytes = 0;
      symk->lamport2ShotLastSlot = 0;
      symk->lamportPoisonInit = false;
      symk->lamportAccumDevBase = nullptr;
      symk->lamport2ShotAccumDevBase = nullptr;
    }
  }
  return ncclSuccess;
}

static bool ncclSymkImplemented(ncclFunc_t coll, int/*ncclDevRedOp_t*/ red, ncclDataType_t ty) {
  bool isFloat;
  switch (ty) {
  case ncclFloat64:
  case ncclFloat32:
  case ncclFloat16:
  case ncclBfloat16:
  case ncclFloat8e4m3:
  case ncclFloat8e5m2:
    isFloat = true;
    break;
  default:
    isFloat = false;
    break;
  }

  switch (coll) {
  case ncclFuncAllGather:
    return true;
  case ncclFuncBroadcast:
    return false;
  case ncclFuncReduce:
    // return red == ncclDevSum && isFloat && ty != ncclFloat64;
    
  case ncclFuncAllReduce:
  case ncclFuncReduceScatter:
    return red == ncclDevSum && ty != ncclInt8 && ty != ncclUint8;
  default:
    return false;
  }
}

// Helper to build mask for all rank variants of a base LLBuffer kernel.
// count includes the base variant (e.g., 5 for base+R4+R8+R16+R32, 2 for base+R8).
static uint64_t llbufferRankMask(ncclSymkKernelId base, int count = 5) {
  uint64_t mask = 0;
  for (int i = 0; i < count; i++) mask |= 1ull << (base + i);
  return mask;
}

static uint64_t ncclSymkMask(struct ncclComm* comm, ncclFunc_t coll, int/*ncclDevRedOp_t*/ red, ncclDataType_t ty, size_t nElts) {
  uint64_t kmask = kernelMask_coll(coll);
  kmask &= kernelMask_user();

  bool hasSTMC = comm->nvlsSupport;
  bool hasLDMC = false;
  if (comm->nvlsSupport) {
    switch (ty) {
    case ncclInt32:
    case ncclUint32:
    case ncclInt64:
    case ncclUint64:
    case ncclFloat16:
    case ncclBfloat16:
      hasLDMC = red == ncclDevSum || red == ncclDevMinMax;
      break;
    case ncclFloat8e4m3:
    case ncclFloat8e5m2:
      hasLDMC = red == ncclDevSum || red == ncclDevMinMax;
      hasLDMC &= comm->compCap >= 100;
      break;
    case ncclFloat:
    case ncclDouble:
      hasLDMC = red == ncclDevSum;
      break;
    default: break;
    }
  }
  if (!hasSTMC) kmask &= ~kernelMask_STMC;
  if (!hasLDMC) kmask &= ~kernelMask_LDMC;
  // LLBuffer multimem kernels require NVLS support
  if (!hasSTMC) kmask &= ~kernelMask_LLBufferMC;

  // LLBuffer kernels have two compiled variants (Poison and LL16), each with rank specializations.
  // Select exactly one sync mode based on NCCL_SYM_LLBUFFER_SYNC.
  {
    int sync = getLLBufferSyncMode(); // 0=Poison, 1=LL
    char const* name = ncclGetEnv("NCCL_SYM_KERNEL");
    bool forcedLLBuffer = (name != nullptr) && (
      strcmp(name, "AllReduce_LLBuffer") == 0 ||
      strcmp(name, "AllReduce_LLBuffer_Twoshot") == 0 ||
      strcmp(name, "ReduceScatter_LLBuffer") == 0 ||
      strcmp(name, "AllGather_LLBuffer") == 0 ||
      strcmp(name, "Reduce_LLBuffer") == 0 ||
      strcmp(name, "Broadcast_LLBuffer") == 0
    );

    if (sync == 0) {
      // Poison mode: disable LL16 variants (all rank specializations + MC)
      kmask &= ~llbufferRankMask(ncclSymkKernelId_AllReduce_LLBuffer_LL16);
      kmask &= ~(1ull<<ncclSymkKernelId_AllReduce_LLBuffer_LL16MC);
      kmask &= ~llbufferRankMask(ncclSymkKernelId_ReduceScatter_LLBuffer_LL16, 2);
      kmask &= ~llbufferRankMask(ncclSymkKernelId_AllGather_LLBuffer_LL16, 2);
      kmask &= ~llbufferRankMask(ncclSymkKernelId_Reduce_LLBuffer_LL16, 2);
      kmask &= ~llbufferRankMask(ncclSymkKernelId_Broadcast_LLBuffer_LL16, 2);
      kmask &= ~llbufferRankMask(ncclSymkKernelId_AllReduce_LLBuffer_Twoshot_LL16, 2);
      kmask &= ~(1ull<<ncclSymkKernelId_AllReduce_LLBuffer_Twoshot_LL16MC);
      // For correctness, poison mode requires that the accumulation buffer is poisoned at init.
      if (!forcedLLBuffer && ncclParamSymLamportPoisonInit() == 0) {
        kmask &= ~llbufferRankMask(ncclSymkKernelId_AllReduce_LLBuffer);
        kmask &= ~llbufferRankMask(ncclSymkKernelId_ReduceScatter_LLBuffer, 2);
        kmask &= ~llbufferRankMask(ncclSymkKernelId_AllGather_LLBuffer, 2);
        kmask &= ~llbufferRankMask(ncclSymkKernelId_Reduce_LLBuffer, 2);
        kmask &= ~llbufferRankMask(ncclSymkKernelId_Broadcast_LLBuffer, 2);
      }
    } else {
      // LL mode (sync=1): ONLY allow LLBuffer_LL16 kernels
      // Build a mask of all LLBuffer_LL16 kernels and intersect with current mask
      uint64_t llbufferLL16Mask =
        llbufferRankMask(ncclSymkKernelId_AllReduce_LLBuffer_LL16) |
        (1ull<<ncclSymkKernelId_AllReduce_LLBuffer_LL16MC) |
        llbufferRankMask(ncclSymkKernelId_AllReduce_LLBuffer_Twoshot_LL16, 2) |
        (1ull<<ncclSymkKernelId_AllReduce_LLBuffer_Twoshot_LL16MC) |
        llbufferRankMask(ncclSymkKernelId_ReduceScatter_LLBuffer_LL16, 2) |
        llbufferRankMask(ncclSymkKernelId_AllGather_LLBuffer_LL16) |
        llbufferRankMask(ncclSymkKernelId_Reduce_LLBuffer_LL16, 2) |
        llbufferRankMask(ncclSymkKernelId_Broadcast_LLBuffer_LL16, 2);
      kmask &= llbufferLL16Mask;
    }
  }

  size_t nBytes = nElts*ncclTypeSize(ty);

  size_t nBusBytes = (coll == ncclFuncAllReduce ? 1 : comm->nRanks)*nBytes;

  // Check if user explicitly forced a kernel via NCCL_SYM_KERNEL
  // If so, skip size limits to honor user's explicit request
  bool userForcedKernel = (kernelMask_user() != ((1ull<<(int)ncclSymkKernelId_Count)-1));
  // Disable all LL style kernels if the message size is too large
  // For AllReduce, only allow LL-based symmetric kernels (1-shot and 2-shot).
  // Non-LL symmetric kernels (RSxLD_AGxST, Lamport, SOL, etc.) are excluded so
  // that large messages naturally fall back to the legacy ring algorithm once
  // the LL size filters below clear both LL masks.
  if (!userForcedKernel && coll == ncclFuncAllReduce) {
    kmask &= (kernelMask_1Shot_LL | kernelMask_2Shot_LL);
  }

  if (!userForcedKernel && nBytes >= NCCL_ONESHOT_LLBUFFER_KERNEL_THRESHOLD) {
    kmask &= ~kernelMask_1Shot_LL;
  }

  // Prefer 1-shot for small messages: disable 2-shot below the 1-shot threshold
  if (!userForcedKernel && nBytes < NCCL_ONESHOT_LLBUFFER_KERNEL_THRESHOLD) {
    kmask &= ~kernelMask_2Shot_LL;
  }

  if (!userForcedKernel && nBytes > NCCL_TWOSHOT_LLBUFFER_KERNEL_THRESHOLD) {
    kmask &= ~kernelMask_2Shot_LL;
  }

  if (!userForcedKernel) {
    // LL kernels use 32-bit ints to track element counts and indices.
    if (nBusBytes >= (size_t(2)<<30)) kmask &= ~(kernelMask_1Shot_LL | kernelMask_2Shot_LL);
    // Any kernel might use 32-bit int to track unrolled loop chunks (which are going
    // to be at least 32 bytes per chunk)
    if (nBusBytes >= 32*(size_t(2)<<30)) kmask = 0;
  }

  kmask &= (comm->nNodes > 1) ? kernelMask_Gin : ~kernelMask_Gin;

  return kmask;
}

bool ncclSymkAvailable(struct ncclComm* comm, ncclFunc_t coll, int/*ncclDevRedOp_t*/ red,
                       ncclDataType_t ty, size_t nElts) {
  if (!comm->isAllDirectNvlink)
    return false;
  if (!ncclSymkImplemented(coll, red, ty))
    return false;

  return (ncclSymkMask(comm, coll, red, ty, nElts) != 0);
}

ncclResult_t ncclSymkPickKernel(
    struct ncclComm* comm, ncclFunc_t coll, int/*ncclDevRedOp_t*/ red, ncclDataType_t ty,
    size_t nEltsTotal, size_t nEltsMax, int nWorks, ncclSymRegType_t winRegType,
    float* estTimeUs, ncclSymkKernelId* kernelId, int* nBlocks, int* nWarps, int* gridDimY, bool* forced
  ) {
  uint64_t kmask = ncclSymkMask(comm, coll, red, ty, nEltsMax);

  *forced = !(kernelMask_user() == (1<<(int)ncclSymkKernelId_Count)-1);
  // We currently don't support grouping for LL kernels.
  if (nWorks > 1) {
    kmask &= ~kernelMask_1Shot_LL;
    kmask &= ~kernelMask_2Shot_LL;
  }

  if (coll == ncclFuncAllReduce) {
    if (winRegType != ncclSymSendRegRecvReg) {
      // 2-shot kernels write to peers' output buffers via peerPtr(), which
      // requires the output window to be symmetrically registered. Only
      // allow 1-shot LL kernels when buffers are not fully registered.
      kmask &= kernelMask_1Shot_LL;
    }
  } else if (coll == ncclFuncAllGather) {
    if (winRegType != ncclSymSendRegRecvReg && winRegType != ncclSymSendNonregRecvReg) {
      kmask &= kernelMask_1Shot_LL;
    }
    if (winRegType != ncclSymSendRegRecvReg && comm->nNodes > 1) kmask &= ~kernelMask_Gin;
  } else if (coll == ncclFuncReduceScatter) {
    if (winRegType != ncclSymSendRegRecvReg && winRegType != ncclSymSendRegRecvNonreg) {
      kmask &= kernelMask_1Shot_LL;
    }
  }

  ncclSymkKernelId bestKernel = ncclSymkKernelId_Count;
  float bestTime = 1.e30f;
  int bestBlocks = 999;
  int bestWarps = 16;  // 512 threads = 16 warps
  int bestGridDimY = 1;
  size_t nBytes = nEltsTotal*ncclTypeSize(ty);
  int nRanks = comm->nRanks;

  constexpr float smPenalty = .025f; // 2.5% percent increase in time per SM
  uint64_t kmaskRemain = kmask;
  while (kmaskRemain != 0) {
    ncclSymkKernelId k = (ncclSymkKernelId)popFirstOneBit(&kmaskRemain);

    // For LLBuffer kernels, select the rank-specialized variant
    if (isLLBufferRankSpecialized(k)) {
      k = getLLBufferRankKernel(k, nRanks);
    }

    float kTime;
    int kBlocks, kWarps, kGridDimY;
    queryModel(comm, k, nBytes, &kTime, &kBlocks, &kWarps, &kGridDimY);
    if (kTime*(1.0f + smPenalty*kBlocks) < bestTime*(1.0f + smPenalty*bestBlocks)) {
      bestKernel = k;
      bestTime = kTime;
      bestBlocks = kBlocks;
      bestWarps = kWarps;
      bestGridDimY = kGridDimY;
    }
  }

  *kernelId = bestKernel;
  *estTimeUs = kmask==0 || kernelMask_user() == (1ull<<ncclSymkKernelId_Count)-1 ? bestTime : 0.0f;
  *nBlocks = bestBlocks;
  *nWarps = bestWarps;
  *gridDimY = bestGridDimY;

  if (bestKernel != ncclSymkKernelId_Count) {
    INFO(NCCL_TUNING, "SymKernel: %s nRanks=%d nElts=%zu nBytes=%zu gridDim=(%d,%d) nWarps=%d",
         ncclSymkKernelIdToString(bestKernel), nRanks, nEltsTotal, nBytes, bestBlocks, bestGridDimY, bestWarps);
  }
  return ncclSuccess;
}

const char* ncclSymkKernelIdToString(int kernelId) {
  if (kernelId < 0 || kernelId >= ncclSymkKernelId_Count) {
    return "Unknown";
  }
  return kernelName[kernelId];
}

/* this function fills in the devWork except nextWorkOffset */
ncclResult_t ncclSymkMakeDevWork(struct ncclComm* comm, struct ncclTaskColl* task, struct ncclSymkDevWork* outDevWork) {
  outDevWork->rootRank = task->root;
  outDevWork->redOpArg = task->opDev.scalarArg;
  outDevWork->nElts = task->count;
  outDevWork->inputWin = task->sendWin ? task->sendWin->vidmem : nullptr;
  outDevWork->inputOff = task->sendWin ? (uint8_t*)task->sendbuff - (uint8_t*)task->sendWin->userPtr : (size_t)task->sendbuff;
  outDevWork->outputWin = task->recvWin ? task->recvWin->vidmem : nullptr;
  outDevWork->outputOff = task->recvWin ? (uint8_t*)task->recvbuff - (uint8_t*)task->recvWin->userPtr : (size_t)task->recvbuff;
  outDevWork->sChannelId = 0xffff;
  outDevWork->nChannels = 0;
  return ncclSuccess;
}


ncclResult_t ncclGetSymRegType(struct ncclDevrWindow* sendWin, struct ncclDevrWindow* recvWin, ncclSymRegType_t* winRegType) {
  bool isSendSymmReg = false;
  bool isRecvSymmReg = false;
  if (sendWin && (sendWin->winFlags & NCCL_WIN_COLL_SYMMETRIC)) isSendSymmReg = true;
  if (recvWin && (recvWin->winFlags & NCCL_WIN_COLL_SYMMETRIC)) isRecvSymmReg = true;
  // determine the registration type
  if (!isSendSymmReg && !isRecvSymmReg) {
    *winRegType = ncclSymSendNonregRecvNonreg;
  } else if (isSendSymmReg && !isRecvSymmReg) {
    *winRegType = ncclSymSendRegRecvNonreg;
  } else if (!isSendSymmReg && isRecvSymmReg) {
    *winRegType = ncclSymSendNonregRecvReg;
  } else if (isSendSymmReg && isRecvSymmReg) {
    *winRegType = ncclSymSendRegRecvReg;
  }
  return ncclSuccess;
}
