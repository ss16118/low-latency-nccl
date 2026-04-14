#include "sym_kernels.h"
#include "nccl_device.h"
#include "kernel.cuh"
#include "primitives.cuh"
#include "device.h"

#include <cuda_fp16.h>
#include <limits.h>

#define CACHELINE_SIZE 128
#define MAX_SUB_LOG 5


/**
 * A helper function to perform a warp shuffle specialized for any type T.
 * It is implemented to support the BytePack type
 */
template <typename T>
NCCL_DEVICE_INLINE T shflXorSync(unsigned mask, T val, int laneMask, int width=WARP_SIZE) {
  #if __cpp_if_constexpr
  if constexpr (sizeof(T) == 2) {
  #else
  if (sizeof(T) == 2) {
  #endif
    union { T tmp; __half h; };
    tmp = val;
    h = __shfl_xor_sync(mask, h, laneMask, width);
    return tmp;
  }
  
  #if __cpp_if_constexpr
  if constexpr (sizeof(T) == 4) {
  #else
  if (sizeof(T) == 4) {
  #endif
    union { T tmp; unsigned int f; };
    tmp = val;
    f = __shfl_xor_sync(mask, f, laneMask, width);
    return tmp;
  }
  
  #if __cpp_if_constexpr
  if constexpr (sizeof(T) == 8) {
  #else
  if (sizeof(T) == 8) {
  #endif
    union { T tmp; unsigned long long f; };
    tmp = val;
    f = __shfl_xor_sync(mask, f, laneMask, width);
    return tmp;
  }
  
  printf("ERROR: Unsupported type size: %ld in shflXorSync\n", sizeof(T));
  return val;
}



#define NCCL_SYM_ATOMICS_EXPERIMENTAL
#include <type_traits>

// Forward declarations for ncclSymkRun_AllReduce_LL_impl
template<ncclLLSyncMode Mode, bool Multimem, int Unroll, template<typename> typename Red, typename T,
         int SubRanks, int SubLog>
__device__ __forceinline__ void ncclSymkRun_AllReduce_LL_impl(ncclSymkDevWorkArgs const* args);


template<ncclLLSyncMode Mode, bool Multimem, int Unroll, template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_LLBuffer_Twoshot_impl(ncclSymkDevWorkArgs const* args);

template<int BytePerPack, int UnrollPacks, int UnrollPeers, typename T, typename Red>
static __device__ __forceinline__ void allreduceDeep(
    ncclSymkArgsHandler const& handler, int tn, int t,
    bool waitNeeded, ncclLsaBarrierSession<ncclCoopCta>& bar,
    Red red, ncclSymPtr<char> input, ncclSymPtr<char> output, int32_t nIters
  ) {
  using Pack = BytePack<BytePerPack>;
  using Acc = typename Red::EltType;
  using AccPack = BytePack<BytePerPack*sizeof(Acc)/sizeof(T)>;

  ncclTeam world = ncclTeamWorld(handler.comm);
  int wn = tn/WARP_SIZE;
  int w = t/WARP_SIZE;
  int lane = t%WARP_SIZE;
  int const& rank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;

  ncclSymPtr<Pack> inpPacks = (ncclSymPtr<Pack>)input + intptr_t(w)*UnrollPacks*WARP_SIZE + lane;
  ncclSymPtr<Pack> outPacks = (ncclSymPtr<Pack>)output + intptr_t(w)*UnrollPacks*WARP_SIZE + lane;
  Pack acc0[UnrollPacks];

  nIters -= w;
  if (0 < nIters) {
    #pragma unroll
    for (int u=0; u < UnrollPacks; u++) {
      acc0[u] = inpPacks.peerPtr(world, rank)[u*WARP_SIZE];
    }
  }

  if (waitNeeded) bar.wait(ncclCoopCta(), cuda::memory_order_relaxed);

  if (0 < nIters) {
    while (true) {
      AccPack acc1[UnrollPacks];
      int r = rank;
      if (++r == nRanks) r = 0;
      { Pack tmp1[UnrollPacks];
        #pragma unroll
        for (int u=0; u < UnrollPacks; u++) {
          tmp1[u] = inpPacks.peerPtr(world, r)[u*WARP_SIZE];
        }
        #pragma unroll
        for (int u=0; u < UnrollPacks; u++) {
          acc1[u] = applyReduce(red, applyCast<T, Acc>(acc0[u]), applyCast<T, Acc>(tmp1[u]));
        }
      }

      if (++r == nRanks) r = 0;

      int dr = 2;
      #pragma unroll 2
      for (int partial=0; partial <= 1; partial++) {
        #pragma unroll 1
        for (int i = 0;
             partial ? i < 1 : (dr + UnrollPeers <= nRanks);
             partial ? i++ : (dr += UnrollPeers)) {
          if (partial && dr == nRanks) break;

          Pack tmp1[UnrollPeers][UnrollPacks];
          #pragma unroll
          for (int ur=0; ur < UnrollPeers-partial; ur++) {
            if (partial && ur!=0 && dr+ur == nRanks) break;
            #pragma unroll UnrollPacks
            for (int u=0; u < UnrollPacks; u++) {
              tmp1[ur][u] = inpPacks.peerPtr(world, r)[u*WARP_SIZE];
            }
            if (++r == nRanks) r = 0;
          }
          #pragma unroll
          for (int ur=0; ur < UnrollPeers-partial; ur++) {
            if (partial && ur!=0 && dr+ur == nRanks) break;
            #pragma unroll UnrollPacks
            for (int u=0; u < UnrollPacks; u++) {
              acc1[u] = applyReduce(red, acc1[u], applyCast<T, Acc>(tmp1[ur][u]));
            }
          }
        }
      }

      #pragma unroll
      for (int u=0; u < UnrollPacks; u++) acc0[u] = applyCast<Acc, T>(acc1[u]);

      dr = 0;
      r = rank;
      #pragma unroll 2
      for (int partial=0; partial <= 1; partial++) {
        #pragma unroll 1
        for (int i = 0;
             partial ? i < 1 : (dr + UnrollPeers <= nRanks);
             partial ? i++ : (dr += UnrollPeers)) {
          #pragma unroll
          for (int ur=0; ur < UnrollPeers-partial; ur++) {
            if (partial && dr == nRanks) break;
            #pragma unroll UnrollPacks
            for (int u=0; u < UnrollPacks; u++) {
              outPacks.peerPtr(world, r)[u*WARP_SIZE] = acc0[u];
            }
            if (++r == nRanks) r = 0;
          }
        }
      }

      inpPacks += intptr_t(wn)*UnrollPacks*WARP_SIZE;
      outPacks += intptr_t(wn)*UnrollPacks*WARP_SIZE;
      nIters -= wn;
      if (nIters <= 0) break;

      // Load data for next iteration.
      #pragma unroll
      for (int u=0; u < UnrollPacks; u++) {
        acc0[u] = inpPacks.peerPtr(world, rank)[u*WARP_SIZE];
      }
    }
  }
}

// Atomic macros for vectorized operations (similar to userbuffers.cu)
#define NCCL_ATOMIC_ADD_F32(val, ptr) \
  atomicAdd(ptr, val)

#define NCCL_ATOMIC_ADD_V4F32(val, ptr) \
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(ptr), \
               "f"(val.x), "f"(val.y), "f"(val.z), "f"(val.w) \
               : "memory")

#define NCCL_ATOMIC_ADD_V4F32_MC(val, ptr) \
  asm volatile("multimem.red.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(ptr), \
               "f"(val.x), "f"(val.y), "f"(val.z), "f"(val.w) \
               : "memory")

#define NCCL_ATOMIC_ADD_V4BF16X2(val, ptr) \
  asm volatile("red.global.v4.bf16x2.add.noftz [%0], {%1, %2, %3, %4};" ::"l"(ptr), \
               "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w) \
               : "memory")

#define NCCL_ATOMIC_ADD_V4FP16X2(val, ptr) \
  asm volatile("red.global.v4.f16x2.add.noftz [%0], {%1, %2, %3, %4};" ::"l"(ptr), \
               "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w) \
               : "memory")


#define NCCL_ATOMIC_ADD_V2F64(val, ptr) \
  do { \
    atomicAdd(reinterpret_cast<double*>(ptr) + 0, (val).x); \
    atomicAdd(reinterpret_cast<double*>(ptr) + 1, (val).y); \
  } while (0)


template <typename T>
NCCL_DEVICE_INLINE void atomicAdd128(BytePack<16>* ptr, BytePack<16>& val) {
  #if __cpp_if_constexpr
  if constexpr (std::is_same<T, float>::value) {
    float4 const* fptr = reinterpret_cast<float4 const*>(ptr);
    float4 fval = reinterpret_cast<float4 const&>(val);
    NCCL_ATOMIC_ADD_V4F32(fval, fptr);
  } else if constexpr (std::is_same<T, __half>::value) {
    uint4 const* hptr = reinterpret_cast<uint4 const*>(ptr);
    uint4 hval = reinterpret_cast<uint4 const&>(val);
    NCCL_ATOMIC_ADD_V4FP16X2(hval, hptr);
#if defined(__CUDA_BF16_TYPES_EXIST__) || defined(__CUDACC__)
  } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
    uint4 const* uptr = reinterpret_cast<uint4 const*>(ptr);
    uint4 uval = reinterpret_cast<uint4 const&>(val);
    NCCL_ATOMIC_ADD_V4BF16X2(uval, uptr);
#endif
  } else {
    printf("ERROR: Unsupported type size: %ld in atomicAdd128\n", sizeof(T));
  }
  #else
  if (std::is_same<T, float>::value) {
    float4 const* fptr = reinterpret_cast<float4 const*>(ptr);
    float4 fval = reinterpret_cast<float4 const&>(val);
    NCCL_ATOMIC_ADD_V4F32(fval, fptr);
  } else if (std::is_same<T, __half>::value) {
    uint4 const* hptr = reinterpret_cast<uint4 const*>(ptr);
    uint4 hval = reinterpret_cast<uint4 const&>(val);
    NCCL_ATOMIC_ADD_V4FP16X2(hval, hptr);
#if defined(__CUDA_BF16_TYPES_EXIST__) || defined(__CUDACC__)
  } else if (std::is_same<T, __nv_bfloat16>::value) {
    uint4 const* uptr = reinterpret_cast<uint4 const*>(ptr);
    uint4 uval = reinterpret_cast<uint4 const&>(val);
    NCCL_ATOMIC_ADD_V4BF16X2(uval, uptr);
#endif
  } else {
    printf("ERROR: Unsupported type size: %ld in atomicAdd128\n", sizeof(T));
  }
  #endif
}

template <typename T>
NCCL_DEVICE_INLINE T packLane0(const BytePack<16>& pack) {
  return *reinterpret_cast<T const*>(&pack);
}

template <typename T>
NCCL_DEVICE_INLINE void setPackLane0(BytePack<16>& pack, T value) {
  *reinterpret_cast<T*>(&pack) = value;
}


NCCL_DEVICE_INLINE void load128(const float4* ptr, float4 &val)
{
  asm volatile("ld.volatile.global.v4.f32 {%0, %1, %2, %3}, [%4];"
      : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w) : "l"(ptr));
}

NCCL_DEVICE_INLINE void load128_int(const uint4* ptr, uint4 &val)
{
  asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
      : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w) : "l"(ptr));
}


NCCL_DEVICE_INLINE void store128(float4* ptr, float4 val)
{
  asm volatile("st.global.cg.v4.f32 [%0], {%1, %2, %3, %4};"
      : : "l"(ptr), "f"(val.x), "f"(val.y), "f"(val.z), "f"(val.w) : "memory");
}


NCCL_DEVICE_INLINE void store128_clear(uint4* ptr)
{
  asm volatile("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};"
      : : "l"(ptr), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
}


NCCL_DEVICE_INLINE void store128_poison(float4* ptr)
{
  const uint32_t poison = NCCL_LAMPORT_INT;
  asm volatile("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};"
      : : "l"(ptr), "r"(poison), "r"(poison), "r"(poison), "r"(poison) : "memory");
}


NCCL_DEVICE_INLINE void store128_mc(float4* ptr, float4 val)
{
  asm volatile("multimem.st.global.v4.f32 [%0], {%1, %2, %3, %4};"
      : : "l"(ptr), "f"(val.x), "f"(val.y), "f"(val.z), "f"(val.w) : "memory");
}


NCCL_DEVICE_INLINE void load32(const float* ptr, float &val)
{
  asm volatile("ld.volatile.global.f32 %0, [%1];"
      : "=f"(val) : "l"(ptr) : "memory");
}

NCCL_DEVICE_INLINE void load32_int(const uint32_t* ptr, uint32_t &val)
{
  asm volatile("ld.volatile.global.u32 %0, [%1];"
      : "=r"(val) : "l"(ptr) : "memory");
}


NCCL_DEVICE_INLINE void store32(float* ptr, float val)
{
  asm volatile("st.volatile.global.f32 [%0], %1;"
      : : "l"(ptr), "f"(val) : "memory");
}

NCCL_DEVICE_INLINE void store32_poison(uint32_t* ptr)
{
  const uint32_t poison = NCCL_LAMPORT_INT;
  asm volatile("st.volatile.global.u32 [%0], %1;"
      : : "l"(ptr), "r"(poison) : "memory");
}


NCCL_DEVICE_INLINE void store32_mc(float* ptr, float val)
{
  asm volatile("multimem.st.global.f32 [%0], %1;"
      : : "l"(ptr), "f"(val) : "memory");
}


template<int UnrollPeers, typename Red, typename T>
static __device__ __forceinline__ void allreduceEnds(
    ncclSymkArgsHandler const& handler, int tn, int t, Red red,
    ncclSymPtr<T> input, ncclSymPtr<T> output,
    size_t nElts, uint32_t nPreElts, size_t nSufElts
  ) {
  using Acc = typename Red::EltType;

  ncclTeam world = ncclTeamWorld(handler.comm);
  int const& rank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;

  ncclSymPtr<BytePack<sizeof(T)>> inpPacks = (ncclSymPtr<BytePack<sizeof(T)>>)input;
  ncclSymPtr<BytePack<sizeof(T)>> outPacks = (ncclSymPtr<BytePack<sizeof(T)>>)output;

  #pragma unroll 1
  for (size_t i = t; i < nPreElts+nSufElts; i += tn) {
    size_t elt = i < nPreElts ? i : nElts-nSufElts-nPreElts+i;
    BytePack<sizeof(T)> acc0 = inpPacks.peerPtr(world, rank)[elt];
    BytePack<sizeof(Acc)> acc1;
    BytePack<sizeof(T)> tmp[UnrollPeers];
    int dr = 1;
    int r = rank+1;
    if (nRanks == r) r = 0;
    bool first = true;

    #pragma unroll 2
    for (int partial=0; partial <= 1; partial++) {
      #pragma unroll 1
      for (int j = 0;
           partial ? j < 1 : (dr + UnrollPeers <= nRanks);
           partial ? j++ : (dr += UnrollPeers)) {
        if (partial && dr == nRanks) break;

        #pragma unroll
        for (int u=0; u < UnrollPeers-partial; u++) {
          if (partial && u!=0 && dr+u == nRanks) break;
          tmp[u] = inpPacks.peerPtr(world, r)[elt];
          r += 1;
          if (r == nRanks) r = 0;
        }
        if (first) {
          first = false;
          acc1 = applyCast<T, Acc>(acc0);
        }
        #pragma unroll
        for (int u=0; u < UnrollPeers-partial; u++) {
          if (partial && u!=0 && dr+u == nRanks) break;
          acc1 = applyReduce(red, acc1, applyCast<T, Acc>(tmp[u]));
        }
      }
    }

    acc0 = applyCast<Acc, T>(acc1);
    dr = 0;
    r = rank;
    #pragma unroll 2
    for (int partial=0; partial <= 1; partial++) {
      #pragma unroll 1
      for (int j=0;
           partial ? j < 1 : (dr + UnrollPeers <= nRanks);
           partial ? j++ : (dr += UnrollPeers)) {
        #pragma unroll
        for (int u=0; u < UnrollPeers-partial; u++) {
          if (partial && dr+u == nRanks) break;
          outPacks.peerPtr(world, r)[elt] = acc0;
          r += 1;
          if (r == nRanks) r = 0;
        }
      }
    }
  }
}

template<typename Red, typename T>
static __device__ void allreduce(
    ncclSymkArgsHandler const& handler, int tn, int t, int nBlocks,
    bool waitNeeded, ncclLsaBarrierSession<ncclCoopCta>& bar,
    Red red, ncclSymPtr<T> input, ncclSymPtr<T> output, size_t nElts
  ) {
  int const& nRanks = handler.comm.nRanks;
  int const& nRanks_rcp32 = handler.nRanks_rcp32;
  size_t nBytes = nElts*sizeof(T);
  uint32_t nBlocks_rcp32 = nccl::utility::idivRcp32_upto64(nBlocks);
  uint32_t nRanks_nBlocks_rcp32 = nccl::utility::imulRcp32(nRanks, nRanks_rcp32, nBlocks, nBlocks_rcp32);

  uint32_t nPreBytes = (16u - input.offset)%16u;
  nPreBytes = min((size_t)nPreBytes, nBytes);
  uintptr_t cursor = nPreBytes;

  constexpr int MinWarpPerBlock = 4;

  if ((input.offset - output.offset)%16 == 0) {
    constexpr int BytePerPack = 16, UnrollPacks = 4, UnrollPeers = 2;
    constexpr int BytePerChunk = MinWarpPerBlock*UnrollPacks*WARP_SIZE*BytePerPack;
    uint32_t chunks = (nBytes-cursor)/BytePerChunk;
    chunks -= imodFast32(chunks, nRanks*nBlocks, nRanks_nBlocks_rcp32);
    if (chunks != 0) {
      uintptr_t cursorAfter = cursor + uintptr_t(chunks)*BytePerChunk;
      allreduceDeep<BytePerPack, UnrollPacks, UnrollPeers, T>(
        handler, tn, t, waitNeeded, bar, red,
        (ncclSymPtr<char>)input + cursor,
        (ncclSymPtr<char>)output + cursor,
        chunks*MinWarpPerBlock
      );
      cursor = cursorAfter;
      waitNeeded = false;
    }
  }

  if (sizeof(T) == 4 || (sizeof(T) < 4 && (input.offset - output.offset)%4 == 0)) {
    constexpr int BytePerPack = 4, UnrollPacks = 4, UnrollPeers = 4;
    constexpr int BytePerChunk = MinWarpPerBlock*UnrollPacks*WARP_SIZE*BytePerPack;
    uint32_t chunks = (nBytes-cursor)/BytePerChunk;
    chunks -= imodFast32(chunks, nRanks*nBlocks, nRanks_nBlocks_rcp32);
    if (chunks != 0) {
      uintptr_t cursorAfter = cursor + uintptr_t(chunks)*BytePerChunk;
      allreduceDeep<(sizeof(T) <= BytePerPack ? BytePerPack : 0), UnrollPacks, UnrollPeers, T>(
        handler, tn, t, waitNeeded, bar, red,
        (ncclSymPtr<char>)input + cursor,
        (ncclSymPtr<char>)output + cursor,
        chunks*MinWarpPerBlock
      );
      cursor = cursorAfter;
      waitNeeded = false;
    }
  }

  if (waitNeeded) bar.wait(ncclCoopCta(), cuda::memory_order_relaxed);

  constexpr int UnrollPeers = 8;
  size_t nSufElts = (nBytes-cursor)/sizeof(T);
  allreduceEnds<UnrollPeers>(handler, tn, t, red, input, output, nElts, nPreBytes/sizeof(T), nSufElts);
}

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_RSxLD_AGxST(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};
  ncclLsaBarrierSession<ncclCoopCta> bar{
    ncclCoopCta(), handler.comm, ncclTeamTagLsa(), blockIdx.x
  };
  Red<typename ncclSymkAccumType<Red, T, /*nvls=*/false>::Type> red(handler.devWork->redOpArg);

  int const& rank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;

  bar.arrive(ncclCoopCta(), cuda::memory_order_relaxed);

  bool waitNeeded = true;
  handler.forEachWork<T>(
      [&]__device__(int block, int nBlocks, size_t nElts, size_t nAllElts,
                    ncclSymPtr<T> input, ncclSymPtr<T> output) {
        // Threads numbered globally such that we round robin warps by rank then block.
        int gt = flattenIx(threadIdx.x%WARP_SIZE, WARP_SIZE,
                           rank, nRanks,
                           block, nBlocks,
                           threadIdx.x/WARP_SIZE, blockDim.x/WARP_SIZE);
        int gtn = nRanks*nBlocks*blockDim.x;

        allreduce(handler, gtn, gt, nBlocks, waitNeeded, bar, red, input, output, nElts);

        waitNeeded = false;
      }
    );

  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

template<typename Red, typename T>
static __device__ void allreduceMultimem(
    int tn, int t, Red red, T* input, T* output, size_t nElts
  ) {
  uintptr_t inputUptr = reinterpret_cast<uintptr_t>(input);
  uintptr_t outputUptr = reinterpret_cast<uintptr_t>(output);
  size_t nBytes = nElts*sizeof(T);

  constexpr int BytePerPack = LoadMultimem_BigPackSize<Red>::BigPackSize;
  uint32_t nPreBytes = (BytePerPack - inputUptr)%BytePerPack;
  nPreBytes = min((size_t)nPreBytes, nBytes);
  uintptr_t nSufBytes;

  if (alignof(T) == BytePerPack || (inputUptr-outputUptr)%BytePerPack == 0) {
    constexpr int UnrollPacks = 16*8/BytePerPack;
    constexpr int BytePerChunk = UnrollPacks*WARP_SIZE*BytePerPack;
    uintptr_t cursor = nPreBytes;
    int nChunks = (nBytes-cursor)/BytePerChunk;
    uintptr_t cursorAfter = cursor + uintptr_t(nChunks)*BytePerChunk;
    nSufBytes = nBytes - cursorAfter;
    cursor += (t/WARP_SIZE)*UnrollPacks*WARP_SIZE*BytePerPack;
    cursor += (t%WARP_SIZE)*BytePerPack;
    int nIters = nChunks - t/WARP_SIZE;
    #pragma unroll 1
    while (0 < nIters) {
      BytePack<BytePerPack> tmp[UnrollPacks];
      #pragma unroll
      for (int u=0; u < UnrollPacks; u++) {
        tmp[u] = applyLoadMultimem<Red, BytePerPack>(red, inputUptr + cursor + u*WARP_SIZE*BytePerPack);
      }
      #pragma unroll
      for (int u=0; u < UnrollPacks; u++) {
        multimem_st_global(outputUptr + cursor + u*WARP_SIZE*BytePerPack, tmp[u]);
      }
      cursor += tn*UnrollPacks*BytePerPack;
      nIters -= tn/WARP_SIZE;
    }
  } else {
    nPreBytes = 0;
    nSufBytes = nBytes;
  }

  // Get the prefix+suffix element one at a time.
  #pragma unroll 4
  for (uintptr_t i = t*sizeof(T); i < nPreBytes + nSufBytes; i += tn*sizeof(T)) {
    uintptr_t cursor = i < nPreBytes ? i : nBytes-nSufBytes+(i-nPreBytes);
    BytePack<sizeof(T)> val = applyLoadMultimem<Red, sizeof(T)>(red, inputUptr + cursor);
    multimem_st_global(outputUptr + cursor, val);
  }
}

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_RSxLDMC_AGxSTMC(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};
  ncclLsaBarrierSession<ncclCoopCta> bar{
    ncclCoopCta(), handler.comm, ncclTeamTagLsa(), blockIdx.x, /*multimem=*/true
  };

  Red<typename ncclSymkAccumType<Red, T, /*nvls=*/true>::Type> red(handler.devWork->redOpArg);

  int const& rank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;
  auto const& multimem = handler.comm.lsaMultimem;

  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  handler.forEachWork<T>(
      [&]__device__(int block, int nBlocks, size_t nElts, size_t nAllElts,
                    ncclSymPtr<T> input, ncclSymPtr<T> output) {
        // Threads numbered globally such that we round robin warps by rank then block.
        int gt = flattenIx(threadIdx.x%WARP_SIZE, WARP_SIZE,
                           rank, nRanks,
                           block, nBlocks,
                           threadIdx.x/WARP_SIZE, blockDim.x/WARP_SIZE);
        int gtn = nRanks*nBlocks*blockDim.x;

        allreduceMultimem(gtn, gt, red, input.multimemPtr(multimem), output.multimemPtr(multimem), nElts);
      }
    );

  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_AGxLL_R_impl(ncclSymkDevWorkArgs const* args, bool multimem) {
  ncclSymkArgsHandler handler{args};
  ncclLLA2ASession<ncclCoopCta> lla2a(
    ncclCoopCta(), handler.comm, ncclTeamLsa(handler.comm), handler.lsaLLA2A,
    blockIdx.x, ncclSymkMaxThreads, multimem, handler.comm.lsaMultimem
  );

  int const& rank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;
  using Acc = typename ncclSymkAccumType<Red, T, /*nvls=*/false>::Type;
  Red<Acc> red(handler.devWork->redOpArg);
  constexpr int BytesPerPack = 8;

  using Pack = BytePack<BytesPerPack>;
  using AccPack = BytePack<BytesPerPack*sizeof(Acc)/sizeof(T)>;
  constexpr int EltPerPack = BytesPerPack/sizeof(T);

  handler.singleWork<T>(
      [&]__device__(int nElts, int nAllElts,
                    ncclSymPtr<T> inputPtr, ncclSymPtr<T> outputPtr) {
        int nPacks = divUp(nElts, EltPerPack);

        T* input = (T*)inputPtr.localPtr();
        T* output = (T*)outputPtr.localPtr();

        bool packAligned = BytesPerPack <= alignof(T) || (nElts*sizeof(T) | (uintptr_t)input | (uintptr_t)output)%BytesPerPack == 0;

        ncclCoopCta cta;
        int t = threadIdx.x;
        int tn = ncclSymkMaxThreads;

        if (__builtin_expect(packAligned, true)) {
          #pragma unroll 1
          while (0 < nPacks) {
            if (t < nPacks) {
              int nIterPacks = min(nPacks, tn);
              Pack inp = loadPack<Pack>((Pack*)input, t, nPacks);
              lla2a.bcast(/*slot=*/nIterPacks*rank + t, inp);
              AccPack out = lla2a.template recvReduce</*Unroll=*/8, Pack>(
                /*slotStart=*/t, /*slotCount=*/nRanks, /*slotStride=*/nIterPacks,
                /*eltToAcc=*/[&] __device__ (Pack x)->AccPack {
                  return applyCast<T, Acc>(x);
                },
                /*reduce=*/[&] __device__ (AccPack a, AccPack b)->AccPack {
                  return applyReduce(red, a, b);
                }
              );
              storePack((Pack*)output, t, nPacks, applyCast<Acc, T>(out));
            }
            lla2a.endEpoch(cta);

            input += tn*EltPerPack;
            output += tn*EltPerPack;
            nPacks -= tn;
          }
        } else {
          #pragma unroll 1
          while (0 < nElts) {
            if (t*EltPerPack < nElts) {
              int nIterPacks = min(nPacks, tn);
              Pack inp = loadPack<Pack>(input, t*EltPerPack, nElts);
              lla2a.bcast(/*slot=*/nIterPacks*rank + t, inp);
              AccPack out = lla2a.template recvReduce</*Unroll=*/8, Pack>(
                /*slotStart=*/t, /*slotCount=*/nRanks, /*slotStride=*/nIterPacks,
                /*eltToAcc=*/[&] __device__ (Pack x)->AccPack {
                  return applyCast<T, Acc>(x);
                },
                /*reduce=*/[&] __device__ (AccPack a, AccPack b)->AccPack {
                  return applyReduce(red, a, b);
                }
              );
              storePack(output, t*EltPerPack, nElts, applyCast<Acc, T>(out));
            }
            lla2a.endEpoch(cta);

            input += tn*EltPerPack;
            output += tn*EltPerPack;
            nElts -= tn*EltPerPack;
            nPacks -= tn;
          }
        }
      }
    );
}

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_AGxLL_R(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_AGxLL_R_impl<Red, T>(args, /*multimem=*/false);
}

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_AGxLLMC_R(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_AGxLL_R_impl<Red, T>(args, /*multimem=*/true);
}


/**
 * Lamport-style 1-shot AllReduce kernel
 * This kernel is similar to Lamport2Shot, but uses a single shot instead of two
 * It is used for small data sizes where the overhead of two shots is too high
 * Currently only supports fp32
 */
 template<int NCACHELINES, int EXTRATHREADS, typename Red, typename T,
 typename std::enable_if<(sizeof(T)!=4), int>::type = 0>
static __device__ __forceinline__ void allreduceLamport1ShotMultimemPerRank(
  ncclSymkArgsHandler const& handler, int nAllElts,
  Red red, ncclSymPtr<T> input, ncclSymPtr<T> output,
  ncclSymPtr<T> accumBuffer
) {}

/**
* Lamport-style 1-shot AllReduce kernel for fp32 with multimem broadcast
* This kernel is similar to Lamport2Shot, but uses a single shot instead of two
* It is used for small data sizes where the overhead of two shots is too high
*/
template<int NCACHELINES, int EXTRATHREADS, typename Red, typename T,
 typename std::enable_if<(sizeof(T)==4), int>::type = 0>
static __device__ __forceinline__ void allreduceLamport1ShotMultimemPerRank(
  ncclSymkArgsHandler const& handler, int nAllElts,
  Red red, ncclSymPtr<T> input, ncclSymPtr<T> output,
  ncclSymPtr<T> accumBuffer)
{
  ncclTeam world = ncclTeamWorld(handler.comm);
  int const& myrank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;
  auto const& multimem = handler.comm.lsaMultimem;

  int const& MAIN_THREADS = NCACHELINES * 8; // 992 threads for main work

  assert(gridDim.y == 1 && gridDim.z == 1);
  // const int totalLines = nAllElts / 4;
  int ctas = gridDim.x; // Total number of CTAs

  // int maxEltsPerIter = (MAIN_THREADS + EXTRATHREADS) * ctas * 4;
  int maxEltsPerIter = MAIN_THREADS * 4 * ctas;
  // int maxEltsPerIterPerRank = maxEltsPerIter;
  int maxEltsPerIterForAccumBuffer = (MAIN_THREADS + EXTRATHREADS) * ctas * 4;
  int maxAccumBuffIter = max((int) (REDUCTION_BUFFER_SIZE / sizeof(T) / maxEltsPerIterForAccumBuffer), 1); // Number of iterations that fit in accumBuffer
  // Calculates the number of iterations needed to process all lines, it should be the ceiling of the division of nAllElts by maxEltsPerIter
  int nIterations = (nAllElts + maxEltsPerIter - 1) / maxEltsPerIter;

  int line, smem_idx, mysmemline;
  // if (myrank == 0 && blockIdx.x == 0 && threadIdx.x == 0)
  //   printf("[Rank %d] blockIdx.y: %d nIterations: %d, maxEltsPerIter: %d, nAllElts: %d, maxAccumBuffIter: %d\n", blockIdx.y, myrank, nIterations, maxEltsPerIter, nAllElts, maxAccumBuffIter);

  ncclSymPtr<T> currInput = input;
  ncclSymPtr<T> currOutput = output;
  ncclSymPtr<T> currAccumBuffer = accumBuffer;

  __shared__ float movedData[128]; // Shared memory for displaced flag data
  bool flagcarrier = ((threadIdx.x & 7) == 0); // Every 8th thread carries flag

  float4 v; // Use float4 for vectorized operations
  float *elt_v = reinterpret_cast<float*>(&v);

  for (int i = 0; i < nIterations; i++)
  {

    // Computes the offset of the current iteration
    int offset = i * maxEltsPerIter;
    currInput = input + offset;
    currOutput = output + offset;
    // Cycles through the accumulation buffer for each iteration
    currAccumBuffer = accumBuffer + (i % maxAccumBuffIter) * maxEltsPerIterForAccumBuffer;

    // Number of lines to process in this iteration
    int lines = MAIN_THREADS * ctas;
    int last_cta_nthreads = MAIN_THREADS;
    int numCTAs = ctas;

    if (i == nIterations - 1)
    {
      // If this is the last iteration
      lines = (nAllElts - offset) / sizeof(T);
      // Calculates the number of CTAs needed for the last iteration
      numCTAs = (lines + MAIN_THREADS - 1) / MAIN_THREADS;
      if (blockIdx.x >= numCTAs)
      {
        // If this CTA is not part of the last iteration, return
        continue;
      }
      last_cta_nthreads = lines % MAIN_THREADS == 0 ? MAIN_THREADS : lines % MAIN_THREADS;
    }

    const int maxthread = blockIdx.x == numCTAs - 1 ? last_cta_nthreads : NCACHELINES * 8;
    bool threadActive = threadIdx.x < maxthread + EXTRATHREADS;
    unsigned int activeMask = __ballot_sync(0xffffffff, threadActive);

    if (threadIdx.x < maxthread)
    {
      // Threads processing actual user data
      line = threadIdx.x + NCACHELINES * 8 * blockIdx.x;

      // Read input data (each rank reads its own input for target rank's portion)
      T* inputPtr = ((ncclSymPtr<T>)currInput).peerPtr(world, myrank);
      // Load 4 consecutive floats as float4 for vectorized operations

      float4* inputPtr4 = (float4*)inputPtr;
      v = inputPtr4[line];
      if (flagcarrier)
      {
        // Save original data and replace with Lamport flag
        // Correct smem_idx calculation for FP32 accumulation (matches userbuffers.cu line 823, 945)
        smem_idx = 1 + (threadIdx.x >> 3) + ((threadIdx.x >> 3) / (NCACHELINES / sizeof(float)));
        movedData[smem_idx] = elt_v[0];
      }
    }
    else if (threadActive)
    {
      // Extra threads handle displaced flag data
      mysmemline = threadIdx.x - maxthread;
      line = lines + blockIdx.x * EXTRATHREADS + mysmemline;
    }

    __syncthreads();

    // Phase 1: Atomic accumulation to target rank's buffer
    // In userbuffers.cu: ATOMIC_ADD32x4UC(v, uc0_base + (target_rank * UC0REGSIZE + line))
    // Broadcast my own data to all ranks
    if (threadActive)
    {
      if (threadIdx.x >= maxthread)
      {
        // Extra threads load displaced data from shared memory
        v = ((float4*) movedData)[mysmemline];
      }

      if (flagcarrier)
        elt_v[0] = 1.0f;

      float4* accumPtr = (float4*)(((ncclSymPtr<T>)currAccumBuffer).multimemPtr(multimem));
      // Multimem atomic add
      NCCL_ATOMIC_ADD_V4F32_MC(v, &accumPtr[line]);
    }
  } // First loop to broadcast my own data to all ranks


  for (int i = 0; i < nIterations; ++i)
  {
    int offset = i * maxEltsPerIter;
    currInput = input + offset;
    currOutput = output + offset;
    currAccumBuffer = accumBuffer + (i % maxAccumBuffIter) * maxEltsPerIterForAccumBuffer;

    int lines = MAIN_THREADS * ctas;
    int last_cta_nthreads = MAIN_THREADS;
    int numCTAs = ctas;

    if (i == nIterations - 1)
    {
      lines = (nAllElts - offset) / sizeof(T);
      numCTAs = (lines + MAIN_THREADS - 1) / MAIN_THREADS;
      if (blockIdx.x >= numCTAs)
      {
        return;
      }
      last_cta_nthreads = lines % MAIN_THREADS == 0 ? MAIN_THREADS : lines % MAIN_THREADS;
    }

    const int maxthread = blockIdx.x == numCTAs - 1 ? last_cta_nthreads : NCACHELINES * 8;
    bool threadActive = threadIdx.x < maxthread + EXTRATHREADS;
    unsigned int activeMask = __ballot_sync(0xffffffff, threadActive);

    if (threadIdx.x < maxthread)
    {
      line = threadIdx.x + NCACHELINES * 8 * blockIdx.x;
      if (flagcarrier)
        smem_idx = 1 + (threadIdx.x >> 3) + ((threadIdx.x >> 3) / (NCACHELINES / sizeof(float)));
    }
    else if (threadActive)
    {
      mysmemline = threadIdx.x - maxthread;
      line = lines + blockIdx.x * EXTRATHREADS + mysmemline;
    }


    if (threadActive)
    {
      float refvalue = (float)nRanks; // Expected flag value when all ranks contribute

      float4* ptr = (float4*)(((ncclSymPtr<T>)currAccumBuffer).peerPtr(world, myrank));
      bool readAgain;
      do
      {
        readAgain = false;
        load128(&ptr[line], v);
        readAgain = flagcarrier && (elt_v[0] != refvalue);
      }
      while (__any_sync(activeMask, readAgain));

      store128_clear((uint4*) &ptr[line]);
    }

    // Store displaced data to the shared memory
    if (threadActive && threadIdx.x >= maxthread)
    {
      ((float4*)movedData)[mysmemline] = v;
    // Don't return here - need to participate in __syncthreads()
    }

    __syncthreads();
    if (threadIdx.x < maxthread)
    {
      if (flagcarrier)
        elt_v[0] = movedData[smem_idx]; // Restore the displaced data

        // Write final result to output using symmetric memory (vectorized write)
        // Copy result to my own output buffer
        float4* outputPtr = (float4*)((ncclSymPtr<T>)currOutput).peerPtr(world, myrank);
        store128(&outputPtr[line], v);
      }
  } // End of the second loop to compute the final result and store it in the output buffer
}


/**
 * Lamport-style 1-shot AllReduce kernel
 * This kernel is similar to Lamport2Shot, but uses a single shot instead of two
 * It is used for small data sizes where the overhead of two shots is too high
 * Currently only supports fp32
 */
template<int NCACHELINES, int EXTRATHREADS, typename Red, typename T,
         typename std::enable_if<(sizeof(T)!=4), int>::type = 0>
static __device__ __forceinline__ void allreduceLamport1ShotPerRank(
    ncclSymkArgsHandler const& handler, int nAllElts,
    Red red, ncclSymPtr<T> input, ncclSymPtr<T> output,
    ncclSymPtr<T> accumBuffer
  ) {}


/**
 * Lamport-style 1-shot AllReduce kernel for fp32
 * This kernel is similar to Lamport2Shot, but uses a single shot instead of two
 * It is used for small data sizes where the overhead of two shots is too high
 */
template<int NCACHELINES, int EXTRATHREADS, typename Red, typename T,
         typename std::enable_if<(sizeof(T)==4), int>::type = 0>
static __device__ __forceinline__ void allreduceLamport1ShotPerRank(
    ncclSymkArgsHandler const& handler, int nAllElts,
    Red red, ncclSymPtr<T> input, ncclSymPtr<T> output,
    ncclSymPtr<T> accumBuffer
  ) {

  ncclTeam world = ncclTeamWorld(handler.comm);
  int const& myrank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;

  int const& MAIN_THREADS = NCACHELINES * 8; // 992 threads for main work

  assert(gridDim.y == 1 && gridDim.z == 1);
  // const int totalLines = nAllElts / 4;
  int ctas = gridDim.x; // Total number of CTAs

  // int maxEltsPerIter = (MAIN_THREADS + EXTRATHREADS) * ctas * 4;
  int maxEltsPerIter = MAIN_THREADS * 4 * ctas;
  // int maxEltsPerIterPerRank = maxEltsPerIter;
  int maxEltsPerIterForAccumBuffer = (MAIN_THREADS + EXTRATHREADS) * ctas * 4;
  int maxAccumBuffIter = max((int) (REDUCTION_BUFFER_SIZE / sizeof(T) / maxEltsPerIterForAccumBuffer), 1); // Number of iterations that fit in accumBuffer
  // Calculates the number of iterations needed to process all lines, it should be the ceiling of the division of nAllElts by maxEltsPerIter
  int nIterations = (nAllElts + maxEltsPerIter - 1) / maxEltsPerIter;

  int line, smem_idx, mysmemline;
  // if (myrank == 0 && blockIdx.x == 0 && threadIdx.x == 0)
  //   printf("[Rank %d] blockIdx.y: %d nIterations: %d, maxEltsPerIter: %d, nAllElts: %d, maxAccumBuffIter: %d\n", blockIdx.y, myrank, nIterations, maxEltsPerIter, nAllElts, maxAccumBuffIter);

  int offset = 0;
  ncclSymPtr<T> currInput = input;
  ncclSymPtr<T> currOutput = output;
  ncclSymPtr<T> currAccumBuffer = accumBuffer;

  __shared__ float movedData[128]; // Shared memory for displaced flag data
  const bool flagcarrier = ((threadIdx.x & 7) == 0); // Every 8th thread carries flag
  __shared__ float4* accumPeerPtr[4];


  float4 v; // Use float4 for vectorized operations
  // float *elt_v = reinterpret_cast<float*>(&v);

  int lines = MAIN_THREADS * ctas;
  int last_cta_nthreads = MAIN_THREADS;
  int numCTAs = ctas;
  int maxthread;
  bool threadActive;

  for (int i = 0; i < nIterations; i++)
  {
    // Computes the offset of the current iteration
    offset = i * maxEltsPerIter;
    currInput = input + offset;
    currOutput = output + offset;
    // Cycles through the accumulation buffer for each iteration
    currAccumBuffer = accumBuffer + (i % maxAccumBuffIter) * maxEltsPerIterForAccumBuffer;

    // Number of lines to process in this iteration

    if (i == nIterations - 1)
    {
      // If this is the last iteration
      lines = (nAllElts - offset) / sizeof(T);
      // Calculates the number of CTAs needed for the last iteration
      numCTAs = (lines + MAIN_THREADS - 1) / MAIN_THREADS;
      if (blockIdx.x >= numCTAs)
      {
        // If this CTA is not part of the last iteration, return
        continue;
      }
      last_cta_nthreads = lines % MAIN_THREADS == 0 ? MAIN_THREADS : lines % MAIN_THREADS;
    }

    maxthread = blockIdx.x == numCTAs - 1 ? last_cta_nthreads : NCACHELINES * 8;
    threadActive = threadIdx.x < maxthread + EXTRATHREADS;

    if (threadIdx.x < maxthread)
    {
      // Threads processing actual user data
      line = threadIdx.x + NCACHELINES * 8 * blockIdx.x;

      // Read input data (each rank reads its own input for target rank's portion)
      T* inputPtr = ((ncclSymPtr<T>)currInput).peerPtr(world, myrank);
      // Load 4 consecutive floats as float4 for vectorized operations
      float4* inputPtr4 = (float4*)inputPtr;
      v = inputPtr4[line];
      if (flagcarrier)
      {
        // Save original data and replace with Lamport flag
        // Correct smem_idx calculation for FP32 accumulation (matches userbuffers.cu line 823, 945)
        smem_idx = 1 + (threadIdx.x >> 3) + ((threadIdx.x >> 3) / 31);
        // movedData[smem_idx] = elt_v[0];
        movedData[smem_idx] = v.x;
      }
    }
    else if (threadActive)
    { // Extra threads handle displaced flag data
      mysmemline = threadIdx.x - maxthread;
      line = lines + blockIdx.x * EXTRATHREADS + mysmemline;
    }

    __syncthreads();

    // Phase 1: Atomic accumulation to target rank's buffer
    // In userbuffers.cu: ATOMIC_ADD32x4UC(v, uc0_base + (target_rank * UC0REGSIZE + line))
    // Broadcast my own data to all ranks
    if (threadActive)
    {
      if (threadIdx.x >= maxthread)
      {
          // Extra threads load displaced data from shared memory
          v = ((float4*) movedData)[mysmemline];
      }

      if (flagcarrier) v.x = 1.0f;
        // elt_v[0] = 1.0f;


      #pragma unroll
      for (int j = 0; j < nRanks; j++)
      {
        float4* accumPtr = (float4*)(((ncclSymPtr<T>)currAccumBuffer).peerPtr(world, j));
        // float4* accumPtr = accumPeerPtr[j];
        NCCL_ATOMIC_ADD_V4F32(v, &accumPtr[line]);
      }
    }
  } // First loop to broadcast my own data to all ranks

  lines = MAIN_THREADS * ctas;
  last_cta_nthreads = MAIN_THREADS;
  numCTAs = ctas;

  for (int i = 0; i < nIterations; i++)
  {

    offset = i * maxEltsPerIter;
    currInput = input + offset;
    currOutput = output + offset;
    currAccumBuffer = accumBuffer + (i % maxAccumBuffIter) * maxEltsPerIterForAccumBuffer;

    if (i == nIterations - 1)
    {
      lines = (nAllElts - offset) / sizeof(T);
      numCTAs = (lines + MAIN_THREADS - 1) / MAIN_THREADS;
      if (blockIdx.x >= numCTAs)
      {
        return;
      }
      last_cta_nthreads = lines % MAIN_THREADS == 0 ? MAIN_THREADS : lines % MAIN_THREADS;
    }

    maxthread = blockIdx.x == numCTAs - 1 ? last_cta_nthreads : NCACHELINES * 8;
    threadActive = threadIdx.x < maxthread + EXTRATHREADS;
    unsigned int activeMask = __ballot_sync(0xffffffff, threadActive);

    if (threadIdx.x < maxthread)
    {
      line = threadIdx.x + NCACHELINES * 8 * blockIdx.x;
      if (flagcarrier)
        smem_idx = 1 + (threadIdx.x >> 3) + ((threadIdx.x >> 3) / 31);
    }
    else if (threadActive)
    {
      mysmemline = threadIdx.x - maxthread;
      line = lines + blockIdx.x * EXTRATHREADS + mysmemline;
    }


    if (threadActive)
    {
      float refvalue = (float)nRanks; // Expected flag value when all ranks contribute

      float4* ptr = (float4*)(((ncclSymPtr<T>)currAccumBuffer).peerPtr(world, myrank));
      bool readAgain;
      do
      {
        readAgain = false;
        load128(&ptr[line], v);
        readAgain = flagcarrier && (v.x != refvalue);
      }
      while (__any_sync(activeMask, readAgain));

      store128_clear((uint4*) &ptr[line]);

      // Store displaced data to the shared memory
      if (threadActive && threadIdx.x >= maxthread)
      {
        ((float4*)movedData)[mysmemline] = v;
      }
    }

    __syncthreads();
    if (threadIdx.x < maxthread)
    {
      if (flagcarrier) v.x = movedData[smem_idx];
        // elt_v[0] = movedData[smem_idx]; // Restore the displaced data

      // Write final result to output using symmetric memory (vectorized write)
      // Copy result to my own output buffer
      float4* outputPtr = (float4*)((ncclSymPtr<T>)currOutput).peerPtr(world, myrank);
      store128(&outputPtr[line], v);
    }
  } // End of second loop to compute the final result and store it in the output buffer
}


// Lamport-style 2-shot AllReduce kernel with multimem broadcast
// Grid configuration: dim3(ctas, nRanks) - each block targets a different rank
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_Lamport1ShotMC(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};

  Red<typename ncclSymkAccumType<Red, T, /*nvls=*/false>::Type> red(handler.devWork->redOpArg);

  struct ncclSymkDevWork const& dw = handler.devWork[0];

  size_t nAllElts = dw.nElts;
  ncclSymPtr<T> input(dw.inputWin, dw.inputOff);
  ncclSymPtr<T> output(dw.outputWin, dw.outputOff);


  if (sizeof(T) == 4) {
    constexpr int NCACHELINES = 124;  // Similar to LL2lines
    constexpr int EXTRATHREADS = 32;  // Similar to LL2extra

    // Get accumulation buffer from device communicator
    if (!((ncclSymkDevComm*)&handler.comm)->accumBuffer) {
      printf("ERROR: Lamport 1-shot MC accumulation buffer not allocated!\n");
      return;
    }

    // Create ncclSymPtr from the allocated accumulation buffer
    ncclSymPtr<T> accumBuffer;
    accumBuffer.offset = ((ncclSymkDevComm*)&handler.comm)->lamportAccumOffset;
    accumBuffer.window = ((ncclSymkDevComm*)&handler.comm)->accumBuffer;


    // Call the multimem per-rank kernel function with hardcoded 1024 threads
    allreduceLamport1ShotMultimemPerRank<NCACHELINES, EXTRATHREADS>(
      handler, nAllElts, red, input, output, accumBuffer
    );
  } else {
    // Fallback for non-fp32 types - use existing deterministic path
    printf("Lamport 1-shot currently only supports fp32 (sizeof(T)==4)\n");
    // Could call existing allreduceEnds here as fallback
  }
}


// Lamport-style 2-shot AllReduce kernel with multimem broadcast
// Grid configuration: dim3(ctas, nRanks) - each block targets a different rank
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_Lamport1Shot(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};

  Red<typename ncclSymkAccumType<Red, T, /*nvls=*/false>::Type> red(handler.devWork->redOpArg);

  // bar.arrive(ncclCoopCta(), cuda::memory_order_relaxed);

  struct ncclSymkDevWork const& dw = handler.devWork[0];

  size_t nAllElts = dw.nElts;
  ncclSymPtr<T> input(dw.inputWin, dw.inputOff);
  ncclSymPtr<T> output(dw.outputWin, dw.outputOff);


  if (sizeof(T) == 4) {
    constexpr int NCACHELINES = 62;  // Similar to LL2lines
    constexpr int EXTRATHREADS = 16;  // Similar to LL2extra

    int nRanks = handler.comm.nRanks;
    int rank = handler.comm.rank;

    // Get accumulation buffer from device communicator
    if (!((ncclSymkDevComm*)&handler.comm)->accumBuffer) {
      printf("ERROR: Lamport 1-shot accumulation buffer not allocated!\n");
      return;
    }

    // Create ncclSymPtr from the allocated accumulation buffer
    ncclSymPtr<T> accumBuffer;
    accumBuffer.offset = ((ncclSymkDevComm*)&handler.comm)->lamportAccumOffset;
    accumBuffer.window = ((ncclSymkDevComm*)&handler.comm)->accumBuffer;


    // Call the multimem per-rank kernel function with hardcoded 1024 threads
    allreduceLamport1ShotPerRank<NCACHELINES, EXTRATHREADS>(
      handler, nAllElts, red, input, output, accumBuffer
    );
  } else {
    // Fallback for non-fp32 types - use existing deterministic path
    printf("Lamport 1-shot currently only supports fp32 (sizeof(T)==4)\n");
    // Could call existing allreduceEnds here as fallback
  }
}


template <typename T>
struct is_supported_ar_dtype : std::false_type {};

template <> struct is_supported_ar_dtype<half> : std::true_type {};
#if defined(__CUDA_BF16_TYPES_EXIST__) || defined(__CUDACC__)
template <> struct is_supported_ar_dtype<__nv_bfloat16> : std::true_type {};
#endif
template <> struct is_supported_ar_dtype<float> : std::true_type {};

template<int NCACHELINES, int EXTRATHREADS, typename T, bool Multimem,
         typename std::enable_if<!is_supported_ar_dtype<T>::value, int>::type = 0>
static __device__ __forceinline__ void allreduceLamport2ShotPerRank(
    ncclSymkArgsHandler const& handler, int nAllElts,
    ncclSymPtr<T> input, ncclSymPtr<T> output,
    ncclSymPtr<T> accumBuffer
  ) {
    printf("ERROR: Unsupported data type for Lamport 2-shot AllReduce!\n");
}


// Lamport-style 2-shot allreduce for fp32 with distributed synchronization
// Follows userbuffers.cu pattern: each block targets a different rank
//
// Key Design: Grid configuration is dim3(ctas, nRanks)
// - blockIdx.x: data segment within a rank
// - blockIdx.y: target rank (which rank this block is contributing to)
// - Each GPU runs nRanks blocks, each targeting a different rank
//
// Algorithm Overview:
// 1. Each GPU runs blocks targeting all ranks (blockIdx.y = target_rank)
// 2. Each block reads target_rank's data portion and contributes to target_rank's accumulation buffer
// 3. Lamport flags (every 8th thread) provide distributed synchronization
// 4. Target rank waits for all contributions, then writes final results
//
// Memory Layout: Each rank has accumulation buffer in symmetric memory
// Similar to userbuffers.cu uc0_base regions
template<int NCACHELINES, int EXTRATHREADS, typename T, bool Multimem,
         typename std::enable_if<is_supported_ar_dtype<T>::value, int>::type = 0>
static __device__ __forceinline__ void allreduceLamport2ShotPerRank(
    ncclSymkArgsHandler const& handler, int nAllElts,
    ncclSymPtr<T> input, ncclSymPtr<T> output,
    ncclSymPtr<T> accumBuffer
  ) {
  ncclTeam world = ncclTeamWorld(handler.comm);
  int const& myrank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;
  int const& accumOffset = accumBuffer.offset;
  int const& slot = accumOffset / REDUCTION_BUFFER_SIZE;

  constexpr int MAIN_THREADS = NCACHELINES * 8; // 496 threads for main work
  using Pack = BytePack<16>;
  constexpr int EltsPerPack = 16 / sizeof(T);

  // CTAs per rank is equal to the X dimension of the grid, which is assigned in enqueue.cc
  int ctasPerRank = gridDim.x;
  constexpr int EltsPerThread = EltsPerPack;
  // Calculates the maximum number of elements that can be processed by a single iteration
  int maxEltsPerIter = ctasPerRank * nRanks * MAIN_THREADS * EltsPerThread;
  int maxEltsPerIterPerRank = (MAIN_THREADS + EXTRATHREADS) * ctasPerRank * EltsPerThread;
  // int maxEltsPerIterPerRank = maxEltsPerIter;
  // Calculate the total buffer space needed
  if (nAllElts * sizeof(T) / nRanks * (1 + sizeof(T) / CACHELINE_SIZE) > REDUCTION_BUFFER_SIZE) {
    if (myrank == 0 && blockIdx.x == 0 && threadIdx.x == 0)
      printf("ERROR: Not enough buffer space for the accumulation buffer in Lamport 2-shot AllReduce!\n");
    return;
  }
  int maxAccumBuffIter = max((int) (REDUCTION_BUFFER_SIZE / sizeof(T) / maxEltsPerIterPerRank), 1); // Number of iterations that fit in accumBuffer
  // Calculates the number of iterations needed to process all lines, it should be the ceiling of the division of nAllElts by maxEltsPerIter
  int nIterations = (nAllElts + maxEltsPerIter - 1) / maxEltsPerIter;
  const int target_rank = blockIdx.y;  // Each block targets a different rank

  // if (myrank == 0 && blockIdx.x == 0 && threadIdx.x == 0)
  //   printf("[Rank %d] blockIdx.y: %d nIterations: %d, maxEltsPerIter: %d, nAllElts: %d, maxAccumBuffIter: %d\n", blockIdx.y, myrank, nIterations, maxEltsPerIter, nAllElts, maxAccumBuffIter);

  ncclSymPtr<T> currInput = input;
  ncclSymPtr<T> currOutput = output;
  ncclSymPtr<T> currAccumBuffer = accumBuffer;

  // Loop 1: sending messages to the target rank
  int line_user, line, smem_idx, mysmemline;
  __shared__ T movedData[(MAIN_THREADS + EXTRATHREADS) / 8]; // Shared memory for displaced flag data
  bool flagcarrier = ((threadIdx.x & 7) == 0); // Every 8th thread carries flag

  ncclLLBuffer<ncclPoison, Multimem> outputBuf(
    output,
    /*bytesPerCtaPerEpoch=*/ 0,
    /*block=*/ 0,
    /*roundRobinFactor=*/ 0,
    /*mmHandle=*/ Multimem ? handler.comm.lsaMultimem : ncclMultimemHandle{}
  );


  // float4 v; // Use float4 for vectorized operations
  // float *elt_v = reinterpret_cast<float*>(&v);
  Pack v;

  const int defaultMaxConcurrentEpochs =
      (nIterations == 1 || REDUCTION_BUFFER_SIZE >= nAllElts * sizeof(T)) ? maxAccumBuffIter : maxAccumBuffIter >> 1;
  // const int configuredMaxConcurrentEpochs = (int)((ncclSymkDevComm*)&handler.comm)->maxConcurrentEpochs;
  const int configuredMaxConcurrentEpochs = 8;
  const int maxConcurrentEpochs =
      configuredMaxConcurrentEpochs ? configuredMaxConcurrentEpochs : defaultMaxConcurrentEpochs;
  assert(maxConcurrentEpochs >= 1);

  // if (myrank == 0 && blockIdx.x == 0 && threadIdx.x == 0)
  //   printf("[Rank %d] maxConcurrentEpochs: %d, nIterations: %d\n", myrank, maxConcurrentEpochs, nIterations);

  int j = 0;
  #pragma unroll 1
  // while (j < nIterations) {
  int currentIter = 0;
  for (int i = j; i < nIterations; ++i) {
    // if (currentIter >= maxConcurrentEpochs) break;
    // Computes the offset of the current iteration
    int offset = i * maxEltsPerIter;
    currInput = input + offset;
    currOutput = output + offset;
    currAccumBuffer = accumBuffer + (i % maxAccumBuffIter) * maxEltsPerIterPerRank;

    int perranklines = MAIN_THREADS * ctasPerRank;
    int last_cta_nthreads = MAIN_THREADS;
    int numCTAs = ctasPerRank;

    if (i == nIterations - 1)
    {
      // If this is the last iteration
      perranklines = (nAllElts - offset) / nRanks / EltsPerPack;
      // Calculates the number of CTAs needed for the last iteration
      numCTAs = (perranklines + MAIN_THREADS - 1) / MAIN_THREADS;
      if (blockIdx.x >= numCTAs)
      {
        // If this CTA is not part of the last iteration, return
        continue;
      }
      last_cta_nthreads = perranklines % MAIN_THREADS == 0 ? MAIN_THREADS : perranklines % MAIN_THREADS;
    }

    const int maxthread = blockIdx.x == numCTAs - 1 ? last_cta_nthreads : NCACHELINES * 8;
    bool threadActive = threadIdx.x < maxthread + EXTRATHREADS;
    // if (threadIdx.x == 0 && myrank == 0)
    //   printf("[Rank %d] i: %d, blockIdx.x: %d, blockIdx.y: %d, threadIdx.x: %d, maxthread: %d, perranklines: %d, last_cta_nthreads: %d\n", myrank, i, blockIdx.x, blockIdx.y, threadIdx.x, maxthread, perranklines, last_cta_nthreads);

    if (threadIdx.x < maxthread)
    {
      // Threads processing actual user data
      line = threadIdx.x + NCACHELINES * 8 * blockIdx.x;
      line_user = target_rank * perranklines + line; // Read target rank's data portion
      // Read input data (each rank reads its own input for target rank's portion)
      T* inputPtr = ((ncclSymPtr<T>)currInput).peerPtr(world, myrank);
      // Load 4 consecutive floats as float4 for vectorized operations
      // float4* inputPtr4 = (float4*)inputPtr;
      // v = inputPtr4[line_user];
      v= loadPack<Pack>((T*)inputPtr, line_user * EltsPerPack, nAllElts);
      if (flagcarrier)
      {
        // Save original data and replace with Lamport flag
        // Correct smem_idx calculation for FP32 accumulation (matches userbuffers.cu line 823, 945)
        // FIXME: Magic number 31 needs to be changed to a variable
        smem_idx = 1 + (threadIdx.x >> 3) + ((threadIdx.x >> 3) / 31);
        movedData[smem_idx] = packLane0<T>(v);
      }
    }
    else if (threadActive)
    { // Extra threads handle displaced flag data
      mysmemline = threadIdx.x - maxthread;
      line = perranklines + blockIdx.x * EXTRATHREADS + mysmemline;
    }

    __syncthreads();
    if (threadActive && threadIdx.x >= maxthread)
    {
      // Extra threads load displaced data from shared memory
      // v = ((float4*) movedData)[mysmemline];
      v = ((Pack*) movedData)[mysmemline];
    }

    if (threadIdx.x < maxthread)
    {
      // Poison the output buffer with NCCL_LAMPORT_INT
      // float4* outputPtr = (float4*)(((ncclSymPtr<T>)currOutput).peerPtr(world, myrank));
      // store128_poison(&outputPtr[line_user]);
      outputBuf.template reset<Pack>(line_user + offset / EltsPerPack);
    }

    // Phase 1: Atomic accumulation to target rank's buffer
    // In userbuffers.cu: ATOMIC_ADD32x4UC(v, uc0_base + (target_rank * UC0REGSIZE + line))
    if (threadActive)
    {
      if (flagcarrier)
        setPackLane0<T>(v, (T)1.0f);

      Pack* accumPtr = (Pack*)(((ncclSymPtr<T>)currAccumBuffer).peerPtr(world, target_rank));
      atomicAdd128<T>(&accumPtr[line], v);
    }
    currentIter++;
  } // End of Phase 1: sending messages to the target rank


  currentIter = 0;
  // Loop 2: receiving messages from the target rank and broadcasting the result to all ranks
  for (int i = j; i < nIterations; ++i) {
    // if (currentIter >= maxConcurrentEpochs) break;
    // Computes the offset of the current iteration
    int offset = i * maxEltsPerIter;
    currOutput = output + offset;
    currAccumBuffer = accumBuffer + (i % maxAccumBuffIter) * maxEltsPerIterPerRank;

    int perranklines = MAIN_THREADS * ctasPerRank;
    int last_cta_nthreads = MAIN_THREADS;
    int numCTAs = ctasPerRank;

    if (i == nIterations - 1)
    {
      // If this is the last iteration
      perranklines = (nAllElts - offset) / nRanks / EltsPerPack;
      // Calculates the number of CTAs needed for the last iteration
      numCTAs = (perranklines + MAIN_THREADS - 1) / MAIN_THREADS;
      // If this CTA is not part of the last iteration, return
      if (blockIdx.x >= numCTAs) return;

      last_cta_nthreads = perranklines % MAIN_THREADS == 0 ? MAIN_THREADS : perranklines % MAIN_THREADS;
    }

    const int maxthread = blockIdx.x == numCTAs - 1 ? last_cta_nthreads : NCACHELINES * 8;
    bool threadActive = threadIdx.x < maxthread + EXTRATHREADS;
    unsigned int activeMask = __ballot_sync(0xffffffff, threadActive);

    if (threadIdx.x < maxthread)
    {
      line = threadIdx.x + NCACHELINES * 8 * blockIdx.x;
      line_user = target_rank * perranklines + line; // Read target rank's data portion
      if (flagcarrier)
        smem_idx = 1 + (threadIdx.x >> 3) + ((threadIdx.x >> 3) / 31);
    }
    else if (threadActive)
    {
      mysmemline = threadIdx.x - maxthread;
      line = perranklines + blockIdx.x * EXTRATHREADS + mysmemline;
    }

    if (target_rank == myrank)
    {
      float refvalue = (float) nRanks; // Expected flag value when all ranks contribute
      if (threadActive)
      {
        Pack* ptr = (Pack*)(((ncclSymPtr<T>)currAccumBuffer).peerPtr(world, myrank));
        bool readAgain;
        do
        {
          readAgain = false;
          // Poll accumulator with a volatile 128-bit load so remote atomic updates
          // become visible; loadPack() is non-volatile and can miss progress here.
          v = ld_volatile_global<16>(cvta_to_global(&ptr[line]));
          readAgain = flagcarrier && (packLane0<T>(v) != (T)refvalue);
        }
        while (__any_sync(activeMask, readAgain));

        store128_clear((uint4*)&ptr[line]);
      }

      // Store displaced data to the shared memory
      if (threadActive && threadIdx.x >= maxthread)
      {
        // ((float4*)movedData)[mysmemline] = v;
        ((Pack*) movedData)[mysmemline] = v;
      }

      __syncthreads();
      if (threadIdx.x < maxthread)
      {
        if (flagcarrier)
          // elt_v[0] = movedData[smem_idx]; // Restore the displaced data
          setPackLane0<T>(v, movedData[smem_idx]);

        // Write final result to output using symmetric memory (vectorized write)
        // Broadcast reduced result to all ranks
        // #pragma unroll
        // for (int j = 0; j < nRanks; j++)
        // {
        //   float4* peerPtr = (float4*)(((ncclSymPtr<T>)currOutput).peerPtr(world, j));
        //   store128(&peerPtr[line_user], v);
        // }
        outputBuf.template bcast<4, Pack>(world, line_user + offset / EltsPerPack, v);
      }
      else
      {
        currentIter++;
        continue;
      }
    }

    if (threadIdx.x >= maxthread) {
      currentIter++;
      continue;
    }
    assert(threadIdx.x < maxthread);
    // if (i == nIterations - 1)
    //   cudaTriggerProgrammaticLaunchCompletion();
    // Poll for completion (all ranks)
    outputBuf.template recv<Pack, /*Reset=*/false>(line_user + offset / EltsPerPack);
    currentIter++;
  } // End of Loop 2: receiving messages from the target rank and broadcasting the result to all ranks
  // j += maxConcurrentEpochs;
  // }
}


// ============================================================================
// Root-based Lamport 2-shot AllReduce for small messages
// ============================================================================
//
// When message size is too small to partition across ranks (nAllElts < nRanks *
// EltsPerPack), this kernel gathers ALL data to a single root rank for
// reduction, then the root broadcasts the result back using Poison-based LL
// synchronization with Lamport flag displacement.
//
// Algorithm:
//   Phase 1: Every rank atomically adds its entire input to root's
//            accumulation buffer.  Flag carriers (every 8th thread) displace
//            their first element into shared memory and replace it with 1.0,
//            serving as a Lamport counter.  Extra threads handle the displaced
//            data packs.
//   Phase 2: Root polls its accumulation buffer until every flag carrier's
//            counter equals nRanks (meaning all ranks have contributed).
//            Root restores the displaced first elements, then broadcasts the
//            fully-reduced result to every rank via outputBuf.bcast().
//            Non-root ranks poll the output buffer with outputBuf.recv().
//
// Grid: Only blockIdx.y == 0 blocks participate.  The kernel is designed to
//       be called from the Lamport2Shot entry point when data is too small
//       for the normal per-rank partitioned path.

template<int NCACHELINES, int EXTRATHREADS, typename T, bool Multimem,
         typename std::enable_if<!is_supported_ar_dtype<T>::value, int>::type = 0>
static __device__ __forceinline__ void allreduceLamport2ShotRootPerRank(
    ncclSymkArgsHandler const& handler, int nAllElts,
    ncclSymPtr<T> input, ncclSymPtr<T> output,
    ncclSymPtr<T> accumBuffer
  ) {
    printf("ERROR: Unsupported data type for Lamport 2-shot Root AllReduce!\n");
}

template<int NCACHELINES, int EXTRATHREADS, typename T, bool Multimem,
         typename std::enable_if<is_supported_ar_dtype<T>::value, int>::type = 0>
static __device__ __forceinline__ void allreduceLamport2ShotRootPerRank(
    ncclSymkArgsHandler const& handler, int nAllElts,
    ncclSymPtr<T> input, ncclSymPtr<T> output,
    ncclSymPtr<T> accumBuffer
  ) {
  ncclTeam world = ncclTeamWorld(handler.comm);
  int const& myrank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;

  constexpr int MAIN_THREADS = NCACHELINES * 8;
  using Pack = BytePack<16>;
  constexpr int EltsPerPack = 16 / sizeof(T);
  constexpr int root = 0;

  int totalPacks = (nAllElts + EltsPerPack - 1) / EltsPerPack;
  int numCTAs = (totalPacks + MAIN_THREADS - 1) / MAIN_THREADS;
  if ((int)blockIdx.x >= numCTAs) return;

  int last_cta_nthreads = (totalPacks % MAIN_THREADS == 0)
                            ? MAIN_THREADS
                            : (totalPacks % MAIN_THREADS);
  const int maxthread = ((int)blockIdx.x == numCTAs - 1)
                          ? last_cta_nthreads : MAIN_THREADS;
  bool threadActive = threadIdx.x < maxthread + EXTRATHREADS;
  unsigned int activeMask = __ballot_sync(0xffffffff, threadActive);

  ncclLLBuffer<ncclPoison, Multimem> outputBuf(
    output,
    /*bytesPerCtaPerEpoch=*/ 0,
    /*block=*/ 0,
    /*roundRobinFactor=*/ 0,
    /*mmHandle=*/ Multimem ? handler.comm.lsaMultimem : ncclMultimemHandle{}
  );

  __shared__ T movedData[(MAIN_THREADS + EXTRATHREADS) / 8];
  bool flagcarrier = ((threadIdx.x & 7) == 0);

  Pack v;
  int line, smem_idx, mysmemline;

  // ===== Phase 1: All ranks atomically add ALL data to root's accum buffer =====
  if (threadIdx.x < maxthread) {
    line = threadIdx.x + MAIN_THREADS * blockIdx.x;
    T* inputPtr = input.peerPtr(world, myrank);
    v = loadPack<Pack>(inputPtr, line * EltsPerPack, nAllElts);
    if (flagcarrier) {
      smem_idx = 1 + (threadIdx.x >> 3) + ((threadIdx.x >> 3) / 31);
      movedData[smem_idx] = packLane0<T>(v);
    }
  } else if (threadActive) {
    mysmemline = threadIdx.x - maxthread;
    line = totalPacks + blockIdx.x * EXTRATHREADS + mysmemline;
  }

  __syncthreads();
  if (threadActive && threadIdx.x >= maxthread) {
    v = ((Pack*)movedData)[mysmemline];
  }

  if (threadIdx.x < maxthread) {
    outputBuf.template reset<Pack>(line);
  }

  if (threadActive) {
    if (flagcarrier)
      setPackLane0<T>(v, (T)1.0f);
    Pack* accumPtr = (Pack*)(accumBuffer.peerPtr(world, root));
    atomicAdd128<T>(&accumPtr[line], v);
  }

  // ===== Phase 2: Root polls + broadcasts, non-root polls output =====
  if (threadIdx.x < maxthread) {
    line = threadIdx.x + MAIN_THREADS * blockIdx.x;
    if (flagcarrier)
      smem_idx = 1 + (threadIdx.x >> 3) + ((threadIdx.x >> 3) / 31);
  } else if (threadActive) {
    mysmemline = threadIdx.x - maxthread;
    line = totalPacks + blockIdx.x * EXTRATHREADS + mysmemline;
  }

  if (myrank == root) {
    float refvalue = (float)nRanks;
    if (threadActive) {
      Pack* ptr = (Pack*)(accumBuffer.peerPtr(world, root));
      bool readAgain;
      do {
        readAgain = false;
        v = ld_volatile_global<16>(cvta_to_global(&ptr[line]));
        readAgain = flagcarrier && (packLane0<T>(v) != (T)refvalue);
      } while (__any_sync(activeMask, readAgain));

      store128_clear((uint4*)&ptr[line]);
    }

    if (threadActive && threadIdx.x >= maxthread) {
      ((Pack*)movedData)[mysmemline] = v;
    }

    __syncthreads();
    if (threadIdx.x < maxthread) {
      if (flagcarrier)
        setPackLane0<T>(v, movedData[smem_idx]);
      outputBuf.template bcast<4, Pack>(world, line, v);
    }
  } else {
    if (threadIdx.x < maxthread) {
      outputBuf.template recv<Pack, /*Reset=*/false>(line);
    }
  }
}


template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_Lamport2ShotRoot(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};

  struct ncclSymkDevWork const& dw = handler.devWork[0];

  size_t nAllElts = dw.nElts;
  ncclSymPtr<T> input(dw.inputWin, dw.inputOff);
  ncclSymPtr<T> output(dw.outputWin, dw.outputOff);

  constexpr int NCACHELINES = 62;
  constexpr int EXTRATHREADS = 16;

  if (!((ncclSymkDevComm*)&handler.comm)->lamport2ShotAccumBuffer) {
    printf("ERROR: Lamport 2-shot Root accumulation buffer not allocated!\n");
    return;
  }

  ncclSymPtr<T> accumBuffer;
  accumBuffer.offset = ((ncclSymkDevComm*)&handler.comm)->lamportAccumOffset;
  accumBuffer.window = ((ncclSymkDevComm*)&handler.comm)->lamport2ShotAccumBuffer;

  allreduceLamport2ShotRootPerRank<NCACHELINES, EXTRATHREADS, T, false>(
    handler, nAllElts, input, output, accumBuffer
  );
}

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_Lamport2ShotRootMC(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};

  struct ncclSymkDevWork const& dw = handler.devWork[0];

  size_t nAllElts = dw.nElts;
  ncclSymPtr<T> input(dw.inputWin, dw.inputOff);
  ncclSymPtr<T> output(dw.outputWin, dw.outputOff);

  constexpr int NCACHELINES = 62;
  constexpr int EXTRATHREADS = 16;

  if (!((ncclSymkDevComm*)&handler.comm)->lamport2ShotAccumBuffer) {
    printf("ERROR: Lamport 2-shot Root MC accumulation buffer not allocated!\n");
    return;
  }

  ncclSymPtr<T> accumBuffer;
  accumBuffer.offset = ((ncclSymkDevComm*)&handler.comm)->lamportAccumOffset;
  accumBuffer.window = ((ncclSymkDevComm*)&handler.comm)->lamport2ShotAccumBuffer;

  allreduceLamport2ShotRootPerRank<NCACHELINES, EXTRATHREADS, T, true>(
    handler, nAllElts, input, output, accumBuffer
  );
}


template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_Lamport2Shot(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};

  struct ncclSymkDevWork const& dw = handler.devWork[0];

  size_t nAllElts = dw.nElts;
  ncclSymPtr<T> input(dw.inputWin, dw.inputOff);
  ncclSymPtr<T> output(dw.outputWin, dw.outputOff);

  
  // Hardcode thread configuration for 1024 threads total
  constexpr int NCACHELINES = 62;  // Similar to LL2lines
  constexpr int EXTRATHREADS = 16;  // Similar to LL2extra

  int nRanks = handler.comm.nRanks;
  int rank = handler.comm.rank;

  if (!((ncclSymkDevComm*)&handler.comm)->lamport2ShotAccumBuffer) {
    printf("ERROR: Lamport 2-shot accumulation buffer not allocated!\n");
    return;
  }

  ncclSymPtr<T> accumBuffer;
  accumBuffer.offset = ((ncclSymkDevComm*)&handler.comm)->lamportAccumOffset;
  accumBuffer.window = ((ncclSymkDevComm*)&handler.comm)->lamport2ShotAccumBuffer;

  constexpr int EltsPerPack = 16 / sizeof(T);
  if (nAllElts < (size_t)nRanks * EltsPerPack) {
    if (blockIdx.y == 0) {
      allreduceLamport2ShotRootPerRank<NCACHELINES, EXTRATHREADS, T, false>(
        handler, nAllElts, input, output, accumBuffer
      );
    }
    return;
  }

  int perrankpacks = nAllElts / nRanks / EltsPerPack;
  int processedElts = perrankpacks * nRanks * EltsPerPack;

  allreduceLamport2ShotPerRank<NCACHELINES, EXTRATHREADS, T, false>(
    handler, processedElts, input, output, accumBuffer
  );

  int remainderElts = nAllElts - processedElts;
  if (remainderElts > 0 && blockIdx.y == 0) {
    allreduceLamport2ShotRootPerRank<NCACHELINES, EXTRATHREADS, T, false>(
      handler, remainderElts, input + processedElts, output + processedElts, accumBuffer
    );
  }
}


template<int NCACHELINES, int EXTRATHREADS, typename Red, typename T,
         typename std::enable_if<(sizeof(T)!=4), int>::type = 0>
static __device__ __forceinline__ void allreduceLamport2ShotMultimemPerRank(
    ncclSymkArgsHandler const& handler, int nAllElts,
    Red red, ncclSymPtr<T> input, ncclSymPtr<T> output,
    ncclSymPtr<T> accumBuffer
  ) {}


// Lamport-style 2-shot allreduce for fp32 with distributed synchronization
// Follows userbuffers.cu pattern: each block targets a different rank
//
// Key Design: Grid configuration is dim3(ctas, nRanks)
// - blockIdx.x: data segment within a rank
// - blockIdx.y: target rank (which rank this block is contributing to)
// - Each GPU runs nRanks blocks, each targeting a different rank
//
// Algorithm Overview:
// 1. Each GPU runs blocks targeting all ranks (blockIdx.y = target_rank)
// 2. Each block reads target_rank's data portion and contributes to target_rank's accumulation buffer
// 3. Lamport flags (every 8th thread) provide distributed synchronization
// 4. Target rank waits for all contributions, then writes final results
//
// Memory Layout: Each rank has accumulation buffer in symmetric memory
// Similar to userbuffers.cu uc0_base regions
template<int NCACHELINES, int EXTRATHREADS, typename Red, typename T,
         typename std::enable_if<(sizeof(T)==4), int>::type = 0>
static __device__ __forceinline__ void allreduceLamport2ShotMultimemPerRank(
    ncclSymkArgsHandler const& handler, int nAllElts,
    Red red, ncclSymPtr<T> input, ncclSymPtr<T> output,
    ncclSymPtr<T> accumBuffer
  ) {
  ncclTeam world = ncclTeamWorld(handler.comm);
  int const& myrank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;
  int const& accumOffset = accumBuffer.offset;
  int const& slot = accumOffset / REDUCTION_BUFFER_SIZE;
  auto const& multimem = handler.comm.lsaMultimem;


  int const& MAIN_THREADS = NCACHELINES * 8; // 992 threads for main work

  // CTAs per rank is equal to the X dimension of the grid, which is assigned in enqueue.cc
  int ctasPerRank = gridDim.x;
  // Calculates the maximum number of elements that can be processed by a single iteration
  int maxEltsPerIter = ctasPerRank * nRanks * MAIN_THREADS * 4;
  int maxEltsPerIterPerRank = (MAIN_THREADS + EXTRATHREADS) * ctasPerRank * 4;
  // int maxEltsPerIterPerRank = maxEltsPerIter;
  int maxAccumBuffIter = max((int) (REDUCTION_BUFFER_SIZE / sizeof(T) / maxEltsPerIterPerRank), 1); // Number of iterations that fit in accumBuffer
  // Calculates the number of iterations needed to process all lines, it should be the ceiling of the division of nAllElts by maxEltsPerIter
  int nIterations = (nAllElts + maxEltsPerIter - 1) / maxEltsPerIter;
  const int defaultMaxConcurrentEpochs =
      (nIterations == 1 || REDUCTION_BUFFER_SIZE >= nAllElts * sizeof(T)) ? maxAccumBuffIter : maxAccumBuffIter >> 1;
  const int configuredMaxConcurrentEpochs = (int)((ncclSymkDevComm*)&handler.comm)->maxConcurrentEpochs;
  const int maxConcurrentEpochs =
      configuredMaxConcurrentEpochs ? configuredMaxConcurrentEpochs : defaultMaxConcurrentEpochs;
  assert(maxConcurrentEpochs >= 1);


  const int target_rank = blockIdx.y;  // Each block targets a different rank

  // if (myrank == 0 && blockIdx.x == 0 && threadIdx.x == 0)
  //   printf("[Rank %d] blockIdx.y: %d nIterations: %d, maxEltsPerIter: %d, nAllElts: %d, maxAccumBuffIter: %d\n", blockIdx.y, myrank, nIterations, maxEltsPerIter, nAllElts, maxAccumBuffIter);

  ncclSymPtr<T> currInput = input;
  ncclSymPtr<T> currOutput = output;
  ncclSymPtr<T> currAccumBuffer = accumBuffer;

  // Loop 1: sending messages to the target rank
  int line_user, line, smem_idx, mysmemline;
  __shared__ float movedData[128]; // Shared memory for displaced flag data
  bool flagcarrier = ((threadIdx.x & 7) == 0); // Every 8th thread carries flag

  float4 v; // Use float4 for vectorized operations
  float *elt_v = reinterpret_cast<float*>(&v);

  int j = 0;
  #pragma unroll 1
  while (j < nIterations) {
    int currentIter = 0;
    for (int i = j; i < nIterations; ++i) {
      if (currentIter >= maxConcurrentEpochs) break;

    // Computes the offset of the current iteration
    int offset = i * maxEltsPerIter;
    currInput = input + offset;
    currOutput = output + offset;
    currAccumBuffer = accumBuffer + (i % maxAccumBuffIter) * maxEltsPerIterPerRank;

    int perranklines = MAIN_THREADS * ctasPerRank;
    int last_cta_nthreads = MAIN_THREADS;
    int numCTAs = ctasPerRank;

    if (i == nIterations - 1)
    {
      // If this is the last iteration
      perranklines = (nAllElts - offset) / nRanks / 4;
      // Calculates the number of CTAs needed for the last iteration
      numCTAs = (perranklines + MAIN_THREADS - 1) / MAIN_THREADS;
      if (blockIdx.x >= numCTAs)
      {
        // If this CTA is not part of the last iteration, return
        continue;
      }
      last_cta_nthreads = perranklines % MAIN_THREADS == 0 ? MAIN_THREADS : perranklines % MAIN_THREADS;
    }

    const int maxthread = blockIdx.x == numCTAs - 1 ? last_cta_nthreads : NCACHELINES * 8;
    bool threadActive = threadIdx.x < maxthread + EXTRATHREADS;
    // if (threadIdx.x == 0 && myrank == 0)
    //   printf("[Rank %d] i: %d, blockIdx.x: %d, blockIdx.y: %d, threadIdx.x: %d, maxthread: %d, perranklines: %d, last_cta_nthreads: %d\n", myrank, i, blockIdx.x, blockIdx.y, threadIdx.x, maxthread, perranklines, last_cta_nthreads);

    if (threadIdx.x < maxthread)
    {
      // Threads processing actual user data
      line = threadIdx.x + NCACHELINES * 8 * blockIdx.x;
      line_user = target_rank * perranklines + line; // Read target rank's data portion
      // Read input data (each rank reads its own input for target rank's portion)
      T* inputPtr = ((ncclSymPtr<T>)currInput).peerPtr(world, myrank);
      // Load 4 consecutive floats as float4 for vectorized operations
      float4* inputPtr4 = (float4*)inputPtr;
      v = inputPtr4[line_user];
      if (flagcarrier)
      {
        // Save original data and replace with Lamport flag
        // Correct smem_idx calculation for FP32 accumulation (matches userbuffers.cu line 823, 945)
        smem_idx = 1 + (threadIdx.x >> 3) + ((threadIdx.x >> 3) / (NCACHELINES / sizeof(float)));
        movedData[smem_idx] = elt_v[0];
      }
    }
    else if (threadActive)
    { // Extra threads handle displaced flag data
      mysmemline = threadIdx.x - maxthread;
      line = perranklines + blockIdx.x * EXTRATHREADS + mysmemline;
    }

    __syncthreads();
    if (threadActive && threadIdx.x >= maxthread)
    {
      // Extra threads load displaced data from shared memory
      v = ((float4*) movedData)[mysmemline];
    }

    if (threadIdx.x < maxthread)
    {
      // Poison the output buffer with NCCL_LAMPORT_INT
      float4* outputPtr = (float4*)(((ncclSymPtr<T>)currOutput).peerPtr(world, myrank));
      store128_poison(&outputPtr[line_user]);
    }

    // Phase 1: Atomic accumulation to target rank's buffer
    // In userbuffers.cu: ATOMIC_ADD32x4UC(v, uc0_base + (target_rank * UC0REGSIZE + line))
    if (threadActive)
    {
      if (flagcarrier)
        elt_v[0] = 1.0f;

      float4* accumPtr = (float4*)(((ncclSymPtr<T>)currAccumBuffer).peerPtr(world, target_rank));
      NCCL_ATOMIC_ADD_V4F32(v, &accumPtr[line]);
    }
      currentIter++;
    } // End of Phase 1: sending messages to the target rank


    // Loop 2: receiving messages from the target rank and broadcasting the result to all ranks
    currentIter = 0;
    for (int i = j; i < nIterations; ++i) {
      if (currentIter >= maxConcurrentEpochs) break;

    // Computes the offset of the current iteration
    int offset = i * maxEltsPerIter;
    currOutput = output + offset;
    currAccumBuffer = accumBuffer + (i % maxAccumBuffIter) * maxEltsPerIterPerRank;

    int perranklines = MAIN_THREADS * ctasPerRank;
    int last_cta_nthreads = MAIN_THREADS;
    int numCTAs = ctasPerRank;

    if (i == nIterations - 1)
    {
      // If this is the last iteration
      perranklines = (nAllElts - offset) / nRanks / 4;
      // Calculates the number of CTAs needed for the last iteration
      numCTAs = (perranklines + MAIN_THREADS - 1) / MAIN_THREADS;
      // If this CTA is not part of the last iteration, return
      if (blockIdx.x >= numCTAs) return;

      last_cta_nthreads = perranklines % MAIN_THREADS == 0 ? MAIN_THREADS : perranklines % MAIN_THREADS;
    }

    const int maxthread = blockIdx.x == numCTAs - 1 ? last_cta_nthreads : NCACHELINES * 8;
    bool threadActive = threadIdx.x < maxthread + EXTRATHREADS;
    unsigned int activeMask = __ballot_sync(0xffffffff, threadActive);

    if (threadIdx.x < maxthread)
    {
      line = threadIdx.x + NCACHELINES * 8 * blockIdx.x;
      line_user = target_rank * perranklines + line; // Read target rank's data portion
      if (flagcarrier)
        smem_idx = 1 + (threadIdx.x >> 3) + ((threadIdx.x >> 3) / (NCACHELINES / sizeof(float)));
    }
    else if (threadActive)
    {
      mysmemline = threadIdx.x - maxthread;
      line = perranklines + blockIdx.x * EXTRATHREADS + mysmemline;
    }

    if (target_rank == myrank)
    {
      float refvalue = (float) nRanks; // Expected flag value when all ranks contribute
      if (threadActive)
      {
        float4* ptr = (float4*)(((ncclSymPtr<T>)currAccumBuffer).peerPtr(world, myrank));
        bool readAgain;
        do
        {
          readAgain = false;
          load128(&ptr[line], v);
          readAgain = flagcarrier && (elt_v[0] != refvalue);
        }
        while (__any_sync(activeMask, readAgain));

        store128_clear((uint4*)&ptr[line]);
      }

      // Store displaced data to the shared memory
      if (threadActive && threadIdx.x >= maxthread)
      {
        ((float4*)movedData)[mysmemline] = v;
      }

      __syncthreads();
      if (threadIdx.x < maxthread)
      {
        if (flagcarrier)
          elt_v[0] = movedData[smem_idx]; // Restore the displaced data

        // Write final result to output using symmetric memory (vectorized write)
        // Broadcast reduced result to all ranks
        float4* outputPtr = (float4*)(((ncclSymPtr<T>)currOutput).multimemPtr(multimem));
        store128_mc(&outputPtr[line_user], v);
      }
      else
      {
        continue;
      }
    }

    if (threadIdx.x >= maxthread)
    {
      continue;
    }
    assert(threadIdx.x < maxthread);
    // if (i == nIterations - 1)
    //   cudaTriggerProgrammaticLaunchCompletion();
    // Poll for completion (all ranks)
    uint4* outputPtr = (uint4*)(((ncclSymPtr<T>)currOutput).peerPtr(world, myrank));
    while (true)
    {
      // Lamport POLL until the output buffer is not poisoned with NCCL_LAMPORT_INT
      uint4 result;
      load128_int(&outputPtr[line_user], result);
      if (result.x != (uint32_t) NCCL_LAMPORT_INT)
      {
        break;
      }
    }
      currentIter++;
    } // End of Loop 2: receiving messages from the target rank and broadcasting the result to all ranks
    j += maxConcurrentEpochs;
  }
}


template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_Lamport2ShotMC(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};

  Red<typename ncclSymkAccumType<Red, T, /*nvls=*/false>::Type> red(handler.devWork->redOpArg);

  struct ncclSymkDevWork const& dw = handler.devWork[0];

  size_t nAllElts = dw.nElts;
  ncclSymPtr<T> input(dw.inputWin, dw.inputOff);
  ncclSymPtr<T> output(dw.outputWin, dw.outputOff);

  // Hardcode thread configuration for 1024 threads total
  constexpr int NCACHELINES = 62;  // Similar to LL2lines
  constexpr int EXTRATHREADS = 16;  // Similar to LL2extra

  int nRanks = handler.comm.nRanks;
  int rank = handler.comm.rank;

  if (!((ncclSymkDevComm*)&handler.comm)->lamport2ShotAccumBuffer) {
    printf("ERROR: Lamport 2-shot MC accumulation buffer not allocated!\n");
    return;
  }

  ncclSymPtr<T> accumBuffer;
  accumBuffer.offset = ((ncclSymkDevComm*)&handler.comm)->lamportAccumOffset;
  accumBuffer.window = ((ncclSymkDevComm*)&handler.comm)->lamport2ShotAccumBuffer;

  constexpr int EltsPerPack = 16 / sizeof(T);
  if (nAllElts < (size_t)nRanks * EltsPerPack) {
    if (blockIdx.y == 0) {
      allreduceLamport2ShotRootPerRank<NCACHELINES, EXTRATHREADS, T, true>(
        handler, nAllElts, input, output, accumBuffer
      );
    }
    return;
  }

  int perrankpacks = nAllElts / nRanks / EltsPerPack;
  int processedElts = perrankpacks * nRanks * EltsPerPack;

  allreduceLamport2ShotPerRank<NCACHELINES, EXTRATHREADS, T, true>(
    handler, processedElts, input, output, accumBuffer
  );

  int remainderElts = nAllElts - processedElts;
  if (remainderElts > 0 && blockIdx.y == 0) {
    allreduceLamport2ShotRootPerRank<NCACHELINES, EXTRATHREADS, T, true>(
      handler, remainderElts, input + processedElts, output + processedElts, accumBuffer
    );
  }
}


/**
 * Lamport 1-Shot AllReduce kernel using ncclLLBuffer API.
 *
 * This kernel implements a 1-shot allreduce using the ncclLLBuffer abstraction:
 *   1. Each rank broadcasts its input data to all peers at slot [myRank * nIterElts + t]
 *   2. Each rank receives from slots [0*nIterElts + t, 1*nIterElts + t, ...] and reduces
 *   3. The reduced result is written to the output buffer
 *   4. Epoch is advanced for next iteration
 *
 * Key design points:
 *   - Supports ncclPoison and ncclLL sync modes (selected at compile time)
 *   - Leverages ncclLLBuffer's bcast() and recvReduce() primitives
 *   - Supports multiple buffering for overlapping communication
 *   - Uses 8-byte packs for efficient vectorized operations
 */
template<ncclLLSyncMode Mode, bool Multimem, int Unroll, template<typename> typename Red, typename T,
         int SubRanks, int SubLog>
__device__ __forceinline__ void ncclSymkRun_AllReduce_LL_impl(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};

  struct ncclSymkDevWork const& dw = handler.devWork[0];
  size_t nAllElts = dw.nElts;
  int const& rank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;
  // using Acc = typename ncclSymkAccumType<Red, T, /*nvls=*/Multimem>::Type;
  // Red<Acc> red(handler.devWork->redOpArg);
  Red<T> red(handler.devWork->redOpArg);

  // constexpr int BytesPerPack = 16;
  // using Pack = BytePack<BytesPerPack>;
  // using AccPack = BytePack<BytesPerPack*sizeof(Acc)/sizeof(T)>;
  // constexpr int EltPerPack = BytesPerPack/sizeof(T);

  // Get accumulation buffer from device communicator
  if (!((ncclSymkDevComm*)&handler.comm)->accumBuffer) {
    if (handler.comm.rank == 0 && blockIdx.x == 0 && threadIdx.x == 0) {
      printf("ERROR: AllReduce_LLBuffer accumulation buffer not allocated!\n");
    }
    return;
  }

  ncclTeam team = ncclTeamLsa(handler.comm);

  // Create ncclSymPtr from the allocated accumulation buffer
  ncclSymPtr<char> scratchSymPtr;
  scratchSymPtr.offset = ((ncclSymkDevComm*)&handler.comm)->lamportAccumOffset;
  scratchSymPtr.window = ((ncclSymkDevComm*)&handler.comm)->accumBuffer;

  ncclSymPtr<T> input(dw.inputWin, dw.inputOff);
  ncclSymPtr<T> output(dw.outputWin, dw.outputOff);
  T* inputPtr = (T*)input.localPtr();
  T* outputPtr = (T*)output.localPtr();
  // int tn = ncclSymkMaxThreads;

  constexpr int BytesPerPack = 8;
  using Pack = BytePack<BytesPerPack>;
  // using AccPack = BytePack<BytesPerPack*sizeof(Acc)/sizeof(T)>;
  constexpr int nEltsPerPack = BytesPerPack / sizeof(T);



  int numRanks = SubRanks;
  #if __cpp_if_constexpr
  if constexpr (SubRanks == 0) {
  #else
  if (SubRanks == 0) {
  #endif
    // If it is the default case, then we use all the ranks
    // and we assert that there is only one sub-warp
    numRanks = nRanks;
    assert(SubLog == 0);
  }

  assert(numRanks * (1 << SubLog) == nRanks);
  // Calculate bytesPerCTA for the LL buffer
  // This is the size of one buffer slot region per block
  size_t bytesPerCtaPerEpoch = nRanks * (blockDim.x >> SubLog) * BytesPerPack;

  int roundRobinFactor = REDUCTION_BUFFER_SIZE / (bytesPerCtaPerEpoch * gridDim.x);
  if (Mode == ncclLL)
    // Divide the multiple buffering factor by 2 for LL sync mode
    roundRobinFactor >>= 1;

  if (roundRobinFactor < 2) {
    printf("[ERROR]: roundRobinFactor < 2\n");
    return;
  }

  roundRobinFactor = min(roundRobinFactor, UINT8_MAX);
  // roundRobinFactor = 2;
  int nPacks = (nAllElts * sizeof(T) + BytesPerPack - 1) / BytesPerPack;
  // Create ncclLLBuffer for the intermediate reduction buffer
  // Mode can be ncclPoison or ncclLL
  // Multimem controls whether to use multicast for broadcast
  ncclLLBuffer<Mode, Multimem> llBuf(
    scratchSymPtr,
    /*bytesPerCtaPerEpoch=*/ bytesPerCtaPerEpoch,
    /*block=*/ blockIdx.x,
    /*roundRobinFactor=*/ (uint8_t)roundRobinFactor,
    /*mmHandle=*/ Multimem ? handler.comm.lsaMultimem : ncclMultimemHandle{}
  );

  const int warpId = threadIdx.x >> 5;
  // The thread ID within the sub-warp
  const int subWarpThreadId = threadIdx.x & (31 >> SubLog);
  // My thread index within the CTA
  const int myThreadIdx = warpId * (WARP_SIZE >> SubLog) + subWarpThreadId;
  const int subWarpId = (threadIdx.x & 31) >> (MAX_SUB_LOG - SubLog);
  int tid = myThreadIdx + blockIdx.x * (blockDim.x >> SubLog);
  int nthreads = (blockDim.x >> SubLog) * gridDim.x;

  int currentIter = 0;

  llBuf.setEpochValue(((ncclSymkDevComm*)&handler.comm)->llBufferEpoch);
  // Main loop with compile-time Unroll factor
  #pragma unroll 1
  for (int i = tid; i < nPacks; i += nthreads) {
    if (!subWarpId) {
      // Only the first sub-warp loads and broadcasts the data
      Pack myData = loadPack<Pack>((T*) inputPtr, i * nEltsPerPack, nAllElts);
      int slot = myThreadIdx + rank * (blockDim.x >> SubLog);
      llBuf.template bcast<Unroll, Pack>(team, slot, myData);
    }

    int eltStart = myThreadIdx + (subWarpId * numRanks) * (blockDim.x >> SubLog);
    Pack result = llBuf.template recvReduce<SubLog == 0 ? Unroll : SubRanks, Pack, /*Reset=*/Mode == ncclPoison>(
      /*eltStart=*/ eltStart,
      /*eltCount=*/ SubLog == 0 ? numRanks : SubRanks,
      /*eltStride=*/ (blockDim.x >> SubLog),
      /*eltToAcc=*/ [&] __device__ (Pack x) -> Pack { return x; },
      /*reduce=*/ [&] __device__ (Pack a, Pack b) -> Pack { return applyReduce(red, a, b); }
    );

    // Performs warp shuffle to sum the values from the participating lanes
    #if __cpp_if_constexpr
    if constexpr (SubLog > 0) {
    #else
    if (SubLog > 0) {
    #endif
      Pack otherResult = shflXorSync<Pack>(0xFFFFFFFF, result, 16);
      if (subWarpId >> (SubLog - 1) == 0) result = applyReduce(red, result, otherResult);
    }

    #if __cpp_if_constexpr
    if constexpr (SubLog > 1) {
    #else
    if (SubLog > 1) {
    #endif
      Pack otherResult = shflXorSync<Pack>(0xFFFFFFFF, result, 8);
      if (subWarpId >> (SubLog - 2) == 0) result = applyReduce(red, result, otherResult);
    }

    #if __cpp_if_constexpr
    if constexpr (SubLog > 2) {
    #else
    if (SubLog > 2) {
    #endif
      Pack otherResult = shflXorSync<Pack>(0xFFFFFFFF, result, 4);
      if (subWarpId >> (SubLog - 3) == 0) result = applyReduce(red, result, otherResult);
    }


    if (!subWarpId) {
      // Only the first sub-warp stores the result
      storePack<Pack>((T*) outputPtr, i * nEltsPerPack, nAllElts, result);
    }
    llBuf.advanceEpoch();
    currentIter++;
    if (currentIter % roundRobinFactor == 0 && Mode == ncclPoison) {
      __threadfence();
    }
  }
}

// Public entry points used by the symmetric-kernel generator.
// - AllReduce_LLBuffer: base version for non-power-of-2 ranks (Unroll=4)
// - AllReduce_LLBuffer_R{2,4,8,16,32,64}: rank-specialized versions
// - AllReduce_LLBuffer_LL16*: same but with LL sync mode

// Base versions (for non-power-of-2 ranks, uses Unroll=4)
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_LLBuffer(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_LL_impl<ncclPoison, /*Multimem=*/false, /*Unroll=*/4, Red, T, /*SubRanks=*/0, /*SubLog=*/0>(args);
}

// Rank-specialized versions (Poison mode)
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_LLBuffer_R4(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_LL_impl<ncclPoison, /*Multimem=*/false, /*Unroll=*/4, Red, T, /*SubRanks=*/4, /*SubLog=*/0>(args);
}
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_LLBuffer_R8(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_LL_impl<ncclPoison, /*Multimem=*/false, /*Unroll=*/8, Red, T, /*SubRanks=*/8, /*SubLog=*/0>(args);
}
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_LLBuffer_R16(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_LL_impl<ncclPoison, /*Multimem=*/false, /*Unroll=*/16, Red, T, /*SubRanks=*/8, /*SubLog=*/1>(args);
}
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_LLBuffer_R32(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_LL_impl<ncclPoison, /*Multimem=*/false, /*Unroll=*/32, Red, T, /*SubRanks=*/8, /*SubLog=*/2>(args);
}

// LL16 sync mode versions
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_LLBuffer_LL16(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_LL_impl<ncclLL, /*Multimem=*/false, /*Unroll=*/4, Red, T, /*SubRanks=*/0, /*SubLog=*/0>(args);
}
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_LLBuffer_LL16_R4(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_LL_impl<ncclLL, /*Multimem=*/false, /*Unroll=*/4, Red, T, /*SubRanks=*/4, /*SubLog=*/0>(args);
}
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_LLBuffer_LL16_R8(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_LL_impl<ncclLL, /*Multimem=*/false, /*Unroll=*/8, Red, T, /*SubRanks=*/8, /*SubLog=*/0>(args);
}
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_LLBuffer_LL16_R16(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_LL_impl<ncclLL, /*Multimem=*/false, /*Unroll=*/16, Red, T, /*SubRanks=*/8, /*SubLog=*/1>(args);
}
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_LLBuffer_LL16_R32(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_LL_impl<ncclLL, /*Multimem=*/false, /*Unroll=*/32, Red, T, /*SubRanks=*/8, /*SubLog=*/2>(args);
}


// Multimem version - uses multicast for broadcast
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_LLBufferMC(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_LL_impl<ncclPoison, /*Multimem=*/true, /*Unroll=*/4, Red, T, /*SubRanks=*/0, /*SubLog=*/0>(args);
}

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_LLBuffer_LL16MC(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_LL_impl<ncclLL, /*Multimem=*/true, /*Unroll=*/4, Red, T, /*SubRanks=*/0, /*SubLog=*/0>(args);
}


/**
 * Two-Shot AllReduce kernel using ncclLLBuffer API.
 *
 * This kernel implements a two-phase allreduce similar to twoShotAllReduceKernel
 * from customAllReduceKernels.cu:
 *
 *   Phase 1 (Reduce-Scatter):
 *     - Message is divided into nRanks portions
 *     - Each rank sends its contribution for each portion to the responsible rank
 *     - Responsible rank (r) receives and reduces contributions for portion r
 *
 *   Phase 2 (All-Gather):
 *     - Each rank broadcasts its reduced portion to all peers
 *     - Each rank receives reduced portions from all peers
 *
 * Memory layout:
 *   - Input/Output: nAllElts elements divided into nRanks portions of eltsPerRank each
 *   - Each rank is responsible for portion[rank], i.e., elements [rank*eltsPerRank, (rank+1)*eltsPerRank)
 *
 * Key design points:
 *   - Uses ncclPoison sync mode for low-latency synchronization
 *   - NO explicit barriers - relies entirely on ncclLLBuffer's send/recv for sync
 *   - Uses send() for point-to-point, bcast() for broadcast, recvReduce() for reduce
 *   - Unroll factor is a template parameter for rank specialization
 *
 * Buffer access pattern (matches customAllReduceKernels.cu twoShotAllReduceKernel):
 *   - Phase 1: Each rank sends input[i] to targetRank where targetRank = i / nPacksPerRank
 *   - Phase 2: Each rank broadcasts output[rank*nPacksPerRank : (rank+1)*nPacksPerRank]
 */
template<ncclLLSyncMode Mode, bool Multimem, int Unroll, template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_LLBuffer_Twoshot_impl(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};

  struct ncclSymkDevWork const& dw = handler.devWork[0];
  size_t nAllElts = dw.nElts;
  int const& rank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;
  // using Acc = typename ncclSymkAccumType<Red, T, /*nvls=*/false>::Type;
  // Red<Acc> red(handler.devWork->redOpArg);
  Red<T> red(handler.devWork->redOpArg);

  // Two-shot requires nAllElts divisible by nRanks
  size_t nEltsPerRank = nAllElts / nRanks;
  if (nEltsPerRank * nRanks != nAllElts) {
    // Fallback to one-shot for non-divisible sizes
    ncclSymkRun_AllReduce_LL_impl<Mode, /*Multimem=*/false, Unroll, Red, T, /*SubRanks=*/0, /*SubLog=*/0>(args);
    return;
  }

  // Get accumulation buffer from device communicator
  if (!((ncclSymkDevComm*)&handler.comm)->accumBuffer) {
    printf("ERROR: Two-shot accumulation buffer not allocated!\n");
    return;
  }

  ncclTeam team = ncclTeamLsa(handler.comm);

  ncclSymPtr<char> scratchSymPtr;
  scratchSymPtr.offset = ((ncclSymkDevComm*)&handler.comm)->lamportAccumOffset;
  scratchSymPtr.window = ((ncclSymkDevComm*)&handler.comm)->accumBuffer;

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

  ncclSymPtr<T> input(dw.inputWin, dw.inputOff);
  ncclSymPtr<char> output(dw.outputWin, dw.outputOff);
  T* inputPtr = (T*)input.localPtr();
  T* outputPtr = (T*)output.localPtr();

  constexpr int BytesPerPack = 16;
  using Pack = BytePack<BytesPerPack>;
  // using AccPack = BytePack<BytesPerPack*sizeof(Acc)/sizeof(T)>;
  constexpr int EltPerPack = BytesPerPack / sizeof(T);
  
  // Calculate pack counts
  int nPacksPerRank = (nEltsPerRank + EltPerPack - 1) / EltPerPack;
  int nTotalPacks = nPacksPerRank * nRanks;

  // LL buffer setup - need nRanks slots per thread per epoch
  size_t bytesPerCtaPerEpoch = nRanks * blockDim.x * BytesPerPack;
  // size_t bytesPerCtaPerEpoch = nAllElts * sizeof(T) / gridDim.x;
  int roundRobinFactor = REDUCTION_BUFFER_SIZE / (bytesPerCtaPerEpoch * gridDim.x);
  if (roundRobinFactor < 1) {
    printf("[ERROR]: roundRobinFactor < 1 in Two-Shot AllReduce\n");
    return;
  }

  // In the kernel, right after creating scratchSymPtr:
  // if (blockIdx.x == 0 && threadIdx.x == 0) {
  //   printf("[DEBUG KERNEL] Rank %d: &args->kcomm = %p, args->kcomm.lamportAccumOffset = %llu\n",
  //         rank, &args->kcomm, (unsigned long long)args->kcomm.lamportAccumOffset);
  // }
  // ncclLsaBarrierSession<ncclCoopCta> barrier{ncclCoopCta(), handler.comm, ncclTeamTagLsa(), blockIdx.x};

  roundRobinFactor = min(roundRobinFactor, (int)UINT8_MAX);

  ncclLLBuffer<Mode, /*Multimem=*/false> reductionBuf(
    scratchSymPtr,
    /*bytesPerCtaPerEpoch=*/ REDUCTION_BUFFER_SIZE,
    /*block=*/ 0,
    /*roundRobinFactor=*/ 0,
    /*mmHandle=*/ ncclMultimemHandle{}
  );

  ncclLLBuffer<ncclPoison, /*Multimem=*/Multimem> outputBuf(
    output,
    /*bytesPerCtaPerEpoch=*/ 0,
    /*block=*/ 0,
    /*roundRobinFactor=*/ 0,
    /*mmHandle=*/ Multimem ? handler.comm.lsaMultimem : ncclMultimemHandle{}
  );

  reductionBuf.setEpochValue(((ncclSymkDevComm*)&handler.comm)->llBufferEpoch);

  // if (gridDim.x % nRanks != 0) {
  if (gridDim.x % nRanks != 0) {

    // ===========================================================================
    // Phase 1: Reduce-Scatter
    // ===========================================================================
    // Each pack i belongs to targetRank = i / nPacksPerRank
    // All ranks send their input[i] to targetRank
    // targetRank receives and reduces contributions from all ranks
    #pragma unroll
    for (int i = tid; i < nTotalPacks; i += nthreads) {
      int targetRank = i / nPacksPerRank;
      int packInRank = i % nPacksPerRank;  // Output pack index within targetRank's output
      // int slotBase = packInRank % blockDim.x;

      // Load my input for this pack
      Pack myData = loadPack<Pack>((T*)inputPtr, i * EltPerPack, nAllElts);
      // Poison the output buffer
      outputBuf.template reset<Pack>(i);
      __threadfence();
      // Send my contribution to the responsible rank
      // int slot = rank * blockDim.x + slotBase;
      uint32_t slot = rank * nPacksPerRank + packInRank;
      reductionBuf.template send<Pack>(team, targetRank, slot, myData);
    }

    #pragma unroll
    for (int i = tid; i < nPacksPerRank; i += nthreads) {
      // Receive and reduce from all ranks
      Pack result = reductionBuf.template recvReduce<Unroll, Pack, /*Reset=*/Mode == ncclPoison>(
        /*eltStart=*/ i,
        /*eltCount=*/ nRanks,
        /*eltStride=*/ nPacksPerRank,
        /*eltToAcc=*/ [&] __device__ (Pack x) -> Pack { return x; },
        /*reduce=*/ [&] __device__ (Pack a, Pack b) -> Pack { return applyReduce(red, a, b); }
      );
      // ===========================================================================
      // Phase 2: All-Gather
      // ===========================================================================
      // Each rank broadcasts its reduced portion (nPacksPerRank packs) to all peers
      // All ranks receive from all ranks
      outputBuf.template bcast<Unroll, Pack>(team, rank * nPacksPerRank + i, result);
    }

    #pragma unroll
    for (int i = tid; i < nTotalPacks; i += nthreads) {
      outputBuf.template recv<Pack, /*Reset=*/false>(i);
    }
  } else {
    assert(gridDim.x % nRanks == 0);
    // If the same number of CTAs can be assigned to process data for each rank,
    // we can optimize the memory access pattern to be the same as that in Lamport2Shot
    int ctasPerRank = gridDim.x / nRanks;
    int targetRank = blockIdx.x / ctasPerRank;
    int blockId = blockIdx.x % ctasPerRank;
    const int nIters = (nTotalPacks + nthreads - 1) / nthreads;

    int packsPerRank = ctasPerRank * blockDim.x;
    int lastCtaThreads = blockDim.x;
    int numCTAs = ctasPerRank;

    roundRobinFactor = REDUCTION_BUFFER_SIZE / bytesPerCtaPerEpoch / ctasPerRank;
    const int defaultMaxConcurrentEpochs =
        (nIters == 1 || REDUCTION_BUFFER_SIZE >= nAllElts * sizeof(T)) ? roundRobinFactor : roundRobinFactor >> 1;
    const int configuredMaxConcurrentEpochs = (int)((ncclSymkDevComm*)&handler.comm)->maxConcurrentEpochs;
    const int maxConcurrentEpochs =
        configuredMaxConcurrentEpochs ? configuredMaxConcurrentEpochs : defaultMaxConcurrentEpochs;
    assert(maxConcurrentEpochs >= 1);

    int j = 0;
    #pragma unroll 1
    // Outer loop: process maxConcurrentEpochs epochs concurrently
    // while (j < nIters) {
    int currentIter = 0;
    for (int i = j; i < nIters; ++i) {
      if (i == nIters - 1) {
        packsPerRank = nPacksPerRank - i * packsPerRank;
        numCTAs = (packsPerRank + blockDim.x - 1) / blockDim.x;
        if (blockId >= numCTAs) break;
        lastCtaThreads = packsPerRank % blockDim.x == 0 ? blockDim.x : packsPerRank % blockDim.x;
      }
      const int maxThreads = blockId == numCTAs - 1 ? lastCtaThreads : blockDim.x;
      if (threadIdx.x >= maxThreads) break;
      // printf("[DEBUG KERNEL] Rank %d, blockIdx.x: %d, targetRank: %d, blockId: %d, threadIdx.x: %d, numCTAs: %d, packsPerRank: %d, maxThreads: %d\n", rank, blockIdx.x, targetRank, blockId, threadIdx.x, numCTAs, packsPerRank, maxThreads);
      int slot = i * ctasPerRank * blockDim.x + threadIdx.x + blockId * blockDim.x;
      int srcSlot = targetRank * nPacksPerRank + slot;
      Pack myData = loadPack<Pack>((T*)inputPtr, srcSlot * EltPerPack, nAllElts);
      outputBuf.template reset<Pack>(srcSlot);
      __threadfence();
      int targetSlot = rank * nPacksPerRank + slot;
      reductionBuf.template send<Pack>(team, targetRank, targetSlot, myData);
      currentIter++;
    }

    packsPerRank = ctasPerRank * blockDim.x;
    lastCtaThreads = blockDim.x;
    numCTAs = ctasPerRank;
    currentIter = 0;

    for (int i = j; i < nIters; ++i) {
      // Compute the number of packs that needs to be processed per CTA for this iteration
      if (i == nIters - 1) {
        // If this is the last iteration
        packsPerRank = nPacksPerRank - i * packsPerRank;
        numCTAs = (packsPerRank + blockDim.x - 1) / blockDim.x;
        // If this CTA is not part of the last iteration, return
        if (blockId >= numCTAs) break;
        lastCtaThreads = packsPerRank % blockDim.x == 0 ? blockDim.x : packsPerRank % blockDim.x;
      }

      const int maxThreads = blockId == numCTAs - 1 ? lastCtaThreads : blockDim.x;

      // If this thread is not part of the last iteration, return
      if (threadIdx.x >= maxThreads) break;
      // printf("[DEBUG KERNEL] Rank %d, blockIdx.x: %d, targetRank: %d, blockId: %d, threadIdx.x: %d, numCTAs: %d, packsPerRank: %d, maxThreads: %d\n", rank, blockIdx.x, targetRank, blockId, threadIdx.x, numCTAs, packsPerRank, maxThreads);
      int slot = i * ctasPerRank * blockDim.x + threadIdx.x + blockId * blockDim.x;
      if (targetRank == rank) {
        Pack result = reductionBuf.template recvReduce<Unroll, Pack, /*Reset=*/Mode == ncclPoison>(
          /*eltStart=*/ slot,
          /*eltCount=*/ nRanks,
          /*eltStride=*/ nPacksPerRank,
          /*eltToAcc=*/ [&] __device__ (Pack x) -> Pack { return x; },
          /*reduce=*/ [&] __device__ (Pack a, Pack b) -> Pack { return applyReduce(red, a, b); }
        );

        outputBuf.template bcast<Unroll, Pack>(team, rank * nPacksPerRank + slot, result);
      }
      outputBuf.template recv<Pack, /*Reset=*/false>(targetRank * nPacksPerRank + slot);
      currentIter++;
    }
    //   j += maxConcurrentEpochs;
    // }
  }
}

// Public entry point for Two-Shot AllReduce (base version for non-power-of-2 ranks)
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_LLBuffer_Twoshot(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_LLBuffer_Twoshot_impl</*Mode=*/ncclPoison, /*Multimem=*/false, /*Unroll=*/4, Red, T>(args);
}

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_LLBuffer_Twoshot_R8(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_LLBuffer_Twoshot_impl</*Mode=*/ncclPoison, /*Multimem=*/false, /*Unroll=*/8, Red, T>(args);
}

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_LLBuffer_Twoshot_LL16(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_LLBuffer_Twoshot_impl</*Mode=*/ncclLL, /*Multimem=*/false, /*Unroll=*/4, Red, T>(args);
}

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_LLBuffer_Twoshot_LL16_R8(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_LLBuffer_Twoshot_impl</*Mode=*/ncclLL, /*Multimem=*/false, /*Unroll=*/8, Red, T>(args);
}

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_LLBuffer_TwoshotMC(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_LLBuffer_Twoshot_impl</*Mode=*/ncclPoison, /*Multimem=*/true, /*Unroll=*/4, Red, T>(args);
}

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_LLBuffer_Twoshot_LL16MC(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_LLBuffer_Twoshot_impl</*Mode=*/ncclLL, /*Multimem=*/true, /*Unroll=*/4, Red, T>(args);
}
