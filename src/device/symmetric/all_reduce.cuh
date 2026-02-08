#include "sym_kernels.h"
#include "nccl_device.h"
#include "kernel.cuh"
#include "primitives.cuh"
#include "device.h"
#include <limits.h>

#define LINE_SIZE 16
#define MAIN_LANES_IN_WARP (WARP_SIZE - 1)
#define MAX_SUB_LOG 5
#define FLAG_CARRIER_MASK_IN_WARP (1u << 0) | (1u << 8) | (1u << 16) | (1u << 24)
#define TIMEOUT 2000000000ull


template<typename T>
__device__ __forceinline__ T deviceAdd(T a, T b) {
  return a + b;
}

template<>
__device__ __forceinline__ __nv_fp8_e4m3 deviceAdd<__nv_fp8_e4m3>(__nv_fp8_e4m3 a, __nv_fp8_e4m3 b) {
  return __nv_fp8_e4m3(__hadd(__half(a), __half(b)));
}

template<>
__device__ __forceinline__ __nv_fp8_e5m2 deviceAdd<__nv_fp8_e5m2>(__nv_fp8_e5m2 a, __nv_fp8_e5m2 b) {
  return __nv_fp8_e5m2(__hadd(__half(a), __half(b)));
}


#define NCCL_SYM_ATOMICS_EXPERIMENTAL
#include <type_traits>

// Forward declaration for ncclSymkRun_AllReduce_LL_impl
template<ncclLLSyncMode Mode, bool Multimem, int Unroll, template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_LL_impl(ncclSymkDevWorkArgs const* args);

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

#if defined(NCCL_SYM_ATOMICS_EXPERIMENTAL)
// Experimental atomic variants: trade determinism for latency by using L2 atomics.
// Notes:
// - Currently optimized for sizeof(T)==4 (e.g., float). Other types require additional handling.
// - Caller must ensure output is zero-initialized over the region being atomically accumulated.
// - waitNeeded barrier should be honored before first atomic wave to ensure zero-init visibility.
template<int UnrollPeers, typename Red, typename T,
         typename std::enable_if<(sizeof(T)==4), int>::type = 0>
static __device__ __forceinline__ void allreduceEndsAtomic(
    ncclSymkArgsHandler const& handler, int tn, int t, Red red,
    ncclLsaBarrierSession<ncclCoopCta>& bar,
    ncclSymPtr<T> input, ncclSymPtr<T> output,
    size_t nElts, uint32_t nPreElts, size_t nSufElts
  ) {
  (void)UnrollPeers; // unused in atomic path

  ncclTeam world = ncclTeamWorld(handler.comm);
  int const& rank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;


  using Acc = typename Red::EltType;

  // Two-shot path with phase barriers outside the per-element loops to avoid
  // mismatched barrier counts across CTAs when nElts < tn.
  // No zero-init to be safe in in-place cases; instead, skip leader's own add
  // when input and output alias on the leader.
  bool const leader = (rank == 0);
  bool const leaderInPlace = leader &&
    (((ncclSymPtr<T>)input).peerPtr(world, rank) == ((ncclSymPtr<T>)output).peerPtr(world, rank));

  // Phase 1: each rank atomically adds its local input into leader accumulator
  #pragma unroll 1
  for (size_t i = t; i < nElts; i += tn) {
    T local = ((ncclSymPtr<T>)input).peerPtr(world, rank)[i];
    if (!(leaderInPlace && leader)) {
      atomicAdd(&(((ncclSymPtr<T>)output).peerPtr(world, 0)[i]), local);
    }
  }
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  // Phase 2: leader broadcasts accumulated totals to all outputs
  if (rank == 0) {
    #pragma unroll 1
    for (size_t i = t; i < nElts; i += tn) {
      T total = ((ncclSymPtr<T>)output).peerPtr(world, 0)[i];
      #pragma unroll 1
      for (int r = 0; r < nRanks; r++) {
        ((ncclSymPtr<T>)output).peerPtr(world, r)[i] = total;
      }
    }
  }
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);
}

// No-op overload for non-32-bit T
template<int UnrollPeers, typename Red, typename T,
         typename std::enable_if<(sizeof(T)!=4), int>::type = 0>
static __device__ __forceinline__ void allreduceEndsAtomic(
    ncclSymkArgsHandler const&, int, int, Red,
    ncclLsaBarrierSession<ncclCoopCta>&,
    ncclSymPtr<T>, ncclSymPtr<T>, size_t, uint32_t, size_t
  ) {}
#endif // NCCL_SYM_ATOMICS_EXPERIMENTAL

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


inline __device__ void load128(const float4* ptr, float4 &val)
{
  asm volatile("ld.volatile.global.v4.f32 {%0, %1, %2, %3}, [%4];"
      : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w) : "l"(ptr));
}

inline __device__ void load128_int(const uint4* ptr, uint4 &val)
{
  asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
      : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w) : "l"(ptr));
}


inline __device__ void store128(float4* ptr, float4 val)
{
  asm volatile("st.global.cg.v4.f32 [%0], {%1, %2, %3, %4};"
      : : "l"(ptr), "f"(val.x), "f"(val.y), "f"(val.z), "f"(val.w) : "memory");
}


inline __device__ void store128_clear(uint4* ptr)
{
  asm volatile("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};"
      : : "l"(ptr), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
}


inline __device__ void store128_poison(float4* ptr)
{
  const uint32_t poison = NCCL_LAMPORT_INT;
  asm volatile("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};"
      : : "l"(ptr), "r"(poison), "r"(poison), "r"(poison), "r"(poison) : "memory");
}


inline __device__ void store128_mc(float4* ptr, float4 val)
{
  asm volatile("multimem.st.global.v4.f32 [%0], {%1, %2, %3, %4};"
      : : "l"(ptr), "f"(val.x), "f"(val.y), "f"(val.z), "f"(val.w) : "memory");
}


inline __device__ void load32(const float* ptr, float &val)
{
  asm volatile("ld.volatile.global.f32 %0, [%1];"
      : "=f"(val) : "l"(ptr) : "memory");
}

inline __device__ void load32_int(const uint32_t* ptr, uint32_t &val)
{
  asm volatile("ld.volatile.global.u32 %0, [%1];"
      : "=r"(val) : "l"(ptr) : "memory");
}


inline __device__ void store32(float* ptr, float val)
{
  asm volatile("st.volatile.global.f32 [%0], %1;"
      : : "l"(ptr), "f"(val) : "memory");
}

inline __device__ void store32_poison(uint32_t* ptr)
{
  const uint32_t poison = NCCL_LAMPORT_INT;
  asm volatile("st.volatile.global.u32 [%0], %1;"
      : : "l"(ptr), "r"(poison) : "memory");
}


inline __device__ void store32_mc(float* ptr, float val)
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
  int const& nranks = handler.comm.nRanks;
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
      float refvalue = (float)nranks; // Expected flag value when all ranks contribute

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
  int const& nranks = handler.comm.nRanks;

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
      for (int j = 0; j < nranks; j++)
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
      float refvalue = (float)nranks; // Expected flag value when all ranks contribute

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


template<int NCACHELINES, int EXTRATHREADS, typename Red, typename T,
         typename std::enable_if<(sizeof(T)!=4), int>::type = 0>
static __device__ __forceinline__ void allreduceLamport2ShotPerRank(
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
static __device__ __forceinline__ void allreduceLamport2ShotPerRank(
    ncclSymkArgsHandler const& handler, int nAllElts,
    Red red, ncclSymPtr<T> input, ncclSymPtr<T> output,
    ncclSymPtr<T> accumBuffer
  ) {
  ncclTeam world = ncclTeamWorld(handler.comm);
  int const& myrank = handler.comm.rank;
  int const& nranks = handler.comm.nRanks;
  int const& accumOffset = accumBuffer.offset;
  int const& slot = accumOffset / REDUCTION_BUFFER_SIZE;

  int const& MAIN_THREADS = NCACHELINES * 8; // 496 threads for main work

  // CTAs per rank is equal to the X dimension of the grid, which is assigned in enqueue.cc
  int ctasPerRank = gridDim.x;
  // Calculates the maximum number of elements that can be processed by a single iteration
  int maxEltsPerIter = ctasPerRank * nranks * MAIN_THREADS * 4;
  int maxEltsPerIterPerRank = (MAIN_THREADS + EXTRATHREADS) * ctasPerRank * 4;
  // int maxEltsPerIterPerRank = maxEltsPerIter;
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
  __shared__ float movedData[128]; // Shared memory for displaced flag data
  bool flagcarrier = ((threadIdx.x & 7) == 0); // Every 8th thread carries flag

  float4 v; // Use float4 for vectorized operations
  float *elt_v = reinterpret_cast<float*>(&v);

  for (int i = 0; i < nIterations; ++i)
  {

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
      perranklines = (nAllElts - offset) / nranks / 4;
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
        // FIXME: Magic number 31 needs to be changed to a variable
        smem_idx = 1 + (threadIdx.x >> 3) + ((threadIdx.x >> 3) / 31);
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
  } // End of Phase 1: sending messages to the target rank


  // Loop 2: receiving messages from the target rank and broadcasting the result to all ranks
  for (int i = 0; i < nIterations; ++i)
  {
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
      perranklines = (nAllElts - offset) / nranks / 4;
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
      float refvalue = (float) nranks; // Expected flag value when all ranks contribute
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
        #pragma unroll
        for (int j = 0; j < nranks; j++)
        {
          float4* peerPtr = (float4*)(((ncclSymPtr<T>)currOutput).peerPtr(world, j));
          store128(&peerPtr[line_user], v);
        }
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
  } // End of Loop 2: receiving messages from the target rank and broadcasting the result to all ranks
}


template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_Lamport2Shot(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};

  Red<typename ncclSymkAccumType<Red, T, /*nvls=*/false>::Type> red(handler.devWork->redOpArg);

  struct ncclSymkDevWork const& dw = handler.devWork[0];

  size_t nAllElts = dw.nElts;
  ncclSymPtr<T> input(dw.inputWin, dw.inputOff);
  ncclSymPtr<T> output(dw.outputWin, dw.outputOff);

  if (sizeof(T) == 4) {
    // Hardcode thread configuration for 1024 threads total
    constexpr int NCACHELINES = 62;  // Similar to LL2lines
    constexpr int EXTRATHREADS = 16;  // Similar to LL2extra

    int nRanks = handler.comm.nRanks;
    int rank = handler.comm.rank;

    // Get accumulation buffer from device communicator
    if (!((ncclSymkDevComm*)&handler.comm)->accumBuffer) {
      printf("ERROR: Lamport 2-shot accumulation buffer not allocated!\n");
      return;
    }

    // Create ncclSymPtr from the allocated accumulation buffer
    ncclSymPtr<T> accumBuffer;
    accumBuffer.offset = ((ncclSymkDevComm*)&handler.comm)->lamportAccumOffset;
    accumBuffer.window = ((ncclSymkDevComm*)&handler.comm)->accumBuffer;


    // Call the per-rank kernel function with hardcoded 1024 threads
    allreduceLamport2ShotPerRank<NCACHELINES, EXTRATHREADS>(
      handler, nAllElts, red, input, output, accumBuffer
    );
  } else {
    // Fallback for non-fp32 types - use existing deterministic path
    printf("Lamport 2-shot currently only supports fp32 (sizeof(T)==4)\n");
    // Could call existing allreduceEnds here as fallback
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
  int const& nranks = handler.comm.nRanks;
  int const& accumOffset = accumBuffer.offset;
  int const& slot = accumOffset / REDUCTION_BUFFER_SIZE;
  auto const& multimem = handler.comm.lsaMultimem;


  int const& MAIN_THREADS = NCACHELINES * 8; // 992 threads for main work

  // CTAs per rank is equal to the X dimension of the grid, which is assigned in enqueue.cc
  int ctasPerRank = gridDim.x;
  // Calculates the maximum number of elements that can be processed by a single iteration
  int maxEltsPerIter = ctasPerRank * nranks * MAIN_THREADS * 4;
  int maxEltsPerIterPerRank = (MAIN_THREADS + EXTRATHREADS) * ctasPerRank * 4;
  // int maxEltsPerIterPerRank = maxEltsPerIter;
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
  __shared__ float movedData[128]; // Shared memory for displaced flag data
  bool flagcarrier = ((threadIdx.x & 7) == 0); // Every 8th thread carries flag

  float4 v; // Use float4 for vectorized operations
  float *elt_v = reinterpret_cast<float*>(&v);

  for (int i = 0; i < nIterations; ++i)
  {

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
      perranklines = (nAllElts - offset) / nranks / 4;
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
  } // End of Phase 1: sending messages to the target rank


  // Loop 2: receiving messages from the target rank and broadcasting the result to all ranks
  for (int i = 0; i < nIterations; ++i)
  {
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
      perranklines = (nAllElts - offset) / nranks / 4;
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
      float refvalue = (float) nranks; // Expected flag value when all ranks contribute
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
  } // End of Loop 2: receiving messages from the target rank and broadcasting the result to all ranks
}


template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_Lamport2ShotMC(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};

  Red<typename ncclSymkAccumType<Red, T, /*nvls=*/false>::Type> red(handler.devWork->redOpArg);

  struct ncclSymkDevWork const& dw = handler.devWork[0];

  size_t nAllElts = dw.nElts;
  ncclSymPtr<T> input(dw.inputWin, dw.inputOff);
  ncclSymPtr<T> output(dw.outputWin, dw.outputOff);

  if (sizeof(T) == 4) {
    // Hardcode thread configuration for 1024 threads total
    constexpr int NCACHELINES = 124;  // Similar to LL2lines
    constexpr int EXTRATHREADS = 32;  // Similar to LL2extra

    int nRanks = handler.comm.nRanks;
    int rank = handler.comm.rank;

    // Get accumulation buffer from device communicator
    if (!((ncclSymkDevComm*)&handler.comm)->accumBuffer) {
      printf("ERROR: Lamport 2-shot MC accumulation buffer not allocated!\n");
      return;
    }

    // Create ncclSymPtr from the allocated accumulation buffer
    ncclSymPtr<T> accumBuffer;
    accumBuffer.offset = ((ncclSymkDevComm*)&handler.comm)->lamportAccumOffset;
    accumBuffer.window = ((ncclSymkDevComm*)&handler.comm)->accumBuffer;


    // if (rank == 0 && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
    //   printf("[Rank %d] Lamport 2-shot kernel started AccumBuffer Offset: %llu\n", rank, accumBuffer.offset);

    // Call the per-rank kernel function with hardcoded 1024 threads
    allreduceLamport2ShotMultimemPerRank<NCACHELINES, EXTRATHREADS>(
      handler, nAllElts, red, input, output, accumBuffer
    );
  } else {
    // Fallback for non-fp32 types - use existing deterministic path
    printf("Lamport 2-shot currently only supports fp32 (sizeof(T)==4)\n");
    // Could call existing allreduceEnds here as fallback
  }
}


/**
 * Lamport-style 1-shot AllReduce kernel
 * This kernel is similar to Lamport2Shot, but uses a single shot instead of two
 * It is used for small data sizes where the overhead of two shots is too high
 * Currently only supports fp32
 */
template<int NCACHELINES, int EXTRATHREADS, typename Red, typename T,
         typename std::enable_if<(sizeof(T)!=4), int>::type = 0>
static __device__ __forceinline__ void allreduceLamport1ShotPerRankV2(
    ncclSymkArgsHandler const& handler, int nAllElts,
    Red red, ncclSymPtr<T> input, ncclSymPtr<T> output,
    ncclSymPtr<T> accumBuffer
  ) {}

/**
 * Lane 31 collects v.x from lanes (0, 8, 16, 24). Others keep their original value.
 */
__device__ __forceinline__ float4 gather_x(const int& lane, float4 v)
{
  // FIXME Should replace all the magic numbers
  // Participating lanes are 0, 8, 16, 24, 31
  const unsigned mask = FLAG_CARRIER_MASK_IN_WARP | (1u << 31);
  if (mask & (1u << lane))
  {
    // Only the participating lanes can enter
    float x0 = __shfl_sync(mask, v.x, 0, WARP_SIZE);
    float x8 = __shfl_sync(mask, v.x, 8, WARP_SIZE);
    float x16 = __shfl_sync(mask, v.x, 16, WARP_SIZE);
    float x24 = __shfl_sync(mask, v.x, 24, WARP_SIZE);
    if (lane == 31)
    {
      return make_float4(x0, x8, x16, x24);
    }
  }
  // All non-participating lanes return their original value
  return v;
}


/**
 * Lane 31 scatters its v values to lanes (0, 8, 16, 24). Others keep their original value.
 */
__device__ __forceinline__ float4 scatter_x(const int& lane, float4 v) {
  // Participants: 0, 8, 16, 24, 31
  const unsigned mask = FLAG_CARRIER_MASK_IN_WARP | (1u << 31);

  // Broadcast each component of lane 31's vector to participants
  float b0  = __shfl_sync(mask, v.x, 31, WARP_SIZE);  // lane31.x -> everyone
  float b8  = __shfl_sync(mask, v.y, 31, WARP_SIZE);  // lane31.y -> everyone
  float b16 = __shfl_sync(mask, v.z, 31, WARP_SIZE);  // lane31.z -> everyone
  float b24 = __shfl_sync(mask, v.w, 31, WARP_SIZE);  // lane31.w -> everyone

  // Only the four destination lanes update their v.x
  if      (lane == 0)  v.x = b0;
  else if (lane == 8)  v.x = b8;
  else if (lane == 16) v.x = b16;
  else if (lane == 24) v.x = b24;

  return v;
}



/**
 * Lamport-style 1-shot AllReduce kernel for fp32
 * This kernel is similar to Lamport2Shot, but uses a single shot instead of two
 * It is used for small data sizes where the overhead of two shots is too high
 */
template<int NCACHELINES, int EXTRATHREADS, typename Red, typename T,
         typename std::enable_if<(sizeof(T)==4), int>::type = 0>
static __device__ __forceinline__ void allreduceLamport1ShotPerRankV2(
    ncclSymkArgsHandler const& handler, int nAllElts,
    Red red, ncclSymPtr<T> input, ncclSymPtr<T> output,
    ncclSymPtr<T> accumBuffer
  ) {

  ncclTeam world = ncclTeamWorld(handler.comm);
  int const& myrank = handler.comm.rank;
  int const& nranks = handler.comm.nRanks;

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

  int userLine, accumLine;
  // if (myrank == 0 && blockIdx.x == 0 && threadIdx.x == 0)
  //   printf("[Rank %d] blockIdx.y: %d nIterations: %d, maxEltsPerIter: %d, nAllElts: %d, maxAccumBuffIter: %d\n", blockIdx.y, myrank, nIterations, maxEltsPerIter, nAllElts, maxAccumBuffIter);

  int offset = 0;
  ncclSymPtr<T> currInput = input;
  ncclSymPtr<T> currOutput = output;
  ncclSymPtr<T> currAccumBuffer = accumBuffer;

  const bool& flagCarrier = ((threadIdx.x & 7) == 0); // Every 8th thread carries flag
  // Every 32nd thread carries displaced data
  const int& displacedDataCarrier = ((threadIdx.x + 1) & 31) == 0;
  const int& warpId = threadIdx.x >> 5;
  const int& laneId = threadIdx.x & 31;
  // float *elt_v = reinterpret_cast<float*>(&v);

  int lines = MAIN_THREADS * ctas;
  int last_cta_nthreads = MAIN_THREADS;
  int numActiveCTAs = ctas;
  int numActiveWarps = (MAIN_THREADS + EXTRATHREADS) / WARP_SIZE;
  int numActiveLanes = MAIN_LANES_IN_WARP;
  int maxWarps;
  // Number of active lanes in the warp
  int maxLanes = MAIN_LANES_IN_WARP;
  bool laneActive;

  // Use float4 for vectorized operations, default value is 0.0f
  float4 v;

  int accumBufferOffset = 0;

  for (int i = 0; i < nIterations; i++)
  {
    // Computes the offset of the address of the buffers in the current iteration
    offset = i * maxEltsPerIter;
    currInput = input + offset;
    currOutput = output + offset;
    // Cycles through the accumulation buffer for each iteration
    currAccumBuffer = accumBuffer + (i % maxAccumBuffIter) * maxEltsPerIterForAccumBuffer;


    // Number of lines to process in this iteration

    if (i == nIterations - 1)
    {
      // If this is the last iteration
      lines = (nAllElts - offset) / (LINE_SIZE / sizeof(T));
      // Calculates the number of CTAs needed for the last iteration
      numActiveCTAs = (lines + MAIN_THREADS - 1) / MAIN_THREADS;
      // If this CTA is not part of the last iteration, break out of the loop and start the next phase
      if (blockIdx.x >= numActiveCTAs) break;
      last_cta_nthreads = lines % MAIN_THREADS == 0 ? MAIN_THREADS : lines % MAIN_THREADS;
      // compute the number of warps needed for the last iteration in the last CTA
      numActiveWarps = (last_cta_nthreads + MAIN_LANES_IN_WARP - 1) / MAIN_LANES_IN_WARP;
      // compute the number of active lanes in the warp
      numActiveLanes = last_cta_nthreads % MAIN_LANES_IN_WARP == 0 ? MAIN_LANES_IN_WARP : last_cta_nthreads % MAIN_LANES_IN_WARP;
    }

    maxWarps = blockIdx.x == numActiveCTAs - 1 ? numActiveWarps : (MAIN_THREADS + EXTRATHREADS) / WARP_SIZE;
    // If warp is not active, break out of the loop and start the next phase
    if (warpId >= maxWarps) break;

    maxLanes = (blockIdx.x == numActiveCTAs - 1 && warpId == maxWarps - 1) ? numActiveLanes : MAIN_LANES_IN_WARP;
    laneActive = laneId < maxLanes;

    // All threads in the non-active warp should have already exited the loop
    v = {0.0f, 0.0f, 0.0f, 0.0f};
    accumLine = laneId + WARP_SIZE * warpId + (MAIN_THREADS + EXTRATHREADS) * blockIdx.x;
    if (laneActive && !displacedDataCarrier)
    {
      // The line of data to process from the input buffer
      userLine = laneId + MAIN_LANES_IN_WARP * warpId + MAIN_THREADS * blockIdx.x;
      // Load the data from the input buffer
      v = ((float4*)currInput.peerPtr(world, myrank))[userLine];
    }
    // Performs warp shuffle to for the displaced data carreir thread
    // to get the data from the participating lanes
    v = gather_x(laneId, v);


    // Sets the value of the first element to 1.0f for the flag carrier thread
    if (flagCarrier) v.x = 1.0f;

    // Performs atomic accumulation to the target rank's buffer
    #pragma unroll 1
    for (int j = 0; j < nranks; ++j)
    {
      int jj = (j + warpId) % nranks;
      NCCL_ATOMIC_ADD_V4F32(v, &((float4*)((ncclSymPtr<T>)currAccumBuffer).peerPtr(world, jj))[accumLine]);
    }
  } // First loop to broadcast my own data to all ranks

  // Resets the variables for the second loop
  lines = MAIN_THREADS * ctas;
  last_cta_nthreads = MAIN_THREADS;
  numActiveCTAs = ctas;
  numActiveWarps = (MAIN_THREADS + EXTRATHREADS) / WARP_SIZE;
  numActiveLanes = MAIN_LANES_IN_WARP;
  maxLanes = MAIN_LANES_IN_WARP;

  for (int i = 0; i < nIterations; i++)
  {

    offset = i * maxEltsPerIter;
    currInput = input + offset;
    currOutput = output + offset;
    currAccumBuffer = accumBuffer + (i % maxAccumBuffIter) * maxEltsPerIterForAccumBuffer;

    if (i == nIterations - 1)
    {
      lines = (nAllElts - offset) / (LINE_SIZE / sizeof(T));
      numActiveCTAs = (lines + MAIN_THREADS - 1) / MAIN_THREADS;
      // If this CTA is not part of the last iteration, just return
      if (blockIdx.x >= numActiveCTAs) return;
      last_cta_nthreads = lines % MAIN_THREADS == 0 ? MAIN_THREADS : lines % MAIN_THREADS;
      numActiveWarps = (last_cta_nthreads + MAIN_LANES_IN_WARP - 1) / MAIN_LANES_IN_WARP;
      numActiveLanes = last_cta_nthreads % MAIN_LANES_IN_WARP == 0 ? MAIN_LANES_IN_WARP : last_cta_nthreads % MAIN_LANES_IN_WARP;
    }

    maxWarps = blockIdx.x == numActiveCTAs - 1 ? numActiveWarps : (MAIN_THREADS + EXTRATHREADS) / WARP_SIZE;
    if (warpId >= maxWarps) return;
    maxLanes = (blockIdx.x == numActiveCTAs - 1 && warpId == maxWarps - 1) ? numActiveLanes : MAIN_LANES_IN_WARP;
    laneActive = laneId < maxLanes;


    accumLine = laneId + WARP_SIZE * warpId + (MAIN_THREADS + EXTRATHREADS) * blockIdx.x;
    if (laneActive && !displacedDataCarrier)
    {
      userLine = laneId + MAIN_LANES_IN_WARP * warpId + MAIN_THREADS * blockIdx.x;
    }

    float4* ptr = (float4*)(((ncclSymPtr<T>)currAccumBuffer).peerPtr(world, myrank));
    // int flagValue = 0.f;
    float refvalue = (float) nranks; // Expected flag value when all ranks contribute
    bool readAgain;
    do
    {
      readAgain = false;
      load128(&ptr[accumLine], v);
      readAgain = flagCarrier && (v.x != refvalue);
    }
    while (__any_sync(0xffffffff, readAgain));

    load128(&ptr[accumLine], v);


    // printf("[Rank %d] warp: %d lane: %d, maxLanes: %d, laneActive: %d, userLine: %d, accumLine: %d, v: %f, %f, %f, %f\n", myrank, warpId, laneId, maxLanes, laneActive, userLine, accumLine, v.x, v.y, v.z, v.w);
    // Uses warp shuffle to scatter the value of v to the participating lanes
    v = scatter_x(laneId, v);

    store128_clear((uint4*) &ptr[accumLine]);

    // Store the data to the output buffer
    if (laneActive)
    {
      store128(&((float4*)((ncclSymPtr<T>)currOutput).peerPtr(world, myrank))[userLine], v);
    }
  } // End of second loop to compute the final result and store it in the output buffer
}


// Lamport-style 2-shot AllReduce kernel with multimem broadcast
// Grid configuration: dim3(ctas, nRanks) - each block targets a different rank
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_Lamport1ShotV2(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};

  Red<typename ncclSymkAccumType<Red, T, /*nvls=*/false>::Type> red(handler.devWork->redOpArg);

  // bar.arrive(ncclCoopCta(), cuda::memory_order_relaxed);

  struct ncclSymkDevWork const& dw = handler.devWork[0];

  size_t nAllElts = dw.nElts;
  ncclSymPtr<T> input(dw.inputWin, dw.inputOff);
  ncclSymPtr<T> output(dw.outputWin, dw.outputOff);


  if (sizeof(T) == 4) {
    constexpr int NCACHELINES = 124;  // Similar to LL2lines
    constexpr int EXTRATHREADS = 32;  // Similar to LL2extra

    // Get accumulation buffer from device communicator
    if (!((ncclSymkDevComm*)&handler.comm)->accumBuffer) {
      printf("ERROR: Lamport 1-shot V2 accumulation buffer not allocated!\n");
      return;
    }

    // Create ncclSymPtr from the allocated accumulation buffer
    ncclSymPtr<T> accumBuffer;
    accumBuffer.offset = ((ncclSymkDevComm*)&handler.comm)->lamportAccumOffset;
    accumBuffer.window = ((ncclSymkDevComm*)&handler.comm)->accumBuffer;


    // Call the multimem per-rank kernel function with hardcoded 1024 threads
    allreduceLamport1ShotPerRankV2<NCACHELINES, EXTRATHREADS>(
      handler, nAllElts, red, input, output, accumBuffer
    );
  } else {
    // Fallback for non-fp32 types - use existing deterministic path
    printf("Lamport 1-shot currently only supports fp32 (sizeof(T)==4)\n");
    // Could call existing allreduceEnds here as fallback
  }
}


/**
 * Lamport-style 1-shot AllReduce kernel
 * This kernel is similar to Lamport2Shot, but uses a single shot instead of two
 * It is used for small data sizes where the overhead of two shots is too high
 * Currently only supports fp32
 */
template<int SUBRANKS, int SUB_LOG, typename Red, typename T,
         typename std::enable_if<(sizeof(T)!=4), int>::type = 0>
static __device__ __forceinline__ void allreduceLamport1ShotPoisonPerRank(
    ncclSymkArgsHandler const& handler, int nAllElts,
    Red red, ncclSymPtr<T> input, ncclSymPtr<T> output,
    ncclSymPtr<T> accumBuffer
  ) {}


/**
 * Lamport-style 1-shot AllReduce kernel for fp32.
 * The algorithm comes from Anton's
 * This kernel relies on the Lamport poison value to perform synchronization.
 * The difference between this kernel and the origina Lamport1Shot is that instead
 * of doing atomic reduction to the accumulator buffer, this
 * kernel has all ranks perform store to all other ranks, then the SM will
 * perform the reduction locally.
 * Each sub-warp, specified by SUB_LOG, will process the data from SUBRANKS number
 * of ranks. For instance, if SUBRANKS is 4 and SUB_LOG is 2, then each sub-warp
 * consists of 8 threads, and each sub-warp will process the data from 4 ranks.
 */
template<int SUBRANKS, int SUB_LOG, typename Red, typename T,
         typename std::enable_if<(sizeof(T)==4), int>::type = 0>
static __device__ __forceinline__ void allreduceLamport1ShotPoisonPerRank(
    ncclSymkArgsHandler const& handler, int nAllElts,
    Red red, ncclSymPtr<T> input, ncclSymPtr<T> output,
    ncclSymPtr<T> accumBuffer
  ) {

    ncclTeam world = ncclTeamWorld(handler.comm);
    int const& myrank = handler.comm.rank;
    int const& nranks = handler.comm.nRanks;


    assert(gridDim.y == 1 && gridDim.z == 1);
    int ctas = gridDim.x; // Total number of CTAs

    const int warpId = threadIdx.x >> 5;
    // The thread ID within the sub-warp
    const int subWarpThreadId = threadIdx.x & (31 >> SUB_LOG);

    const int myThreadIdx = warpId * (WARP_SIZE >> SUB_LOG) + subWarpThreadId;

    // The data segment worked on by this thread
    const int elt = myThreadIdx + (blockDim.x >> SUB_LOG) * blockIdx.x;

    // Determines which sub-warp this thread belongs to within its warp
    const int subWarpId = (threadIdx.x & 31) >> (MAX_SUB_LOG - SUB_LOG);


    if (elt >= nAllElts)
      return;


    // printf("[Rank %d] elt: %d, subWarpId: %d, subWarpThreadId: %d, myThreadIdx: %d, blockIdx.x: %d\n", myrank, elt, subWarpId, subWarpThreadId, myThreadIdx, blockIdx.x);
    // Initialize the value in the accumulation buffer to the poison value
    // #pragma unroll
    // for (int r = 0; r < SUBRANKS; ++r)
    // {
    //   int offset = elt + (subWarpId * SUBRANKS + r) * nAllElts;
    //   store32_poison(&((uint32_t*)((ncclSymPtr<T>)accumBuffer).peerPtr(world, myrank))[offset]);
    // }


    // Load the data from the input buffer if the thread belongs to sub-warp 0
    if (!subWarpId)
    {
      float inputVal = ((float*)((ncclSymPtr<T>)input).peerPtr(world, myrank))[elt];
      // Broadcast the input value to all other ranks
      int offset = elt + myrank * nAllElts;
      #pragma unroll
      for (int r = 0; r < nranks; ++r)
      {
        store32(&((float*)((ncclSymPtr<T>)accumBuffer).peerPtr(world, r))[offset], inputVal);
      }
    }

    uint32_t v[SUBRANKS];
    float *vFloat = reinterpret_cast<float *>(v);

    float sum = 0.0f;

    // int offsets[SUBRANKS];
    // #pragma unroll
    // for (int r = 0; r < SUBRANKS; ++r)
    //   offsets[r] = elt + (subWarpId * SUBRANKS + r) * nAllElts;

    #pragma unroll 1
    while (true)
    {
      bool allReceived = true;
      #pragma unroll
      for (int r = 0; r < SUBRANKS; ++r)
      {
        int offset = elt + (subWarpId * SUBRANKS + r) * nAllElts;
        // load32_int(&((uint32_t*)((ncclSymPtr<T>)accumBuffer).peerPtr(world, myrank))[offsets[r]], v[r]);
        load32_int(&((uint32_t*)((ncclSymPtr<T>)accumBuffer).peerPtr(world, myrank))[offset], v[r]);
        allReceived &= (v[r] != (uint32_t) NCCL_LAMPORT_INT);
        // printf("[Rank %d] elt: %d, offset: %d, v[r]: %x\n", myrank, elt, offset, v[r]);
      }
      if (allReceived) break;
    }

    #pragma unroll
    for (int r = 0; r < SUBRANKS; ++r)
    {
      sum += vFloat[r];
      int offset = elt + (subWarpId * SUBRANKS + r) * nAllElts;
      store32_poison(&((uint32_t*)((ncclSymPtr<T>)accumBuffer).peerPtr(world, myrank))[offset]);
    }


    // Performs warp shuffle to sum the values from the participating lanes
    // if (SUB_LOG > 0)
    // {
    //   float othersum = __shfl_xor_sync(0xFFFFFFFF, sum, 16);
    //   if (subWarpId >> (SUB_LOG - 1) == 0) sum += othersum;
    // }

    // if (SUB_LOG > 1)
    // {
    //   float othersum = __shfl_xor_sync(0xFFFFFFFF, sum, 8);
    //   if (subWarpId >> (SUB_LOG - 2) == 0) sum += othersum;
    // }

    // if (SUB_LOG > 2)
    // {
    //   float othersum = __shfl_xor_sync(0xFFFFFFFF, sum, 4);
    //   if (subWarpId >> (SUB_LOG - 3) == 0) sum += othersum;
    // }

    // Store the result to the output buffer
    if (!subWarpId)
    {
      ((float*)((ncclSymPtr<T>)output).peerPtr(world, myrank))[elt] = sum;
    }
}


// Lamport-style 2-shot AllReduce kernel with multimem broadcast
// Grid configuration: dim3(ctas, nRanks) - each block targets a different rank
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_Lamport1ShotPoison(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};

  Red<typename ncclSymkAccumType<Red, T, /*nvls=*/false>::Type> red(handler.devWork->redOpArg);

  struct ncclSymkDevWork const& dw = handler.devWork[0];

  size_t nAllElts = dw.nElts;
  ncclSymPtr<T> input(dw.inputWin, dw.inputOff);
  ncclSymPtr<T> output(dw.outputWin, dw.outputOff);


  if (sizeof(T) == 4) {
    constexpr int SUBRANKS = 4;
    constexpr int SUB_LOG = 0;

    // Get accumulation buffer from device communicator
    if (!((ncclSymkDevComm*)&handler.comm)->accumBuffer) {
      if (handler.comm.rank == 0 && blockIdx.x == 0 && threadIdx.x == 0) {
        printf("ERROR: Lamport 1-shot Poison accumulation buffer not allocated!\n");
      }
      return;
    }

    // Create ncclSymPtr from the allocated accumulation buffer
    ncclSymPtr<T> accumBuffer;
    accumBuffer.offset = ((ncclSymkDevComm*)&handler.comm)->lamportAccumOffset;
    accumBuffer.window = ((ncclSymkDevComm*)&handler.comm)->accumBuffer;


    // Call the multimem per-rank kernel function with hardcoded 1024 threads
    allreduceLamport1ShotPoisonPerRank<SUBRANKS, SUB_LOG>(
      handler, nAllElts, red, input, output, accumBuffer
    );
  } else {
    // Fallback for non-fp32 types - use existing deterministic path
    printf("Lamport 1-shot currently only supports fp32 (sizeof(T)==4)\n");
    // Could call existing allreduceEnds here as fallback
  }
}


/**
 * Lamport-style 1-shot AllReduce kernel
 * This kernel is similar to Lamport2Shot, but uses a single shot instead of two
 * It is used for small data sizes where the overhead of two shots is too high
 * Currently only supports fp32
 */
template<int SUBRANKS, int SUB_LOG, typename Red, typename T,
         typename std::enable_if<(sizeof(T)!=4), int>::type = 0>
static __device__ __forceinline__ void allreduceLamport1ShotPoisonMultimemPerRank(
    ncclSymkArgsHandler const& handler, int nAllElts,
    Red red, ncclSymPtr<T> input, ncclSymPtr<T> output,
    ncclSymPtr<T> accumBuffer
  ) {}


/**
 * Lamport-style 1-shot AllReduce kernel for fp32.
 * The algorithm comes from Anton's
 * This kernel relies on the Lamport poison value to perform synchronization.
 * The difference between this kernel and the origina Lamport1Shot is that instead
 * of doing atomic reduction to the accumulator buffer, this
 * kernel has all ranks perform store to all other ranks, then the SM will
 * perform the reduction locally.
 * Each sub-warp, specified by SUB_LOG, will process the data from SUBRANKS number
 * of ranks. For instance, if SUBRANKS is 4 and SUB_LOG is 2, then each sub-warp
 * consists of 8 threads, and each sub-warp will process the data from 4 ranks.
 */
template<int SUBRANKS, int SUB_LOG, typename Red, typename T,
         typename std::enable_if<(sizeof(T)==4), int>::type = 0>
static __device__ __forceinline__ void allreduceLamport1ShotPoisonMultimemPerRank(
    ncclSymkArgsHandler const& handler, int nAllElts,
    Red red, ncclSymPtr<T> input, ncclSymPtr<T> output,
    ncclSymPtr<T> accumBuffer
  ) {

    ncclTeam world = ncclTeamWorld(handler.comm);
    int const& myrank = handler.comm.rank;
    int const& nranks = handler.comm.nRanks;
    auto const& multimem = handler.comm.lsaMultimem;


    assert(gridDim.y == 1 && gridDim.z == 1);
    int ctas = gridDim.x; // Total number of CTAs

    const int warpId = threadIdx.x >> 5;
    // The thread ID within the sub-warp
    const int subWarpThreadId = threadIdx.x & (31 >> SUB_LOG);

    const int myThreadIdx = warpId * (WARP_SIZE >> SUB_LOG) + subWarpThreadId;

    // The data segment worked on by this thread
    const int elt = myThreadIdx + (blockDim.x >> SUB_LOG) * blockIdx.x;

    // Determines which sub-warp this thread belongs to within its warp
    const int subWarpId = (threadIdx.x & 31) >> (MAX_SUB_LOG - SUB_LOG);


    if (elt >= nAllElts)
      return;


    // printf("[Rank %d] elt: %d, subWarpId: %d, subWarpThreadId: %d, myThreadIdx: %d, blockIdx.x: %d\n", myrank, elt, subWarpId, subWarpThreadId, myThreadIdx, blockIdx.x);
    // Initialize the value in the accumulation buffer to the poison value
    // #pragma unroll
    // for (int r = 0; r < SUBRANKS; ++r)
    // {
    //   int offset = elt + (subWarpId * SUBRANKS + r) * nAllElts;
    //   store32_poison(&((uint32_t*)((ncclSymPtr<T>)accumBuffer).peerPtr(world, myrank))[offset]);
    // }

    // Load the data from the input buffer if the thread belongs to sub-warp 0
    if (!subWarpId)
    {
      float inputVal = ((float*)((ncclSymPtr<T>)input).peerPtr(world, myrank))[elt];
      // Broadcast the input value to all other ranks
      int offset = elt + myrank * nAllElts;
      store32_mc(&((float*)((ncclSymPtr<T>)accumBuffer).multimemPtr(multimem))[offset], inputVal);
    }

    uint32_t v[SUBRANKS];
    float *vFloat = reinterpret_cast<float *>(v);

    float sum = 0.0f;

    // int offsets[SUBRANKS];
    // #pragma unroll
    // for (int r = 0; r < SUBRANKS; ++r)
    //   offsets[r] = elt + (subWarpId * SUBRANKS + r) * nAllElts;

    #pragma unroll 1
    while (true)
    {
      bool allReceived = true;
      #pragma unroll
      for (int r = 0; r < SUBRANKS; ++r)
      {
        int offset = elt + (subWarpId * SUBRANKS + r) * nAllElts;
        load32_int(&((uint32_t*)((ncclSymPtr<T>)accumBuffer).peerPtr(world, myrank))[offset], v[r]);
        allReceived &= (v[r] != (uint32_t) NCCL_LAMPORT_INT);
        // printf("[Rank %d] elt: %d, offset: %d, v[r]: %x\n", myrank, elt, offset, v[r]);
      }
      if (allReceived) break;
    }

    #pragma unroll
    for (int r = 0; r < SUBRANKS; ++r)
    {
      sum += vFloat[r];
      int offset = elt + (subWarpId * SUBRANKS + r) * nAllElts;
      store32_poison(&((uint32_t*)((ncclSymPtr<T>)accumBuffer).peerPtr(world, myrank))[offset]);
    }


    // Performs warp shuffle to sum the values from the participating lanes
    if (SUB_LOG > 0)
    {
      float othersum = __shfl_xor_sync(0xFFFFFFFF, sum, 16);
      if (subWarpId >> (SUB_LOG - 1) == 0) sum += othersum;
    }

    if (SUB_LOG > 1)
    {
      float othersum = __shfl_xor_sync(0xFFFFFFFF, sum, 8);
      if (subWarpId >> (SUB_LOG - 2) == 0) sum += othersum;
    }

    if (SUB_LOG > 2)
    {
      float othersum = __shfl_xor_sync(0xFFFFFFFF, sum, 4);
      if (subWarpId >> (SUB_LOG - 3) == 0) sum += othersum;
    }

    // Store the result to the output buffer
    if (!subWarpId)
    {
      ((float*)((ncclSymPtr<T>)output).peerPtr(world, myrank))[elt] = sum;
    }
}


// Lamport-style 2-shot AllReduce kernel with multimem broadcast
// Grid configuration: dim3(ctas, nRanks) - each block targets a different rank
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_Lamport1ShotPoisonMC(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};

  Red<typename ncclSymkAccumType<Red, T, /*nvls=*/false>::Type> red(handler.devWork->redOpArg);

  struct ncclSymkDevWork const& dw = handler.devWork[0];

  size_t nAllElts = dw.nElts;
  ncclSymPtr<T> input(dw.inputWin, dw.inputOff);
  ncclSymPtr<T> output(dw.outputWin, dw.outputOff);


  if (sizeof(T) == 4) {
    constexpr int SUBRANKS = 32;
    constexpr int SUB_LOG = 0;

    // Get accumulation buffer from device communicator
    if (!((ncclSymkDevComm*)&handler.comm)->accumBuffer) {
      if (handler.comm.rank == 0 && blockIdx.x == 0 && threadIdx.x == 0) {
        printf("ERROR: Lamport 1-shot Poison MC accumulation buffer not allocated!\n");
      }
      return;
    }

    // Create ncclSymPtr from the allocated accumulation buffer
    ncclSymPtr<T> accumBuffer;
    accumBuffer.offset = ((ncclSymkDevComm*)&handler.comm)->lamportAccumOffset;
    accumBuffer.window = ((ncclSymkDevComm*)&handler.comm)->accumBuffer;


    // Call the multimem per-rank kernel function with hardcoded 1024 threads
    allreduceLamport1ShotPoisonMultimemPerRank<SUBRANKS, SUB_LOG>(
      handler, nAllElts, red, input, output, accumBuffer
    );
  } else {
    // Fallback for non-fp32 types - use existing deterministic path
    printf("Lamport 1-shot currently only supports fp32 (sizeof(T)==4)\n");
    // Could call existing allreduceEnds here as fallback
  }
}


template<int SUBRANKS, int SUB_LOG, int NTHREADS, typename Red, typename T,
         typename std::enable_if<(sizeof(T)!=4), int>::type = 0>
static __device__ __forceinline__ void allreduceLamport2ShotPoisonPerRank(
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
template<int SUBRANKS, int SUB_LOG, int NTHREADS, typename Red, typename T,
         typename std::enable_if<(sizeof(T)==4), int>::type = 0>
static __device__ __forceinline__ void allreduceLamport2ShotPoisonPerRank(
    ncclSymkArgsHandler const& handler, int nAllElts,
    Red red, ncclSymPtr<T> input, ncclSymPtr<T> output,
    ncclSymPtr<T> accumBuffer
  ) {
  ncclTeam world = ncclTeamWorld(handler.comm);
  int const& myrank = handler.comm.rank;
  int const& nranks = handler.comm.nRanks;
  int const& accumOffset = accumBuffer.offset;
  int const& slot = accumOffset / REDUCTION_BUFFER_SIZE;

  // CTAs per rank is equal to the X dimension of the grid, which is assigned in enqueue.cc
  int ctasPerRank = gridDim.x;
  // Calculates the maximum number of elements that can be processed by a single iteration
  int maxEltsPerIter = ctasPerRank * nranks * NTHREADS * 4;
  // int maxEltsPerIterPerRank = maxEltsPerIter;
  int maxAccumBuffIter = max((int) (REDUCTION_BUFFER_SIZE / sizeof(T) / maxEltsPerIter), 1); // Number of iterations that fit in accumBuffer
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
  bool flagcarrier = ((threadIdx.x & 7) == 0); // Every 8th thread carries flag

  const int warpId = threadIdx.x >> 5;
  const int subWarpThreadId = threadIdx.x & (31 >> SUB_LOG);
  const int myThreadIdx = warpId * (WARP_SIZE >> SUB_LOG) + subWarpThreadId;
  const int subWarpId = (threadIdx.x & 31) >> (MAX_SUB_LOG - SUB_LOG);

  float4 v; // Use float4 for vectorized operations
  float *elt_v = reinterpret_cast<float*>(&v);

  for (int i = 0; i < nIterations; ++i)
  {

    // Computes the offset of the current iteration
    int offset = i * maxEltsPerIter;
    currInput = input + offset;
    currOutput = output + offset;
    currAccumBuffer = accumBuffer + (i % maxAccumBuffIter) * maxEltsPerIter;

    int perranklines = NTHREADS * ctasPerRank;
    int last_cta_nthreads = NTHREADS;
    int numCTAs = ctasPerRank;

    if (i == nIterations - 1)
    {
      // If this is the last iteration
      perranklines = (nAllElts - offset) / nranks / 4;
      // Calculates the number of CTAs needed for the last iteration
      numCTAs = (perranklines + NTHREADS - 1) / NTHREADS;
      if (blockIdx.x >= numCTAs)
      {
        // If this CTA is not part of the last iteration, return
        continue;
      }
      last_cta_nthreads = perranklines % NTHREADS == 0 ? NTHREADS : perranklines % NTHREADS;
    }

    const int maxthread = blockIdx.x == numCTAs - 1 ? last_cta_nthreads : NTHREADS;

    if (threadIdx.x >= maxthread)
      // If this thread is not part of the last iteration, return
      break;

    int eltLine = myThreadIdx + (blockDim.x >> SUB_LOG) * blockIdx.x;

    if (!subWarpId)
    {
      int inputLine = eltLine + target_rank * perranklines;
      // Load the data from the input buffer only if the thread belongs to sub-warp 0
      float4* inputPtr = (float4*)(((ncclSymPtr<T>)currInput).peerPtr(world, myrank));
      v = inputPtr[inputLine];

      // Poison the output buffer with NCCL_LAMPORT_INT
      float4* outputPtr = (float4*)(((ncclSymPtr<T>)currOutput).peerPtr(world, myrank));
      store128_poison(&outputPtr[inputLine]);

      __threadfence();

      int offset = eltLine + myrank * perranklines;
      store128(&((float4*)((ncclSymPtr<T>)currAccumBuffer).peerPtr(world, target_rank))[offset], v);

    }
  } // End of Phase 1: sending messages to the target rank


  // Loop 2: receiving messages from the target rank and broadcasting the result to all ranks
  for (int i = 0; i < nIterations; ++i)
  {
    // Computes the offset of the current iteration
    int offset = i * maxEltsPerIter;
    currOutput = output + offset;
    currAccumBuffer = accumBuffer + (i % maxAccumBuffIter) * maxEltsPerIter;

    int perranklines = NTHREADS * ctasPerRank;
    int last_cta_nthreads = NTHREADS;
    int numCTAs = ctasPerRank;

    if (i == nIterations - 1)
    {
      // If this is the last iteration
      perranklines = (nAllElts - offset) / nranks / sizeof(float);
      // Calculates the number of CTAs needed for the last iteration
      numCTAs = (perranklines + NTHREADS - 1) / NTHREADS;
      // If this CTA is not part of the last iteration, return
      if (blockIdx.x >= numCTAs)
        return;

      last_cta_nthreads = perranklines % NTHREADS == 0 ? NTHREADS : perranklines % NTHREADS;
    }

    const int maxthread = blockIdx.x == numCTAs - 1 ? last_cta_nthreads : NTHREADS;

    if (threadIdx.x >= maxthread)
      return;

    uint4 v[SUBRANKS];
    float4 *vFloat4 = reinterpret_cast<float4 *>(v);
    float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};

    int eltLine = myThreadIdx + (blockDim.x >> SUB_LOG) * blockIdx.x;
    if (target_rank == myrank)
    {
      // Polling
      #pragma unroll 1
      while (true)
      {
        bool allReceived = true;
        #pragma unroll
        for (int r = 0; r < SUBRANKS; ++r)
        {
          int offset = eltLine + (subWarpId * SUBRANKS + r) * perranklines;
          load128_int(&((uint4*)((ncclSymPtr<T>)currAccumBuffer).peerPtr(world, target_rank))[offset], v[r]);
          allReceived &= (v[r].x != (uint32_t) NCCL_LAMPORT_INT);
        }
        if (allReceived) break;
      }

      #pragma unroll
      for (int r = 0; r < SUBRANKS; ++r)
      {
        sum.x += vFloat4[r].x;
        sum.y += vFloat4[r].y;
        sum.z += vFloat4[r].z;
        sum.w += vFloat4[r].w;
        int offset = eltLine + (subWarpId * SUBRANKS + r) * perranklines;
        store128_poison(&((float4*)((ncclSymPtr<T>)currAccumBuffer).peerPtr(world, target_rank))[offset]);
      }
      // TODO: Need warp shuffle to sum the values from all sub-warps


      if (!subWarpId)
      {
        int offset = eltLine + myrank * perranklines;
        // Broadcast the result to all ranks
        #pragma unroll
        for (int r = 0; r < nranks; ++r)
        {
          float4* outputPtr = (float4*)(((ncclSymPtr<T>)currOutput).peerPtr(world, r));
          store128(&outputPtr[offset], sum);
        }
      }
    }

    // Poll for completion (all ranks)
    uint4* outputPtr = (uint4*)(((ncclSymPtr<T>)currOutput).peerPtr(world, myrank));
    while (true)
    {
      uint4 result;
      int offset = eltLine + target_rank * perranklines;
      load128_int(&outputPtr[offset], result);
      if (result.x != (uint32_t) NCCL_LAMPORT_INT) break;
    }

  } // End of Loop 2: receiving messages from the target rank and broadcasting the result to all ranks
}

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_Lamport2ShotPoison(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};

  Red<typename ncclSymkAccumType<Red, T, /*nvls=*/false>::Type> red(handler.devWork->redOpArg);

  struct ncclSymkDevWork const& dw = handler.devWork[0];

  size_t nAllElts = dw.nElts;
  ncclSymPtr<T> input(dw.inputWin, dw.inputOff);
  ncclSymPtr<T> output(dw.outputWin, dw.outputOff);

  if (sizeof(T) == 4) {
    constexpr int SUBRANKS = 8;  // Number of ranks per sub-warp
    constexpr int SUB_LOG = 0;  // Number of sub-warps per rank. 0 -> 1 sub-warp per rank, 1 -> 2 sub-warps per rank, etc.
    constexpr int NTHREADS = 512;  // Total number of threads per CTA

    int nRanks = handler.comm.nRanks;
    int rank = handler.comm.rank;

    // Get accumulation buffer from device communicator
    if (!((ncclSymkDevComm*)&handler.comm)->accumBuffer) {
      if (handler.comm.rank == 0 && blockIdx.x == 0 && threadIdx.x == 0) {
        printf("ERROR: Lamport 2-shot Poison accumulation buffer not allocated!\n");
      }
      return;
    }

    // Create ncclSymPtr from the allocated accumulation buffer
    ncclSymPtr<T> accumBuffer;
    accumBuffer.offset = ((ncclSymkDevComm*)&handler.comm)->lamportAccumOffset;
    accumBuffer.window = ((ncclSymkDevComm*)&handler.comm)->accumBuffer;


    allreduceLamport2ShotPoisonPerRank<SUBRANKS, SUB_LOG, NTHREADS>(
      handler, nAllElts, red, input, output, accumBuffer
    );
  } else {
    // Fallback for non-fp32 types - use existing deterministic path
    printf("Lamport 2-shot currently only supports fp32 (sizeof(T)==4)\n");
    // Could call existing allreduceEnds here as fallback
  }
}


template<typename Red, typename T,
         typename std::enable_if<(sizeof(T)!=4), int>::type = 0>
static __device__ __forceinline__ void allreduceSOLPerRank(
    ncclSymkArgsHandler const& handler, int nAllElts,
    Red red, ncclSymPtr<T> input, ncclSymPtr<T> output,
    ncclSymPtr<T> accumBuffer
  ) {}


/**
 * The fastest AllReduce kernel for fp32, does not guarantee correctness, just
 * wants to know what the theoretical limit is in terms of latency if
 * implemented with atomic operations.
 */
template<typename Red, typename T,
         typename std::enable_if<(sizeof(T)==4), int>::type = 0>
static __device__ __forceinline__ void allreduceSOLPerRank(
    ncclSymkArgsHandler const& handler, int nAllElts,
    Red red, ncclSymPtr<T> input, ncclSymPtr<T> output,
    ncclSymPtr<T> accumBuffer
  ) {

  ncclTeam world = ncclTeamWorld(handler.comm);
  int const& myrank = handler.comm.rank;
  int const& nranks = handler.comm.nRanks;

  int elt = threadIdx.x + blockDim.x * blockIdx.x;
  if (elt >= nAllElts)
    return;

  float accum = 0.0f;

}


// Speed-of-light AllReduce kernel, it is just a kernel that tests the theoretical
// minimum latency of the AllReduce operation.
// It simply performs a load, an atomic add, and a load, and another store to the output buffer.
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_SOL(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};

  Red<typename ncclSymkAccumType<Red, T, /*nvls=*/false>::Type> red(handler.devWork->redOpArg);

  // bar.arrive(ncclCoopCta(), cuda::memory_order_relaxed);

  struct ncclSymkDevWork const& dw = handler.devWork[0];

  size_t nAllElts = dw.nElts;
  ncclSymPtr<T> input(dw.inputWin, dw.inputOff);
  ncclSymPtr<T> output(dw.outputWin, dw.outputOff);
  ncclSymPtr<T> accumBuffer;
  accumBuffer.offset = 0;
  accumBuffer.window = ((ncclSymkDevComm*)&handler.comm)->accumBuffer;

  if (sizeof(T) == 4) {
    // Call the multimem per-rank kernel function with hardcoded 1024 threads
    // allreduceSOLPerRank(
    //   handler, nAllElts, red, input, output, accumBuffer
    // );
  } else {
    // Fallback for non-fp32 types - use existing deterministic path
    printf("Lamport 1-shot currently only supports fp32 (sizeof(T)==4)\n");
    // Could call existing allreduceEnds here as fallback
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
template<ncclLLSyncMode Mode, bool Multimem, int Unroll, template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_LL_impl(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};


  struct ncclSymkDevWork const& dw = handler.devWork[0];
  size_t nAllElts = dw.nElts;
  int const& rank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;
  using Acc = typename ncclSymkAccumType<Red, T, /*nvls=*/Multimem>::Type;
  Red<Acc> red(handler.devWork->redOpArg);

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

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

  ncclSymPtr<T> input(dw.inputWin, dw.inputOff);
  ncclSymPtr<T> output(dw.outputWin, dw.outputOff);
  T* inputPtr = (T*)input.localPtr();
  T* outputPtr = (T*)output.localPtr();
  // int tn = ncclSymkMaxThreads;

  constexpr int BytesPerPack = 8;
  using Pack = BytePack<BytesPerPack>;
  using AccPack = BytePack<BytesPerPack*sizeof(Acc)/sizeof(T)>;
  constexpr int nEltsPerPack = BytesPerPack / sizeof(T);


  // Calculate bytesPerCTA for the LL buffer
  // This is the size of one buffer slot region per block
  size_t bytesPerCtaPerEpoch = nRanks * blockDim.x * BytesPerPack;

  int roundRobinFactor = REDUCTION_BUFFER_SIZE / (bytesPerCtaPerEpoch * gridDim.x);
  if (Mode == ncclLL)
    // Divide the multiple buffering factor by 2 for LL sync mode
    roundRobinFactor >>= 1;

  if (roundRobinFactor < 1) {
    printf("[ERROR]: roundRobinFactor < 1\n");
    return;
  }

  roundRobinFactor = min(roundRobinFactor, UINT8_MAX);
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

  // Main loop with compile-time Unroll factor
  #pragma unroll 1
  for (int i = tid; i < nPacks; i += nthreads) {
    Pack myData = loadPack<Pack>((T*) inputPtr, i * nEltsPerPack, nAllElts);
    int slot = threadIdx.x + rank * blockDim.x;
    llBuf.template bcast<Unroll, Pack>(team, slot, myData);

    AccPack result = llBuf.template recvReduce<Unroll, Pack, /*Reset=*/true>(
      /*eltStart=*/ threadIdx.x,
      /*eltCount=*/ nRanks,
      /*eltStride=*/ blockDim.x,
      /*eltToAcc=*/ [&] __device__ (Pack x) -> AccPack { return applyCast<T, Acc>(x); },
      /*reduce=*/ [&] __device__ (AccPack a, AccPack b) -> AccPack { return applyReduce(red, a, b); }
    );
    storePack<Pack>((T*) outputPtr, i * nEltsPerPack, nAllElts, applyCast<Acc, T>(result));
    llBuf.advanceEpoch();
  }
}

// Public entry points used by the symmetric-kernel generator.
// - AllReduce_LLBuffer: base version for non-power-of-2 ranks (Unroll=4)
// - AllReduce_LLBuffer_R{2,4,8,16,32,64}: rank-specialized versions
// - AllReduce_LLBuffer_LL16*: same but with LL sync mode

// Base versions (for non-power-of-2 ranks, uses Unroll=4)
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_LLBuffer(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_LL_impl<ncclPoison, /*Multimem=*/false, /*Unroll=*/4, Red, T>(args);
}

// Rank-specialized versions (Poison mode)
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_LLBuffer_R4(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_LL_impl<ncclPoison, /*Multimem=*/false, /*Unroll=*/4, Red, T>(args);
}
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_LLBuffer_R8(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_LL_impl<ncclPoison, /*Multimem=*/false, /*Unroll=*/8, Red, T>(args);
}
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_LLBuffer_R16(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_LL_impl<ncclPoison, /*Multimem=*/false, /*Unroll=*/16, Red, T>(args);
}
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_LLBuffer_R32(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_LL_impl<ncclPoison, /*Multimem=*/false, /*Unroll=*/32, Red, T>(args);
}

// LL16 sync mode versions
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_LLBuffer_LL16(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_LL_impl<ncclLL, /*Multimem=*/false, /*Unroll=*/4, Red, T>(args);
}
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_LLBuffer_LL16_R4(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_LL_impl<ncclLL, /*Multimem=*/false, /*Unroll=*/4, Red, T>(args);
}
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_LLBuffer_LL16_R8(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_LL_impl<ncclLL, /*Multimem=*/false, /*Unroll=*/8, Red, T>(args);
}
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_LLBuffer_LL16_R16(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_LL_impl<ncclLL, /*Multimem=*/false, /*Unroll=*/16, Red, T>(args);
}
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_LLBuffer_LL16_R32(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_LL_impl<ncclLL, /*Multimem=*/false, /*Unroll=*/32, Red, T>(args);
}


// Multimem version - uses multicast for broadcast
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_LLBufferMC(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_LL_impl<ncclPoison, /*Multimem=*/true, /*Unroll=*/4, Red, T>(args);
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
template<int Unroll, template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_LLBuffer_Twoshot_impl(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};

  struct ncclSymkDevWork const& dw = handler.devWork[0];
  size_t nAllElts = dw.nElts;
  int const& rank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;
  using Acc = typename ncclSymkAccumType<Red, T, /*nvls=*/false>::Type;
  Red<Acc> red(handler.devWork->redOpArg);

  // Two-shot requires nAllElts divisible by nRanks
  size_t nEltsPerRank = nAllElts / nRanks;
  if (nEltsPerRank * nRanks != nAllElts) {
    // Fallback to one-shot for non-divisible sizes
    ncclSymkRun_AllReduce_LL_impl<ncclPoison, /*Multimem=*/false, Unroll, Red, T>(args);
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
  // size_t bytesPerCtaPerEpoch = nRanks * blockDim.x * BytesPerPack;
  size_t bytesPerCtaPerEpoch = nAllElts * sizeof(T) / gridDim.x;
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

  ncclLLBuffer<ncclPoison, /*Multimem=*/false> reductionBuf(
    scratchSymPtr,
    /*bytesPerCtaPerEpoch=*/ REDUCTION_BUFFER_SIZE,
    /*block=*/ 0,
    /*roundRobinFactor=*/ 0,
    /*mmHandle=*/ ncclMultimemHandle{}
  );

  ncclLLBuffer<ncclPoison, /*Multimem=*/false> outputBuf(
    output,
    /*bytesPerCtaPerEpoch=*/ 0,
    /*block=*/ 0,
    /*roundRobinFactor=*/ 0,
    /*mmHandle=*/ ncclMultimemHandle{}
  );

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
      Pack result = reductionBuf.template recvReduce<Unroll, Pack, /*Reset=*/true>(
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
    int nIterations = (nTotalPacks + nthreads - 1) / nthreads;

    int packsPerRank = ctasPerRank * blockDim.x;
    int lastCtaThreads = blockDim.x;
    int numCTAs = ctasPerRank;

    for (int i = 0; i < nIterations; ++i) {
      if (i == nIterations - 1) {
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
    }

    packsPerRank = ctasPerRank * blockDim.x;
    lastCtaThreads = blockDim.x;
    numCTAs = ctasPerRank;

    for (int i = 0; i < nIterations; ++i) {
      // Compute the number of packs that needs to be processed per CTA for this iteration
      if (i == nIterations - 1) {
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
        Pack result = reductionBuf.template recvReduce<Unroll, Pack, /*Reset=*/true>(
          /*eltStart=*/ slot,
          /*eltCount=*/ nRanks,
          /*eltStride=*/ nPacksPerRank,
          /*eltToAcc=*/ [&] __device__ (Pack x) -> Pack { return x; },
          /*reduce=*/ [&] __device__ (Pack a, Pack b) -> Pack { return applyReduce(red, a, b); }
        );

        outputBuf.template bcast<Unroll, Pack>(team, rank * nPacksPerRank + slot, result);
      }
      outputBuf.template recv<Pack, /*Reset=*/false>(targetRank * nPacksPerRank + slot);
    }
  }
}

// Public entry point for Two-Shot AllReduce (base version for non-power-of-2 ranks)
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_LLBuffer_Twoshot(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_LLBuffer_Twoshot_impl</*Unroll=*/4, Red, T>(args);
}

// Rank-specialized versions for Two-Shot AllReduce
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_LLBuffer_Twoshot_R4(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_LLBuffer_Twoshot_impl</*Unroll=*/4, Red, T>(args);
}
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_LLBuffer_Twoshot_R8(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_LLBuffer_Twoshot_impl</*Unroll=*/8, Red, T>(args);
}
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_LLBuffer_Twoshot_R16(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_LLBuffer_Twoshot_impl</*Unroll=*/16, Red, T>(args);
}