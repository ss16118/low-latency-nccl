#include "sym_kernels.h"
#include "kernel.cuh"
#include "primitives.cuh"
#include "nccl_device/ll_buffer.h"

template<int BytePerPack, int UnrollPacks, int UnrollPeers, typename T, typename Red>
static __device__ void reduceDeep(
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
      int r = rank+1;
      if (r == nRanks) r = 0;
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

      r += 1;
      if (r == nRanks) r = 0;

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
            r += 1;
            if (r == nRanks) r = 0;
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

      #pragma unroll UnrollPacks
      for (int u=0; u < UnrollPacks; u++) outPacks.localPtr()[u*WARP_SIZE] = acc0[u];

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

template<int UnrollPeers, typename Red, typename T>
static __device__ void reduceEnds(
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
    outPacks.localPtr()[elt] = acc0;
  }
}

template<typename Red, typename T>
static __device__ void reduce(
    ncclSymkArgsHandler const& handler, int tn, int t, int nBlocks,
    bool waitNeeded, ncclLsaBarrierSession<ncclCoopCta>& bar,
    Red red, ncclSymPtr<T> input, ncclSymPtr<T> output, size_t nElts
  ) {
  int const& nRanks = handler.comm.nRanks;
  int const& nRanks_rcp32 = handler.nRanks_rcp32;
  uint32_t nBlocks_rcp32 = nccl::utility::idivRcp32_upto64(nBlocks);
  uint32_t nRanks_nBlocks_rcp32 = nccl::utility::imulRcp32(nRanks, nRanks_rcp32, nBlocks, nBlocks_rcp32);

  uint32_t alignment = uint32_t(input.offset - output.offset);
  size_t nBytes = nElts*sizeof(T);

  uint32_t nPreBytes = (16u - input.offset)%16u;
  nPreBytes = min((size_t)nPreBytes, nBytes);
  uintptr_t cursor = nPreBytes;

  constexpr int MinWarpPerBlock = 4;

  if (alignment%16 == 0) {
    constexpr int BytePerPack = 16, UnrollPacks = 4, UnrollPeers = 2;
    constexpr int BytePerChunk = MinWarpPerBlock*UnrollPacks*WARP_SIZE*BytePerPack;
    uint32_t chunks = (nBytes-cursor)/BytePerChunk;
    chunks -= imodFast32(chunks, nRanks*nBlocks, nRanks_nBlocks_rcp32);
    if (chunks != 0) {
      uintptr_t cursorAfter = cursor + uintptr_t(chunks)*BytePerChunk;
      reduceDeep<BytePerPack, UnrollPacks, UnrollPeers, T>(
        handler, tn, t, waitNeeded, bar, red,
        (ncclSymPtr<char>)input + cursor, (ncclSymPtr<char>)output + cursor,
        chunks*MinWarpPerBlock
      );
      cursor = cursorAfter;
      waitNeeded = false;
    }
  }

  if (sizeof(T) == 4 || (sizeof(T) < 4 && alignment%4 == 0)) {
    constexpr int BytePerPack = 4, UnrollPacks = 4, UnrollPeers = 4;
    constexpr int BytePerChunk = MinWarpPerBlock*UnrollPacks*WARP_SIZE*BytePerPack;
    uint32_t chunks = (nBytes-cursor)/BytePerChunk;
    chunks -= imodFast32(chunks, nRanks*nBlocks, nRanks_nBlocks_rcp32);
    if (chunks != 0) {
      uintptr_t cursorAfter = cursor + uintptr_t(chunks)*BytePerChunk;
      reduceDeep<(sizeof(T) <= BytePerPack ? BytePerPack : 0), UnrollPacks, UnrollPeers, T>(
        handler, tn, t, waitNeeded, bar, red,
        (ncclSymPtr<char>)input + cursor, (ncclSymPtr<char>)output + cursor,
        chunks*MinWarpPerBlock
      );
      cursor = cursorAfter;
      waitNeeded = false;
    }
  }

  if (waitNeeded) bar.wait(ncclCoopCta(), cuda::memory_order_relaxed);

  constexpr int UnrollPeers = 8;
  size_t nSufElts = (nBytes-cursor)/sizeof(T);
  reduceEnds<UnrollPeers>(handler, tn, t, red, input, output, nElts, nPreBytes/sizeof(T), nSufElts);
}

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_LD(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};
  ncclLsaBarrierSession<ncclCoopCta> bar{
    ncclCoopCta(), handler.comm, ncclTeamTagLsa(), blockIdx.x
  };
  Red<typename ncclSymkAccumType<Red, T, /*nvls=*/false>::Type> red(handler.devWork->redOpArg);
  int const& rank = handler.comm.rank;

  bar.arrive(ncclCoopCta(), cuda::memory_order_relaxed);

  bool waitNeeded = true;
  handler.forEachWork<T>(
      [&]__device__(int block, int nBlocks, size_t nElts, size_t nAllElts,
                    ncclSymPtr<T> input, ncclSymPtr<T> output) {
        // Round robin warps over blocks.
        int t = flattenIx(threadIdx.x%WARP_SIZE, WARP_SIZE,
                          block, nBlocks,
                          threadIdx.x/WARP_SIZE, blockDim.x/WARP_SIZE);
        int tn = nBlocks*blockDim.x;

        reduce(handler, tn, t, nBlocks, waitNeeded, bar, red, input + rank*nAllElts, output, nElts);

        waitNeeded = false;
      }
    );

  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);
}

template<typename Red, typename T>
static __device__ void reduceMultimem(
    int tn, int t, Red red, T* input, T* output, size_t nElts
  ) {
  uintptr_t inputUptr = reinterpret_cast<uintptr_t>(input);
  uintptr_t outputUptr = reinterpret_cast<uintptr_t>(output);
  size_t nBytes = nElts*sizeof(T);

  constexpr int BytePerPack = LoadMultimem_BigPackSize<Red>::BigPackSize;
  uint32_t nPreBytes = (BytePerPack - inputUptr)%BytePerPack;
  nPreBytes = min((size_t)nPreBytes, nBytes);
  uintptr_t nSufBytes;

  if (sizeof(T) == BytePerPack || (inputUptr-outputUptr)%BytePerPack == 0) {
    constexpr int UnrollPacks = 8*(16/BytePerPack);
    constexpr int BytePerChunk = UnrollPacks*WARP_SIZE*BytePerPack;
    uintptr_t cursor = nPreBytes;
    uint32_t nChunks = (nBytes-cursor)/BytePerChunk;
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
        *reinterpret_cast<BytePack<BytePerPack>*>(outputUptr + cursor + u*WARP_SIZE*BytePerPack) = tmp[u];
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
    *reinterpret_cast<BytePack<sizeof(T)>*>(outputUptr + cursor) = val;
  }
}

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_LDMC(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};
  ncclLsaBarrierSession<ncclCoopCta> bar{
    ncclCoopCta(), handler.comm, ncclTeamTagLsa(), blockIdx.x, /*multimem=*/true
  };
  Red<typename ncclSymkAccumType<Red, T, /*nvls=*/true>::Type> red(handler.devWork->redOpArg);

  int const& rank = handler.comm.rank;
  auto const& multimem = handler.comm.lsaMultimem;

  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  handler.forEachWork<T>(
      [&]__device__(int block, int nBlocks, size_t nElts, size_t nAllElts,
                    ncclSymPtr<T> input, ncclSymPtr<T> output) {
        // Round robin warps over blocks.
        int t = flattenIx(threadIdx.x%WARP_SIZE, WARP_SIZE,
                          block, nBlocks,
                          threadIdx.x/WARP_SIZE, blockDim.x/WARP_SIZE);
        int tn = nBlocks*blockDim.x;

        reduceMultimem(tn, t, red, input.multimemPtr(multimem) + rank*nAllElts, output.localPtr(), nElts);
      }
    );

  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);
}

// T is user type, EltType is the most aligned type
template<typename T, typename Red, typename EltType>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_LL_body(
    ncclSymkArgsHandler& handler, ncclLLA2ASession<ncclCoopCta>& lla2a,
    Red red, EltType* input, EltType* output, int nElts, int nPacks, int nStrideElts) {
  using Pack = BytePack<8>;
  using Acc = typename Red::EltType;
  using AccPack = BytePack<8*sizeof(Acc)/sizeof(T)>;
  constexpr int EltPerPack = 8/sizeof(EltType);

  int const& nRanks = handler.comm.nRanks;
  int const& rank = handler.comm.rank;
  int t = threadIdx.x;
  int tn = blockDim.x;  // Use actual thread count instead of hardcoded 512
  ncclCoopCta cta;

  #pragma unroll 1
  while (0 < nElts) {
    int nIterPacks = min(nPacks, tn);
    int tn_div_nPacks = tn/nIterPacks;
    int tn_mod_nPacks = tn%nIterPacks;
    int peer = t/nIterPacks;
    int pack = t%nIterPacks;

    #pragma unroll 1
    for (int i = t; i < nRanks*nIterPacks; i += tn) {
      Pack got = loadPack<Pack>(input + peer*nStrideElts, pack*EltPerPack, nElts);
      lla2a.send(peer, rank*nIterPacks + pack, got);
      peer += tn_div_nPacks;
      pack += tn_mod_nPacks;
      if (nIterPacks <= pack) { peer += 1; pack -= nIterPacks; }
    }

    if (t < nIterPacks) {
      AccPack got = lla2a.template recvReduce</*Unroll=*/8, Pack>(
        /*slotStart=*/t, /*slotCount=*/nRanks, /*slotStride=*/nIterPacks,
        /*eltToAcc=*/[&] __device__ (Pack x)->AccPack {
          return applyCast<T, Acc>(x);
        },
        /*reduce=*/[&] __device__ (AccPack a, AccPack b)->AccPack {
          return applyReduce(red, a, b);
        }
      );
      storePack(output, t*EltPerPack, nElts, applyCast<Acc, T>(got));
    }
    lla2a.endEpoch(cta);

    input += tn*EltPerPack;
    output += tn*EltPerPack;
    nElts -= tn*EltPerPack;
    nPacks -= tn;
  }
}

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_LL(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};
  ncclLLA2ASession<ncclCoopCta> lla2a(
    ncclCoopCta(), handler.comm, ncclTeamLsa(handler.comm), handler.lsaLLA2A, blockIdx.x, (int)blockDim.x
  );
  Red<typename ncclSymkAccumType<Red, T, /*nvls=*/false>::Type> red(handler.devWork->redOpArg);
  using Pack = BytePack<8>;
  constexpr int EltPerPack = 8/sizeof(T);

  handler.singleWork<T>(
      [&]__device__(int nElts, int nAllElts,
                    ncclSymPtr<T> inputPtr, ncclSymPtr<T> outputPtr) {
        int nPacks = divUp(nElts, EltPerPack);

        T* input = (T*)inputPtr.localPtr();
        T* output = (T*)outputPtr.localPtr();

        uint32_t lowBits = nAllElts*sizeof(T);
        lowBits |= (uintptr_t)input;
        lowBits |= (uintptr_t)output;
        if (__builtin_expect(lowBits%8 == 0, true)) {
          ncclSymkRun_ReduceScatter_LL_body<T>(handler, lla2a, red, (Pack*)input, (Pack*)output,
                                               nPacks, nPacks, divUp(nAllElts, EltPerPack));
        } else {
          ncclSymkRun_ReduceScatter_LL_body<T>(handler, lla2a, red, input, output, nElts, nPacks, nAllElts);
        }
      }
    );
}


/**
 * ReduceScatter kernel using ncclLLBuffer API.
 *
 * This kernel implements a reduce-scatter using the ncclLLBuffer abstraction:
 *   1. Input is divided into nRanks chunks; chunk r is destined for rank r
 *   2. For each pack i, all ranks send input[i] to targetRank = i / nPacksPerRank
 *   3. Only targetRank receives contributions from all ranks, reduces, and stores
 *
 * Memory layout for reduce-scatter:
 *   - Input: Each rank has nAllElts elements (nRanks chunks of nElts each)
 *   - Output: Rank r gets nElts = nAllElts / nRanks elements (reduced chunk r)
 *   - Scratch buffer: slot = srcRank * blockDim.x + (packInRank % blockDim.x)
 *
 * Key design points:
 *   - Uses ncclPoison sync mode for low-latency synchronization
 *   - Each thread handles one input pack per epoch
 *   - All ranks participate in sending; only targetRank receives/reduces/stores
 *   - Slot layout ensures each source rank's data is at a unique offset
 */
template<ncclLLSyncMode Mode, bool Multimem, int Unroll, template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_LLBuffer_impl(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};

  struct ncclSymkDevWork const& dw = handler.devWork[0];
  size_t nElts = dw.nElts; // Warning: this is the number of elements per rank in the output
  int const& rank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;
  using Acc = typename ncclSymkAccumType<Red, T, /*nvls=*/Multimem>::Type;
  Red<Acc> red(handler.devWork->redOpArg);

  // Get accumulation buffer from device communicator
  if (!((ncclSymkDevComm*)&handler.comm)->accumBuffer) {
    printf("ERROR: ReduceScatter_LLBuffer accumulation buffer not allocated!\n");
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

  // nElts is the number of elements per rank in the output
  // size_t nElts = nAllElts / nRanks;

  // FIXME: Will have a problem when it comes to data types that are not 4 bytes aligned
  constexpr int BytesPerPack = 8;
  constexpr int EltPerPack = BytesPerPack / sizeof(T);
  using Pack = BytePack<BytesPerPack>;
  using AccPack = BytePack<BytesPerPack*sizeof(Acc)/sizeof(T)>;

  // Calculate bytesPerCTA for the LL buffer
  // For reduce-scatter: each rank sends to all peers, so buffer needs nRanks slots per thread
  size_t bytesPerCtaPerEpoch = nRanks * blockDim.x * BytesPerPack;

  int roundRobinFactor = REDUCTION_BUFFER_SIZE / (bytesPerCtaPerEpoch * gridDim.x);
  if (Mode == ncclLL)
    roundRobinFactor >>= 1;

  if (roundRobinFactor < 1) {
    printf("[ERROR]: roundRobinFactor < 1 in ReduceScatter_LLBuffer\n");
    return;
  }

  roundRobinFactor = min(roundRobinFactor, (int)UINT8_MAX);

  // Calculate based on OUTPUT elements (nElts per rank), not total input
  // This ensures targetRank = i / nPacksPerRank is always < nRanks
  size_t nAllElts = nElts * nRanks;  // Total elements in input
  int nPacksPerRank = (nElts * sizeof(T) + BytesPerPack - 1) / BytesPerPack;
  int nTotalPacks = nPacksPerRank * nRanks;  // Total packs = exactly nPacksPerRank * nRanks
  uint32_t nIters = (nTotalPacks + nthreads - 1) / nthreads;
  // if (blockIdx.x == 0 && threadIdx.x == 0)
  //   printf("[DEBUG] Rank %d nAllElts: %d, nElts: %d, nPacksPerRank: %d, nTotalPacks: %d\n", rank, nAllElts, nElts, nPacksPerRank, nTotalPacks);
  // Create ncclLLBuffer for the intermediate reduction buffer
  // Multimem controls whether to use multicast for broadcast
  ncclLLBuffer<Mode, Multimem> llBuf(
    scratchSymPtr,
    /*bytesPerCtaPerEpoch=*/ bytesPerCtaPerEpoch,
    /*block=*/ blockIdx.x,
    /*roundRobinFactor=*/ (uint8_t)roundRobinFactor,
    /*mmHandle=*/ Multimem ? handler.comm.lsaMultimem : ncclMultimemHandle{}
  );

  // Input layout: [chunk_for_peer0][chunk_for_peer1]...[chunk_for_peer_{nRanks-1}]
  // Each chunk has nPacksPerRank packs
  // Output: we produce nPacksPerRank packs (reduced across all ranks)
  Pack* inputPacks = (Pack*)inputPtr;
  Pack* outputPacks = (Pack*)outputPtr;

  // Use blockDim.x for iteration size, not nthreads (global thread count)
  // This matches the buffer size calculation: bytesPerCtaPerEpoch = nRanks * blockDim.x * BytesPerPack
  int tn = blockDim.x;
  int t = threadIdx.x;

  // Divide output packs among CTAs: each CTA handles nPacksPerRank / gridDim.x packs
  // CTA b handles packs [b * packsPerCta, (b+1) * packsPerCta)
  int packsPerCta = (nPacksPerRank + gridDim.x - 1) / gridDim.x;
  int myStartPack = blockIdx.x * packsPerCta;
  int myEndPack = min(myStartPack + packsPerCta, nPacksPerRank);
  int myNumPacks = myEndPack - myStartPack;

  if (__builtin_expect(nIters > roundRobinFactor, false)) {
    // Synchronized version: process nIterPacks output packs per epoch
    // This ensures all ranks stay synchronized (like ReduceScatter_LL)
    int outputPacksProcessed = 0;

    #pragma unroll 1
    while (outputPacksProcessed < myNumPacks) {
      int nIterPacks = min(myNumPacks - outputPacksProcessed, tn);
      int tn_div_nPacks = tn / nIterPacks;
      int tn_mod_nPacks = tn % nIterPacks;
      int peer = t / nIterPacks;
      int pack = t % nIterPacks;

      // Absolute pack index in output
      int absPackBase = myStartPack + outputPacksProcessed;

      // Phase 1: All ranks send to ALL peers for this batch of output packs
      // Each thread sends its pack to multiple peers (or multiple threads send same pack)
      #pragma unroll 1
      for (int i = t; i < nRanks * nIterPacks; i += tn) {
        // Read from peer's chunk at position (absPackBase + pack)
        int inputOffset = peer * nPacksPerRank + absPackBase + pack;
        Pack myData = inputPacks[inputOffset];
        // Send to peer's buffer, slot = myRank * nIterPacks + pack
        int slot = rank * nIterPacks + pack;
        llBuf.template send<Pack>(team, peer, slot, myData);
        // Advance to next peer/pack
        peer += tn_div_nPacks;
        pack += tn_mod_nPacks;
        if (nIterPacks <= pack) { peer += 1; pack -= nIterPacks; }
      }

      // Phase 2: ALL ranks receive (threads t < nIterPacks)
      // Each receiving thread reduces contributions from all source ranks
      if (t < nIterPacks) {
        AccPack result = llBuf.template recvReduce<Unroll, Pack, /*Reset=*/true>(
          /*eltStart=*/ t,
          /*eltCount=*/ nRanks,
          /*eltStride=*/ nIterPacks,
          /*eltToAcc=*/ [&] __device__ (Pack x) -> AccPack { return applyCast<T, Acc>(x); },
          /*reduce=*/ [&] __device__ (AccPack a, AccPack b) -> AccPack { return applyReduce(red, a, b); }
        );
        // Store to output at absolute position (absPackBase + t)
        outputPacks[absPackBase + t] = applyCast<Acc, T>(result);
      }

      // CTA sync before advancing epoch (ensures all threads complete before next epoch)
      llBuf.advanceEpoch();

      outputPacksProcessed += nIterPacks;
    }
  } else {
    #pragma unroll 1
    for (int i = tid; i < nTotalPacks; i += nthreads) {
      int targetRank = i / nPacksPerRank;
      int packInRank = i % nPacksPerRank;  // Output pack index within targetRank's output
      int slotBase = packInRank % blockDim.x;  // Slot within this CTA's buffer region

      // Phase 1: All ranks send their contribution for this pack to targetRank
      Pack myData = loadPack<Pack>((Pack*)inputPtr, i, nTotalPacks);
      int slot = rank * blockDim.x + slotBase;
      llBuf.template send<Pack>(team, targetRank, slot, myData);

      // Phase 2: Only targetRank receives and reduces from all peers
      if (rank == targetRank) {
        AccPack result = llBuf.template recvReduce<Unroll, Pack, /*Reset=*/true>(
          /*eltStart=*/ slotBase,
          /*eltCount=*/ nRanks,
          /*eltStride=*/ blockDim.x,
          /*eltToAcc=*/ [&] __device__ (Pack x) -> AccPack { return applyCast<T, Acc>(x); },
          /*reduce=*/ [&] __device__ (AccPack a, AccPack b) -> AccPack { return applyReduce(red, a, b); }
        );
        storePack((Pack*) outputPtr, packInRank, nPacksPerRank, applyCast<Acc, T>(result));
      }

      llBuf.advanceEpoch();
    }
  }
}

// Public entry points used by the symmetric-kernel generator.
// Base versions (for non-power-of-2 ranks, uses Unroll=4)
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_LLBuffer(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_ReduceScatter_LLBuffer_impl<ncclPoison, /*Multimem=*/false, /*Unroll=*/4, Red, T>(args);
}

// Rank-specialized versions (Poison mode)
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_LLBuffer_R4(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_ReduceScatter_LLBuffer_impl<ncclPoison, /*Multimem=*/false, /*Unroll=*/4, Red, T>(args);
}
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_LLBuffer_R8(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_ReduceScatter_LLBuffer_impl<ncclPoison, /*Multimem=*/false, /*Unroll=*/8, Red, T>(args);
}
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_LLBuffer_R16(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_ReduceScatter_LLBuffer_impl<ncclPoison, /*Multimem=*/false, /*Unroll=*/16, Red, T>(args);
}
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_LLBuffer_R32(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_ReduceScatter_LLBuffer_impl<ncclPoison, /*Multimem=*/false, /*Unroll=*/32, Red, T>(args);
}

// LL16 sync mode versions
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_LLBuffer_LL16(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_ReduceScatter_LLBuffer_impl<ncclLL, /*Multimem=*/false, /*Unroll=*/4, Red, T>(args);
}
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_LLBuffer_LL16_R4(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_ReduceScatter_LLBuffer_impl<ncclLL, /*Multimem=*/false, /*Unroll=*/4, Red, T>(args);
}
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_LLBuffer_LL16_R8(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_ReduceScatter_LLBuffer_impl<ncclLL, /*Multimem=*/false, /*Unroll=*/8, Red, T>(args);
}
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_LLBuffer_LL16_R16(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_ReduceScatter_LLBuffer_impl<ncclLL, /*Multimem=*/false, /*Unroll=*/16, Red, T>(args);
}
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_LLBuffer_LL16_R32(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_ReduceScatter_LLBuffer_impl<ncclLL, /*Multimem=*/false, /*Unroll=*/32, Red, T>(args);
}

// Multimem version - uses multicast for broadcast
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_LLBufferMC(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_ReduceScatter_LLBuffer_impl<ncclPoison, /*Multimem=*/true, /*Unroll=*/4, Red, T>(args);
}
