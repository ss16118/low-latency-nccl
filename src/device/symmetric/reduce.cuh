#include "sym_kernels.h"
#include "kernel.cuh"
#include "primitives.cuh"
#include "nccl_device/ll_buffer.h"

/**
 * Reduce kernel using ncclLLBuffer API.
 *
 * This kernel implements a reduce operation (many-to-one) using the ncclLLBuffer abstraction:
 *   1. All ranks send their input data to the root rank
 *   2. Only the root rank receives from all peers, reduces, and stores to output
 *
 * Memory layout for reduce:
 *   - Input: Each rank has nElts elements
 *   - Output: Only root rank gets nElts elements (reduced result)
 *   - Scratch buffer: slot = srcRank * blockDim.x + (packIdx % blockDim.x)
 *
 * Key design points:
 *   - Uses ncclPoison or LL16 sync mode for low-latency synchronization
 *   - Each thread handles one pack per iteration
 *   - All ranks participate in sending; only root rank receives/reduces/stores
 *   - Slot layout ensures each source rank's data is at a unique offset
 */
template<ncclLLSyncMode Mode, int Unroll, template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_Reduce_LLBuffer_impl(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};

  ncclLsaBarrierSession<ncclCoopCta> bar{
    ncclCoopCta(), handler.comm, ncclTeamTagLsa(), blockIdx.x, /*multimem=*/false
  };

  struct ncclSymkDevWork const& dw = handler.devWork[0];
  size_t nElts = dw.nElts;
  int const root = dw.rootRank;
  int const& rank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;
  using Acc = typename ncclSymkAccumType<Red, T, /*nvls=*/false>::Type;
  Red<Acc> red(handler.devWork->redOpArg);

  // Get accumulation buffer from device communicator
  if (!((ncclSymkDevComm*)&handler.comm)->accumBuffer) {
    printf("ERROR: Reduce_LLBuffer accumulation buffer not allocated!\n");
    return;
  }

  ncclTeam team = ncclTeamLsa(handler.comm);

  // Create ncclSymPtr from the allocated accumulation buffer
  ncclSymPtr<char> scratchSymPtr;
  scratchSymPtr.offset = ((ncclSymkDevComm*)&handler.comm)->lamportAccumOffset;
  scratchSymPtr.window = ((ncclSymkDevComm*)&handler.comm)->accumBuffer;

  int currentSlot = (int) (scratchSymPtr.offset / REDUCTION_BUFFER_SIZE);

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

  ncclSymPtr<T> input(dw.inputWin, dw.inputOff);
  ncclSymPtr<T> output(dw.outputWin, dw.outputOff);
  T* inputPtr = (T*)input.localPtr();
  T* outputPtr = (T*)output.localPtr();

  constexpr int BytesPerPack = 8;
  constexpr int EltPerPack = BytesPerPack / sizeof(T);
  using Pack = BytePack<BytesPerPack>;
  using AccPack = BytePack<BytesPerPack*sizeof(Acc)/sizeof(T)>;

  constexpr int nEltsPerPack = BytesPerPack / sizeof(T);

  // Check for the size of the scratch buffer to make sure it is large enough to hold all the data
  size_t maxSize = REDUCTION_BUFFER_SIZE;

  #if __cpp_if_constexpr
  if constexpr (Mode == ncclLL)
    maxSize >>= 1;
  #else
  if (Mode == ncclLL)
    maxSize >>= 1;
  #endif

  if (rank == root && maxSize < nElts * nRanks * sizeof(T)) {
    printf("[ERROR]: The scratch buffer is too small to hold all the data in Reduce_LLBuffer (nElts: %ld, nRanks: %d, maxSize: %ld)\n", nElts, nRanks, maxSize);
    return;
  }

  // Initialize the ncclLLBuffer for the intermediate buffer
  // Note that multimem is not used because it reduce operation does not need to broadcast to all ranks
  ncclLLBuffer<Mode, false> llBuf(
    scratchSymPtr,
    /*bytesPerCtaPerEpoch=*/ 0,
    /*block=*/ blockIdx.x,
    /*roundRobinFactor=*/ 0,
    /*mmHandle=*/ ncclMultimemHandle{}
  );

  int nPacks = (nElts + EltPerPack - 1) / EltPerPack;

  for (int i = tid; i < nPacks; i += nthreads) {
    Pack myData = loadPack<Pack>((T*)inputPtr, i * nEltsPerPack, nElts);
    int slot = rank * nPacks + i;
    llBuf.template send<Pack>(team, root, slot, myData);
  }
  
  if (rank == root) {
    for (int i = tid; i < nPacks; i += nthreads) {
      AccPack result = llBuf.template recvReduce<Unroll, Pack, /*Reset=*/true>(
        /*eltStart=*/ i,
        /*eltCount=*/ nRanks,
        /*eltStride=*/ nPacks,
        /*eltToAcc=*/ [&] __device__ (Pack x) -> AccPack { return applyCast<T, Acc>(x); },
        /*reduce=*/ [&] __device__ (AccPack a, AccPack b) -> AccPack { return applyReduce(red, a, b); }
      );
      storePack<Pack>((T*)outputPtr, i * nEltsPerPack, nElts, applyCast<Acc, T>(result));
    }
  }

  if (currentSlot == ncclSymkLamportAccumSlots - 1)
    bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);
}

// Public entry points used by the symmetric-kernel generator.
// Base versions (for non-power-of-2 ranks, uses Unroll=4)
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_Reduce_LLBuffer(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_Reduce_LLBuffer_impl<ncclPoison, /*Unroll=*/4, Red, T>(args);
}

// Rank-specialized versions (Poison mode)
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_Reduce_LLBuffer_R4(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_Reduce_LLBuffer_impl<ncclPoison, /*Unroll=*/4, Red, T>(args);
}
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_Reduce_LLBuffer_R8(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_Reduce_LLBuffer_impl<ncclPoison, /*Unroll=*/8, Red, T>(args);
}

// LL16 sync mode versions
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_Reduce_LLBuffer_LL16(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_Reduce_LLBuffer_impl<ncclLL, /*Unroll=*/4, Red, T>(args);
}
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_Reduce_LLBuffer_LL16_R4(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_Reduce_LLBuffer_impl<ncclLL, /*Unroll=*/4, Red, T>(args);
}
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_Reduce_LLBuffer_LL16_R8(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_Reduce_LLBuffer_impl<ncclLL, /*Unroll=*/8, Red, T>(args);
}
