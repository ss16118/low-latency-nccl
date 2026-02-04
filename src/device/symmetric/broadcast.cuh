#include "sym_kernels.h"
#include "kernel.cuh"
#include "primitives.cuh"
#include "nccl_device/ll_buffer.h"

/**
 * Broadcast kernel using ncclLLBuffer API.
 *
 * This kernel implements a broadcast operation (one-to-many) using the ncclLLBuffer abstraction:
 *   1. Root rank broadcasts its input data to all peers
 *   2. All ranks receive from root and store to output
 *
 * Memory layout for broadcast:
 *   - Input: Only root rank has meaningful input (nElts elements)
 *   - Output: All ranks get nElts elements (copy of root's data)
 *   - Scratch buffer: slot = packIdx % blockDim.x (only root writes)
 *
 * Key design points:
 *   - Uses ncclPoison or LL16 sync mode for low-latency synchronization
 *   - Each thread handles one pack per iteration
 *   - Only root rank broadcasts; all ranks receive
 *   - Slot layout: root writes to slot = threadIdx.x for each pack
 */
template<ncclLLSyncMode Mode, bool Multimem, int Unroll>
__device__ __forceinline__ void ncclSymkRun_Broadcast_LLBuffer_impl(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};
  ncclLsaBarrierSession<ncclCoopCta> bar{
    ncclCoopCta(), handler.comm, ncclTeamTagLsa(), blockIdx.x, /*multimem=*/Multimem
  };
  struct ncclSymkDevWork const& dw = handler.devWork[0];
  size_t nElts = dw.nElts;
  int const root = dw.rootRank;
  int const& rank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;

  // Get accumulation buffer from device communicator
  if (!((ncclSymkDevComm*)&handler.comm)->accumBuffer) {
    printf("ERROR: Broadcast_LLBuffer accumulation buffer not allocated!\n");
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

  ncclSymPtr<char> input(dw.inputWin, dw.inputOff);
  ncclSymPtr<char> output(dw.outputWin, dw.outputOff);
  char* inputPtr = input.localPtr();
  char* outputPtr = output.localPtr();

  constexpr int BytesPerPack = 8;
  using Pack = BytePack<BytesPerPack>;

  // Need to make sure the scratch buffer is large enough to hold the data
  // For broadcast: root sends one slot per thread (no need for nRanks slots since only root sends)
  size_t maxSize = REDUCTION_BUFFER_SIZE;

  #if __cpp_if_constexpr
  if constexpr (Mode == ncclLL)
    maxSize >>= 1;
  #else
  if (Mode == ncclLL)
    maxSize >>= 1;
  #endif

  if (nElts > maxSize) {
    printf("[ERROR]: The scratch buffer is too small to hold all the data in Broadcast_LLBuffer (nElts: %d, maxSize: %d)\n", nElts, maxSize);
    return;
  }

  
  int nPacks = (nElts + BytesPerPack - 1) / BytesPerPack;

  // Create ncclLLBuffer for the intermediate buffer
  ncclLLBuffer<Mode, Multimem> llBuf(
    scratchSymPtr,
    /*bytesPerCtaPerEpoch=*/ 0,
    /*block=*/ 0,
    /*roundRobinFactor=*/ 0,
    /*mmHandle=*/ Multimem ? handler.comm.lsaMultimem : ncclMultimemHandle{}
  );

  if (rank == root) {
    for (int i = tid; i < nPacks; i += nthreads) {
      Pack myData = loadPack<Pack>((Pack*)inputPtr, i, nPacks);
      llBuf.template bcast<Unroll, Pack>(team, i, myData);
    }
  }
  for (int i = tid; i < nPacks; i += nthreads) {
    Pack got = llBuf.template recv<Pack, /*Reset=*/true>(i);
    storePack<Pack>((Pack*)outputPtr, i, nPacks, got);
  }

  if (currentSlot == ncclSymkLamportAccumSlots - 1)
    bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

}

// Public entry points used by the symmetric-kernel generator.
// Base versions (for non-power-of-2 ranks, uses Unroll=4)
__device__ __forceinline__ void ncclSymkRun_Broadcast_LLBuffer(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_Broadcast_LLBuffer_impl<ncclPoison, /*Multimem=*/false, /*Unroll=*/4>(args);
}

// Rank-specialized versions (Poison mode)
__device__ __forceinline__ void ncclSymkRun_Broadcast_LLBuffer_R4(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_Broadcast_LLBuffer_impl<ncclPoison, /*Multimem=*/false, /*Unroll=*/4>(args);
}
__device__ __forceinline__ void ncclSymkRun_Broadcast_LLBuffer_R8(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_Broadcast_LLBuffer_impl<ncclPoison, /*Multimem=*/false, /*Unroll=*/8>(args);
}

// LL16 sync mode versions
__device__ __forceinline__ void ncclSymkRun_Broadcast_LLBuffer_LL16(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_Broadcast_LLBuffer_impl<ncclLL, /*Multimem=*/false, /*Unroll=*/4>(args);
}
__device__ __forceinline__ void ncclSymkRun_Broadcast_LLBuffer_LL16_R4(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_Broadcast_LLBuffer_impl<ncclLL, /*Multimem=*/false, /*Unroll=*/4>(args);
}
__device__ __forceinline__ void ncclSymkRun_Broadcast_LLBuffer_LL16_R8(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_Broadcast_LLBuffer_impl<ncclLL, /*Multimem=*/false, /*Unroll=*/8>(args);
}

// Multimem version - uses multicast for broadcast
__device__ __forceinline__ void ncclSymkRun_Broadcast_LLBufferMC(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_Broadcast_LLBuffer_impl<ncclPoison, /*Multimem=*/true, /*Unroll=*/4>(args);
}
