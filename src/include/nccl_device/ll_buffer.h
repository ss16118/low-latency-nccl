/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_LL_BUFFER_H_
#define _NCCL_DEVICE_LL_BUFFER_H_

#include "core.h"  // For NCCL_EXTERN_C, NCCL_HOST_DEVICE_INLINE, ncclDataType_t, ncclResult_t

// ==================== Low-Latency Sync Mode ====================
//
// The synchronization mode is specified as a template parameter when
// creating an ncclLLBuffer. This enables compile-time optimization.
//
// Three modes:
//   - ncclLL: 8 bytes data + 8 bytes flags per 16-byte slot
//             Layout: {data[0], flag, data[1], flag} as uint4
//   - ncclLL128: 120 bytes data + 8 bytes flags per 128-byte line
//                Every 8th thread is a flag carrier (TODO)
//   - ncclPoison: 16 bytes data per 16-byte slot, no flags
//                 Poll until non-poison value appears
//                 User MUST poison buffer before receiving
//
enum ncclLLSyncMode {
  ncclLL      = 0,  // 8+8 flag format (8 bytes data, 8 bytes flags)
  ncclLL128   = 1,  // 120+8 flag format (NOT YET IMPLEMENTED)
  ncclPoison  = 2   // Poison-based synchronization
};

typedef enum ncclLLSyncMode ncclLLSyncMode_t;

// ==================== Host-Side API ====================

// Calculate minimum buffer size for a given element count and sync mode.
// Assuming there is no double-buffering. The user can multiply this
// value by the number of buffers they want to use to achieve double-buffering or more.
// In addition, this calculates the buffer size per CTA,
// so the user also needs to multiply this value by the number of CTAs they plan to use.
// nElts: number of elements to communicate
// eltSize: size of each element in bytes
// mode: synchronization mode (affects slot layout)
// Returns: bytes needed
//
// Slot sizes:
//   ncclLL:     16 bytes per slot (8 data + 8 flags)
//   ncclLL128: 128 bytes per slot (120 data + 8 flags)
//   ncclPoison: 16 bytes per slot (16 data, no flags)
//
NCCL_EXTERN_C __host__ size_t ncclCalcMinReductionBufferSize(ncclLLSyncMode_t mode, int nElts, int eltSize, int nRanks, int nBlocks, int roundRobinFactor);

// Get the 32-bit poison value for a given NCCL data type.
// Returns the value that should be used to fill buffer for poison-based sync.
NCCL_EXTERN_C __host__ uint32_t ncclLLGetPoisonValue(ncclDataType_t dtype);

// Initialize a buffer with poison values for a given NCCL data type.
// This prepares a buffer for ncclPoison mode.
//
// Parameters:
//   buffer: device pointer to the buffer to poison
//   sizeBytes: size of the buffer in bytes (must be a multiple of 4)
//   dtype: NCCL data type (determines the poison value to use)
//   stream: CUDA stream for async execution
//
// Synchronous version (uses default stream and synchronizes)
NCCL_EXTERN_C __host__ ncclResult_t ncclLLPoisonBuffer(
    void* buffer,
    size_t sizeBytes,
    ncclDataType_t dtype
);

// Async version with explicit stream control
NCCL_EXTERN_C __host__ ncclResult_t ncclLLPoisonBufferAsync(
    void* buffer,
    size_t sizeBytes,
    ncclDataType_t dtype,
    cudaStream_t stream
);

// ==================== Device-Side API ====================

#if __CUDACC__

#include "impl/core__types.h"
#include "impl/ll_buffer__types.h"
#include "ptr.h"

template<ncclLLSyncMode Mode, bool Multimem>
struct ncclLLBuffer_internal;

// ncclLLBuffer: Low-latency buffer wrapper with compile-time sync mode
//
// Template parameters:
//   Mode: Synchronization mode (ncclLL, ncclLL128, ncclPoison)
//   Multimem: Whether to use multimem for broadcast (compile-time for optimization)
//
// The user allocates symmetric memory, passes it to the kernel, and wraps it
// in ncclLLBuffer to perform low-latency send/recv operations.
//
// Key features:
// - Sync mode and multimem are template parameters for compile-time optimization
// - Flag value (uint32_t) maintained internally for LL/LL128 modes
// - For ncclPoison: user must poison buffer before receiving
// - Buffer positions are controlled by the user (no internal slot tracking)
// - Optimized for register residency (~44 bytes internal state)
//
// Usage:
//   // Host: allocate symmetric memory via ncclMemAlloc
//   // Kernel: wrap the buffer
//   ncclTeam team = ncclTeamLsa(comm);
//   ncclLLBuffer<ncclPoison, false> buf(team, symBuf, maxSlots, block, roundRobinFactor, mmHandle);
//   buf.bcast(team, slot, myValue);   // broadcast to all peers
//   auto val = buf.recv<float>(slot); // recv from local buffer
//
template<ncclLLSyncMode Mode, bool Multimem = false>
struct ncclLLBuffer : ncclLLBuffer_internal<Mode, Multimem> {
  // Constructor
  //
  // Parameters:
  //   buf: Symmetric pointer to user-allocated buffer
  //   bytesPerCtaPerEpoch: Maximum number of bytes per rank per epoch
  //   block: Block index (e.g., blockIdx.x) for multi-CTA buffer offset calculation
  //   roundRobinFactor: Number of buffers to use for multiple buffering
  //   mmHandle: Multimem handle for broadcast (only used if Multimem=true)
  //
  NCCL_DEVICE_INLINE ncclLLBuffer(
    ncclSymPtr<char> buf,
    int bytesPerCtaPerEpoch,
    int block,
    uint8_t roundRobinFactor,
    ncclMultimemHandle mmHandle
  );

  NCCL_DEVICE_INLINE ~ncclLLBuffer();

  ncclLLBuffer(ncclLLBuffer const&) = delete;

  // ==================== Configuration Queries ====================
  NCCL_DEVICE_INLINE uint32_t currentEpoch() const;  // For LL/LL128 modes
  NCCL_DEVICE_INLINE constexpr bool hasMultimem() const { return Multimem; }
  // Set the epoch value, handling overflow by wrapping to 2 (skip 0 and 1)
  NCCL_DEVICE_INLINE void setEpoch(uint8_t epoch) {
    // Epochs 0 and 1 are reserved (uninitialized/initial state), wrap to 2
    this->epoch = (epoch < 2) ? 2 : epoch;
    this->currentBufferOffset = this->calcBufferOffset();
  }

  // ==================== Buffer Access ====================
  // Get local buffer pointer (for receiving)
  NCCL_DEVICE_INLINE void* localPtr() const;

  // Get peer buffer pointer (for sending)
  // team: Team for peer addressing (passed by caller to avoid storing in struct)
  NCCL_DEVICE_INLINE void* peerPtr(ncclTeam const& team, int peer) const;

  // Get multimem pointer (for broadcast) - nullptr if not available
  NCCL_DEVICE_INLINE void* multimemPtr() const;

  // Get the underlying ncclSymPtr
  NCCL_DEVICE_INLINE ncclSymPtr<char> symPtr() const;

  // ==================== Send Operations ====================
  // Send to peer's buffer at specified slot offset
  // For ncclLL: data is packed with flags
  // For ncclPoison: data is stored directly
  // team: Team for peer addressing
  // peer: destination rank
  // elt: element offset in the buffer
  template<typename T>
  NCCL_DEVICE_INLINE void send(ncclTeam const& team, int peer, int elt, T const& data) const;

  // ==================== Broadcast Operations ====================
  // Broadcast to all peers at specified slot offset
  // Uses multimem if available, otherwise loops over peers
  // team: Team for peer addressing
  // elt: element offset in the buffer
  // Unroll: Number of peers to unroll in the inner loop (default 8 for optimal ILP)
  template<int Unroll = 8, typename T>
  NCCL_DEVICE_INLINE void bcast(ncclTeam const& team, int elt, T const& data) const;

  // ==================== Receive Operations ====================
  // Receive from peer's buffer at specified slot offset
  // For ncclLL: polls until flags match
  // For ncclPoison: polls until value differs from poison
  // Uses internal buf for peer addressing
  // peer: source rank
  // slot: slot offset in the buffer
  // Reset: whether to reset the slot after receiving (template parameter for compile-time optimization)
  template<typename T, bool Reset = true>
  NCCL_DEVICE_INLINE T recv(int elt) const;

  // Receive multiple slots with unrolling
  // eltStart: starting element index
  // eltCount: number of elements to receive
  // eltStride: stride between elements
  // elts: array to store the received elements
  // Reset: whether to reset the slots after receiving (template parameter for compile-time optimization)
  template<int MinEltCount, int MaxEltCount, typename T, bool Reset = true>
  NCCL_DEVICE_INLINE void recvUnrolled(
      int eltStart, int eltCount, int eltStride, T (&elts)[MaxEltCount]
  ) const;

  // Receive and reduce from peer's buffer with accumulator type support.
  // Wraps recvUnrolled and reduces the results.
  // Unroll: number of slots to receive and reduce at once
  // T: element type
  // EltToAcc: function to convert the element to the accumulator type
  // Reduce: function to reduce the elements
  // eltStart: starting element index
  // eltCount: number of elements to receive
  // eltStride: stride between elements
  // Reset: whether to reset the slots after receiving (template parameter for compile-time optimization)
  template<int Unroll, typename T, bool Reset = true, typename EltToAcc, typename Reduce>
  NCCL_DEVICE_INLINE auto recvReduce(
      int slotStart, int slotCount, int slotStride,
      EltToAcc eltToAcc, Reduce reduce
    ) const -> decltype(eltToAcc(nccl::utility::declval<T>()));


  // ==================== Reset Operations ====================
  // Reset slots based on sync mode:
  // - LL/LL128: clear to zero
  // - Poison: set to poison value using size-appropriate instruction

  template<typename T>
  NCCL_DEVICE_INLINE void reset(int elt) const;

  template<typename T>
  NCCL_DEVICE_INLINE void resetRange(int eltStart, int eltCount, int eltStride) const;

  // ==================== Epoch Flag Management ====================
  // Increment epoch flag after completing a communication round
  NCCL_DEVICE_INLINE void advanceEpoch();

  // Reset epoch flag to 2
  NCCL_DEVICE_INLINE void resetEpoch();
};

#endif // __CUDACC__

#endif // _NCCL_DEVICE_LL_BUFFER_H_
