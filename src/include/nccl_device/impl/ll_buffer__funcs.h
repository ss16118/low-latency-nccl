/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_LL_BUFFER__FUNCS_H_
#define _NCCL_DEVICE_LL_BUFFER__FUNCS_H_
#include "ll_buffer__types.h"
#include "comm__types.h"
#include "../utility.h"
#include "ptr__funcs.h"

#if __CUDACC__

template <typename T>
constexpr bool is_fp8 = std::is_same<T, __nv_fp8_e4m3>::value || std::is_same<T, __nv_fp8_e5m2>::value;

// ==================== Low-level Load/Store Helpers ====================

// ---8-bit operations ---
NCCL_DEVICE_INLINE uint8_t ncclLLBufferLoad8(void* ptr) {
  unsigned int tmp;
  asm volatile("ld.volatile.global.u8 %0, [%1];"
    : "=r"(tmp) : "l"(ptr));
  return static_cast<uint8_t>(tmp);
}

NCCL_DEVICE_INLINE void ncclLLBufferStore8(void* ptr, uint8_t val) {
  unsigned int tmp = static_cast<unsigned int>(val);
  asm volatile("st.volatile.global.u8 [%0], %1;"
    :: "l"(ptr), "r"(tmp) : "memory");
}


// --- 16-bit operations (for half precision) ---
NCCL_DEVICE_INLINE uint16_t ncclLLBufferLoad16(void* ptr) {
  uint16_t val;
  asm volatile("ld.volatile.global.u16 %0, [%1];"
    : "=h"(val) : "l"(ptr));
  return val;
}

NCCL_DEVICE_INLINE void ncclLLBufferStore16(void* ptr, uint16_t val) {
  asm volatile("st.volatile.global.u16 [%0], %1;"
    :: "l"(ptr), "h"(val) : "memory");
}

// Note: multimem.st does not support 16-bit, use 32-bit with padding
NCCL_DEVICE_INLINE void ncclLLBufferStoreMultimem16(void* ptr, uint16_t val) {
  // Pack into 32-bit and use 32-bit multimem store
  uint32_t val32 = val;
  asm volatile("multimem.st.global.u32 [%0], %1;"
    :: "l"(ptr), "r"(val32) : "memory");
}

// --- 32-bit (v1) operations ---
NCCL_DEVICE_INLINE uint32_t ncclLLBufferLoad32(void* ptr) {
  uint32_t val;
  asm volatile("ld.volatile.global.u32 %0, [%1];"
    : "=r"(val) : "l"(ptr));
  return val;
}

NCCL_DEVICE_INLINE void ncclLLBufferStore32(void* ptr, uint32_t val) {
  asm volatile("st.volatile.global.u32 [%0], %1;"
    :: "l"(ptr), "r"(val) : "memory");
}

NCCL_DEVICE_INLINE void ncclLLBufferStoreMultimem32(void* ptr, uint32_t val) {
  asm volatile("multimem.st.global.u32 [%0], %1;"
    :: "l"(ptr), "r"(val) : "memory");
}

// --- 64-bit (v2) operations ---
NCCL_DEVICE_INLINE uint2 ncclLLBufferLoad64(void* ptr) {
  uint2 val;
  asm volatile("ld.volatile.global.v2.u32 {%0,%1}, [%2];"
    : "=r"(val.x), "=r"(val.y) : "l"(ptr));
  return val;
}

NCCL_DEVICE_INLINE void ncclLLBufferStore64(void* ptr, uint2 val) {
  asm volatile("st.volatile.global.v2.u32 [%0], {%1,%2};"
    :: "l"(ptr), "r"(val.x), "r"(val.y) : "memory");
}

NCCL_DEVICE_INLINE void ncclLLBufferStoreMultimem64(void* ptr, uint2 val) {
  asm volatile("multimem.st.global.v2.f32 [%0], {%1,%2};"
    :: "l"(ptr), "r"(val.x), "r"(val.y) : "memory");
}

// --- 128-bit (v4) operations ---
NCCL_DEVICE_INLINE uint4 ncclLLBufferLoad128(void* ptr) {
  uint4 val;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
    : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
    : "l"(ptr));
  return val;
}

NCCL_DEVICE_INLINE void ncclLLBufferStore128(void* ptr, uint4 val) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};"
    :: "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w) : "memory");
}

NCCL_DEVICE_INLINE void ncclLLBufferStoreMultimem128(void* ptr, uint4 val) {
  asm volatile("multimem.st.global.v4.f32 [%0], {%1,%2,%3,%4};"
    :: "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w) : "memory");
}

// ==================== Load/Store Operations for Poison Mode ====================

// --- 8-bit Load ---
template<typename T>
NCCL_DEVICE_INLINE T ncclLLBufferLoadImpl(void* ptr, ncclLLOp8) {
  union { T val; uint8_t u8; };
  u8 = ncclLLBufferLoad8(ptr);
  return val;
}


// --- 16-bit Load ---
template<typename T>
NCCL_DEVICE_INLINE T ncclLLBufferLoadImpl(void* ptr, ncclLLOp16) {
  union { T val; uint16_t u16; };
  u16 = ncclLLBufferLoad16(ptr);
  return val;
}


// --- 32-bit Load ---
template<typename T>
NCCL_DEVICE_INLINE T ncclLLBufferLoadImpl(void* ptr, ncclLLOp32) {
  union { T val; uint32_t u32; };
  u32 = ncclLLBufferLoad32(ptr);
  return val;
}

// --- 64-bit Load ---
template<typename T>
NCCL_DEVICE_INLINE T ncclLLBufferLoadImpl(void* ptr, ncclLLOp64) {
  union { T val; uint2 u64; };
  u64 = ncclLLBufferLoad64(ptr);
  return val;
}

// --- 128-bit Load ---
template<typename T>
NCCL_DEVICE_INLINE T ncclLLBufferLoadImpl(void* ptr, ncclLLOp128) {
  union { T val; uint4 u128; };
  u128 = ncclLLBufferLoad128(ptr);
  return val;
}


// --- 8-bit Store ---
template<bool Multimem, typename T>
NCCL_DEVICE_INLINE void ncclLLBufferStoreImpl(void* ptr, T const& data, ncclLLOp8) {
  union { T val; uint8_t u8; };
  val = data;
  #if __cpp_if_constexpr
  if constexpr (Multimem) {
  #else
  if (Multimem) {
  #endif
    printf("Error: Multimem store is not supported for 1-byte types. Use ncclLLBufferStorePoisonSync instead.\n");
    return;
  } else {
    ncclLLBufferStore8(ptr, u8);
  }
}

// --- 16-bit Store ---
template<bool Multimem, typename T>
NCCL_DEVICE_INLINE void ncclLLBufferStoreImpl(void* ptr, T const& data, ncclLLOp16) {
  union { T val; uint16_t u16; };
  val = data;
  #if __cpp_if_constexpr
  if constexpr (Multimem) {
  #else
  if (Multimem) {
  #endif
    printf("Error: Multimem store is not supported for 2-byte types. Use ncclLLBufferStorePoisonSync instead.\n");
    return;
  } else {
    ncclLLBufferStore16(ptr, u16);
  }
}

// --- 32-bit Store ---
template<bool Multimem, typename T>
NCCL_DEVICE_INLINE void ncclLLBufferStoreImpl(void* ptr, T const& data, ncclLLOp32) {
  union { T val; uint32_t u32; };
  val = data;
  #if __cpp_if_constexpr
  if constexpr (Multimem) {
  #else
  if (Multimem) {
  #endif
    ncclLLBufferStoreMultimem32(ptr, u32);
  } else {
    ncclLLBufferStore32(ptr, u32);
  }
}

// --- 64-bit Store ---
template<bool Multimem, typename T>
NCCL_DEVICE_INLINE void ncclLLBufferStoreImpl(void* ptr, T const& data, ncclLLOp64) {
  union { T val; uint2 u64; };
  val = data;
  #if __cpp_if_constexpr
  if constexpr (Multimem) {
  #else
  if (Multimem) {
  #endif
    ncclLLBufferStoreMultimem64(ptr, u64);
  } else {
    ncclLLBufferStore64(ptr, u64);
  }
}

// --- 128-bit Store ---
template<bool Multimem, typename T>
NCCL_DEVICE_INLINE void ncclLLBufferStoreImpl(void* ptr, T const& data, ncclLLOp128) {
  union { T val; uint4 u128; };
  val = data;
  #if __cpp_if_constexpr
  if constexpr (Multimem) {
  #else
  if (Multimem) {
  #endif
    ncclLLBufferStoreMultimem128(ptr, u128);
  } else {
    ncclLLBufferStore128(ptr, u128);
  }
}

/************** LL mode store operations **************/

template<bool Multimem, typename T>
NCCL_DEVICE_INLINE void ncclLLBufferStoreLLSyncImpl(void* ptr, T const& data, int pitch, uint8_t epoch, ncclLLOp8) {
  // If the type is 1 byte, we are effectively sending out a single flag of 1 byte plus the data of 1 byte.
  // FIXME: It might cause a problem if the epoch value is larger than 255.
  union { uint16_t payload; uint8_t payloadU8[2]; };
  payloadU8[0] = epoch;
  union { T tmp; uint8_t u8; };
  tmp = data;
  payloadU8[1] = u8;
  #if __cpp_if_constexpr
  if constexpr (Multimem) {
  #else
  if (Multimem) {
  #endif
    ncclLLBufferStoreMultimem16(ptr, payload);
  } else {
    ncclLLBufferStore16(ptr, payload);
  }
}

template<bool Multimem, typename T>
NCCL_DEVICE_INLINE void ncclLLBufferStoreLLSyncImpl(void* ptr, T const& data, int pitch, uint8_t epoch, ncclLLOp16) {
  // If the type is 2 bytes, we are sending out a single flag of 2 bytes plus the data of 2 bytes.
  union { uint32_t payload; uint16_t payloadU16[2]; };
  payloadU16[0] = (uint16_t) epoch;
  union { T tmp; uint16_t u16; };
  tmp = data;
  payloadU16[1] = u16;
  #if __cpp_if_constexpr
  if constexpr (Multimem) {
  #else
  if (Multimem) {
  #endif
    ncclLLBufferStoreMultimem32(ptr, payload);
  } else {
    ncclLLBufferStore32(ptr, payload);
  }
}

template<bool Multimem, typename T>
NCCL_DEVICE_INLINE void ncclLLBufferStoreLLSyncImpl(void* ptr, T const& data, int pitch, uint8_t epoch, ncclLLOp32) {
  // If the type is 4 bytes, we are sending out a single flag of 4 bytes plus the data of 4 bytes.
  uint2 payload;
  payload.x = (uint32_t) epoch;
  union { T tmp; uint32_t u32; };
  tmp = data;
  payload.y = u32;
  #if __cpp_if_constexpr
  if constexpr (Multimem) {
  #else
  if (Multimem) {
  #endif
    ncclLLBufferStoreMultimem64(ptr, payload);
  } else {
    ncclLLBufferStore64(ptr, payload);
  }
}


template<bool Multimem, typename T>
NCCL_DEVICE_INLINE void ncclLLBufferStoreLLSyncImpl(void* ptr, T const& data, int pitch, uint8_t epoch, ncclLLOp64) {
  // If the type is 8 bytes
  uint4 payload;
  payload.x = (uint32_t) epoch;
  payload.z = (uint32_t) epoch;
  union { T tmp; uint32_t u32[2]; };
  tmp = data;
  payload.y = u32[0];
  payload.w = u32[1];
  #if __cpp_if_constexpr
  if constexpr (Multimem) {
  #else
  if (Multimem) {
  #endif
    ncclLLBufferStoreMultimem128(ptr, payload);
  } else {
    ncclLLBufferStore128(ptr, payload);
  }
}

template<bool Multimem, typename T>
NCCL_DEVICE_INLINE void ncclLLBufferStoreLLSyncImpl(void* ptr, T const& data, int pitch, uint8_t epoch, ncclLLOp128) {
  // If the type is 16 bytes
  union { T tmp; uint32_t u32[2][2]; };
  tmp = data;

  // Send out the data in 2 parts
  #pragma unroll
  for (int u = 0; u < 2; ++u) {
    uint4 payload;
    payload.x = (uint32_t) epoch;
    payload.z = (uint32_t) epoch;
    payload.y = u32[u][0];
    payload.w = u32[u][1];
    #if __cpp_if_constexpr
    if constexpr (Multimem) {
    #else
    if (Multimem) {
    #endif
      ncclLLBufferStoreMultimem128(ptr + u * pitch / 2, payload);
    } else {
      ncclLLBufferStore128(ptr + u * pitch / 2, payload);
    }
  }
}

// ==================== Public Trait-dispatched Interface ====================
// These use ncclLLMemOpTraits to select the appropriate implementation

/**
 * Load the value from the given pointer in Poison mode, it is
 * simply a wrapper around ncclLLBufferLoadImpl
 */
template<typename T>
NCCL_DEVICE_INLINE T ncclLLBufferLoadPoisonSync(void* ptr) {
  using Traits = ncclLLBufferMemOpTraits<T>;
  return ncclLLBufferLoadImpl<T>(ptr, typename Traits::OpTag{});
}


/************** LL mode unpacked load operations (out-parameters for better perf) **************/

template<typename T>
NCCL_DEVICE_INLINE void ncclLLBufferLoadLLSyncUnpackedImpl(void* ptr, int pitch, T& outVal, uint32_t& outFlag, ncclLLOp8) {
  union { uint16_t tmp; uint8_t u8[2]; };
  tmp = ncclLLBufferLoad16(ptr);
  outFlag = (uint32_t) u8[0];
  union { T val; uint8_t u8_val; };
  u8_val = u8[1];
  outVal = val;
}

template<typename T>
NCCL_DEVICE_INLINE void ncclLLBufferLoadLLSyncUnpackedImpl(void* ptr, int pitch, T& outVal, uint32_t& outFlag, ncclLLOp16) {
  union { uint32_t tmp; uint16_t u16[2]; };
  tmp = ncclLLBufferLoad32(ptr);
  outFlag = (uint32_t) u16[0];
  union { T val; uint16_t u16_val; };
  u16_val = u16[1];
  outVal = val;
}

template<typename T>
NCCL_DEVICE_INLINE void ncclLLBufferLoadLLSyncUnpackedImpl(void* ptr, int pitch, T& outVal, uint32_t& outFlag, ncclLLOp32) {
  uint2 tmp = ncclLLBufferLoad64(ptr);
  outFlag = (uint32_t) tmp.x;
  union { T val; uint32_t u32; };
  u32 = tmp.y;
  outVal = val;
}

template<typename T>
NCCL_DEVICE_INLINE void ncclLLBufferLoadLLSyncUnpackedImpl(void* ptr, int pitch, T& outVal, uint32_t& outFlag, ncclLLOp64) {
  uint4 tmp = ncclLLBufferLoad128(ptr);
  outFlag = (uint32_t) tmp.x;
  union { T val; uint32_t u32[2]; };
  u32[0] = tmp.y;
  u32[1] = tmp.w;
  outVal = val;
}

template<typename T>
NCCL_DEVICE_INLINE void ncclLLBufferLoadLLSyncUnpackedImpl(void* ptr, int pitch, T& outVal, uint32_t& outFlag, ncclLLOp128) {
  union { T val; uint32_t u32[2][2]; };
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    uint4 tmp = ncclLLBufferLoad128((char*)ptr + i * pitch / 2);
    outFlag = (uint32_t) tmp.x;
    u32[i][0] = tmp.y;
    u32[i][1] = tmp.w;
  }
  outVal = val;
}

// Main dispatcher for unpacked load
template<typename T>
NCCL_DEVICE_INLINE void ncclLLBufferLoadLLSyncUnpacked(void* ptr, int pitch, T& outVal, uint32_t& outFlag) {
  using Traits = ncclLLBufferMemOpTraits<T>;
  ncclLLBufferLoadLLSyncUnpackedImpl<T>(ptr, pitch, outVal, outFlag, typename Traits::OpTag{});
}


/**
 * Store the value to the given pointer in Poison mode, it is
 * simply a wrapper around ncclLLBufferStoreImpl
 */
template<bool Multimem, typename T>
NCCL_DEVICE_INLINE void ncclLLBufferStorePoisonSync(void* ptr, T const& data) {
  using Traits = ncclLLBufferMemOpTraits<T>;
  ncclLLBufferStoreImpl<Multimem, T>(ptr, data, typename Traits::OpTag{});
}

template<bool Multimem, typename T>
NCCL_DEVICE_INLINE void ncclLLBufferStoreLLSync(void* ptr, T const& data, int pitch, uint32_t epoch) {
  using Traits = ncclLLBufferMemOpTraits<T>;
  ncclLLBufferStoreLLSyncImpl<Multimem, T>(ptr, data, pitch, epoch, typename Traits::OpTag{});
}


// Returns the slot size in bytes for a given data type in LL sync mode,
// it is basically multiplying the number of bytes of the data type by 2 when
// the size of T is less than 8 bytes. This should be used to calculate
// the correct slot / offset when sending out and receiving in LL sync mode
template<typename T>
NCCL_DEVICE_INLINE constexpr size_t ncclLLBufferLLSyncSlotSize() {
  #if __cpp_if_constexpr
  if constexpr (sizeof(T) <= 8)
  #else
  if (sizeof(T) <= 8)
  #endif
    return sizeof(T) * 2;
  return 16;
}


// ==================== Constructor/Destructor ====================

template<ncclLLSyncMode Mode, bool Multimem>
NCCL_DEVICE_INLINE ncclLLBuffer<Mode, Multimem>::ncclLLBuffer(
    ncclSymPtr<char> buf, int bytesPerCtaPerEpoch, int block,
    uint8_t roundRobinFactor,
    ncclMultimemHandle mmHandle
  ):
  ncclLLBuffer_internal<Mode, Multimem>{
    buf, mmHandle, /*currentBufferOffset=*/ 0,
    /*pitch=*/ bytesPerCtaPerEpoch, block,
    /*roundRobinFactor=*/ roundRobinFactor,
    /*epoch=*/ 2
  }
{
  // if (Mode != ncclPoison && !this->isReductionBuffer()) {
  //   // If the sync mode is LL or LL128 and it is not a reduction buffer,
  //   // throw an error
  //   printf("Error: Multiple buffering needs to be used to support LL or LL128 sync mode\n");
  //   return;
  // }
  if (Mode == ncclLL && this->isReductionBuffer()) {
    this->pitch <<= 1;
  }
  this->currentBufferOffset = this->calcBufferOffset();
}

template<ncclLLSyncMode Mode, bool Multimem>
NCCL_DEVICE_INLINE ncclLLBuffer<Mode, Multimem>::~ncclLLBuffer() {
  // No cleanup needed, user manages the buffer
}

// ==================== Configuration Queries ====================
template<ncclLLSyncMode Mode, bool Multimem>
NCCL_DEVICE_INLINE uint32_t ncclLLBuffer<Mode, Multimem>::currentEpoch() const {
  return this->epoch;
}

// ==================== Buffer Access ====================

template<ncclLLSyncMode Mode, bool Multimem>
NCCL_DEVICE_INLINE void* ncclLLBuffer<Mode, Multimem>::localPtr() const {
  return this->buf.localPtr();
}

template<ncclLLSyncMode Mode, bool Multimem>
NCCL_DEVICE_INLINE void* ncclLLBuffer<Mode, Multimem>::peerPtr(ncclTeam const& team, int peer) const {
  return this->buf.peerPtr(team, peer);
}

template<ncclLLSyncMode Mode, bool Multimem>
NCCL_DEVICE_INLINE void* ncclLLBuffer<Mode, Multimem>::multimemPtr() const {
  #if __cpp_if_constexpr
  if constexpr (!Multimem) return nullptr;
  #else
  if (!Multimem) return nullptr;
  #endif
  return this->buf.multimemPtr(this->mmHandle);
}

template<ncclLLSyncMode Mode, bool Multimem>
NCCL_DEVICE_INLINE ncclSymPtr<char> ncclLLBuffer<Mode, Multimem>::symPtr() const {
  return this->buf;
}

// ==================== Private Helpers ====================

// ==================== Send Operations ====================

template<ncclLLSyncMode Mode, bool Multimem>
template<typename T>
NCCL_DEVICE_INLINE void ncclLLBuffer<Mode, Multimem>::send(ncclTeam const& team, int peer, int elt, T const& data) const
{
  void* peerBuf = this->buf.peerPtr(team, peer) + this->currentBufferOffset;
  #if __cpp_if_constexpr
  if constexpr (Mode == ncclLL) {
  #else
  if (Mode == ncclLL) {
  #endif
    // LL format: {data[0], flag, data[1], flag}
    // Slot size is always 16 bytes (uint4) for LL mode
    size_t eltSize = ncclLLBufferLLSyncSlotSize<T>();
    char *dst = (char*)peerBuf + elt * eltSize;
    ncclLLBufferStoreLLSync</*Multimem=*/false, T>(dst, data, this->pitch, this->epoch);
  } else
  #if __cpp_if_constexpr
  if constexpr (Mode == ncclLL128) {
  #else
  if (Mode == ncclLL128) {
  #endif
    // LL128: TODO
  } else {
    // ncclPoison: Use size-appropriate vector instruction
    // Slot size depends on sizeof(T)
    T* dst = (T*) peerBuf + elt;
    ncclLLBufferStorePoisonSync</*Multimem=*/false, T>(dst, data);
  }
}

// ==================== Broadcast Operations ====================

/**
 * Broadcast the given data to all peers at the specified slot.
 * Uses two-level unrolling for optimal instruction-level parallelism:
 *   - Outer loop: iterates in chunks of Unroll peers (not unrolled by compiler)
 *   - Inner loop: processes Unroll peers (fully unrolled for ILP)
 * @tparam Unroll: Number of peers to process per inner loop iteration (default 8)
 * @param team: Team for peer addressing
 * @param elt: the element index to broadcast to
 * @param data: the data to broadcast
 */
template<ncclLLSyncMode Mode, bool Multimem>
template<int Unroll, typename T>
NCCL_DEVICE_INLINE void ncclLLBuffer<Mode, Multimem>::bcast(ncclTeam const& team, int elt, T const& data) const
{
  // Copy struct members to local variables ONCE to encourage register usage
  // This avoids repeated this-> dereferences in the hot loop
  ncclSymPtr<char> buf = this->buf;
  uint32_t offset = this->currentBufferOffset;
  int nRanks = team.nRanks;

  #if __cpp_if_constexpr
  if constexpr (Multimem) {
  #else
  if (Multimem) {
  #endif
    void* mmBuf = buf.multimemPtr(this->mmHandle) + offset;
    // If multi-cast is available
    #if __cpp_if_constexpr
    if constexpr (Mode == ncclLL) {
    #else
    if (Mode == ncclLL) {
    #endif
      size_t eltSize = ncclLLBufferLLSyncSlotSize<T>();
      char *dst = (char*)mmBuf + elt * eltSize;
      ncclLLBufferStoreLLSync</*Multimem=*/true, T>((void *) dst, data, this->pitch, this->epoch);
    } else
    #if __cpp_if_constexpr
    if constexpr (Mode == ncclLL128) {
    #else
    if (Mode == ncclLL128) {
    #endif
      // TODO
    } else
    #if __cpp_if_constexpr
    if constexpr (Mode == ncclPoison) {
    #else
    if (Mode == ncclPoison) {
    #endif
      T* dst = (T*)mmBuf + elt;
      ncclLLBufferStorePoisonSync</*Multimem=*/true, T>((void *) dst, data);
    }

  } else {
    // Two-level unrolling for uni-cast: outer loop not unrolled, inner loop fully unrolled
    // This pattern maximizes ILP while keeping code size manageable
    #if __cpp_if_constexpr
    if constexpr (Mode == ncclPoison) {
    #else
    if (Mode == ncclPoison) {
    #endif
      int dr = 0;
      int r = 0;
      // Outer loop: iterate in chunks of Unroll peers (not unrolled by compiler)
      #pragma unroll 1
      for (; dr + Unroll <= nRanks; dr += Unroll) {
        // Inner loop: fully unrolled for Unroll peers
        #pragma unroll
        for (int ur = 0; ur < Unroll; ur++) {
          void* peerBuf = buf.peerPtr(team, r) + offset;
          T* dst = (T*)peerBuf + elt;
          ncclLLBufferStorePoisonSync</*Multimem=*/false, T>(dst, data);
          ++r;
          if (r == nRanks) r = 0;
        }
      }
      // Tail loop: handle remaining peers (unrolled with early exit)
      #pragma unroll
      for (int ur = 0; ur < Unroll; ur++, dr++) {
        if (dr == nRanks) break;
        void* peerBuf = buf.peerPtr(team, r) + offset;
        T* dst = (T*)peerBuf + elt;
        ncclLLBufferStorePoisonSync</*Multimem=*/false, T>(dst, data);
        r += 1;
        if (r == nRanks) r = 0;
      }
    } else
    #if __cpp_if_constexpr
    if constexpr (Mode == ncclLL) {
    #else
    if (Mode == ncclLL) {
    #endif
      size_t eltSize = ncclLLBufferLLSyncSlotSize<T>();
      int pitch = this->pitch;
      uint8_t epoch = this->epoch;
      int dr = 0;
      int r = 0;
      // Outer loop: iterate in chunks of Unroll peers (not unrolled by compiler)
      #pragma unroll 1
      for (; dr + Unroll <= nRanks; dr += Unroll) {
        // Inner loop: fully unrolled for Unroll peers
        #pragma unroll
        for (int ur = 0; ur < Unroll; ur++) {
          void* peerBuf = buf.peerPtr(team, r) + offset;
          char *dst = (char*)peerBuf + elt * eltSize;
          ncclLLBufferStoreLLSync</*Multimem=*/false, T>(dst, data, pitch, epoch);
          r += 1;
          if (r == nRanks) r = 0;
        }
      }
      // Tail loop: handle remaining peers (unrolled with early exit)
      #pragma unroll
      for (int ur = 0; ur < Unroll; ur++, dr++) {
        if (dr == nRanks) break;
        void* peerBuf = buf.peerPtr(team, r) + offset;
        char *dst = (char*)peerBuf + elt * eltSize;
        ncclLLBufferStoreLLSync</*Multimem=*/false, T>(dst, data, pitch, epoch);
        r += 1;
        if (r == nRanks) r = 0;
      }
    }
  }
}

// ==================== Receive Operations ====================

template<ncclLLSyncMode Mode, bool Multimem>
template<typename T, bool Reset>
NCCL_DEVICE_INLINE T ncclLLBuffer<Mode, Multimem>::recv(int elt) const {
  T ret[1];
  this->template recvUnrolled</*MinEltCount=*/1, /*MaxEltCount=*/1, T, Reset>(elt, 1, 0, ret);
  return ret[0];
}


template<ncclLLSyncMode Mode, bool Multimem>
template<int MinEltCount, int MaxEltCount, typename T, bool Reset>
NCCL_DEVICE_INLINE void ncclLLBuffer<Mode, Multimem>::recvUnrolled(
    int eltStart, int eltCount, int eltStride,
    T (&elts)[MaxEltCount]
  ) const {

  // Copy struct members to local variables ONCE to encourage register usage
  void* buf = this->buf.localPtr() + this->currentBufferOffset;

  #if __cpp_if_constexpr
  if constexpr (Mode == ncclLL) {
  #else
  if (Mode == ncclLL) {
  #endif
    // ************ SyncMode LL ************
    // Copy pitch and epoch to locals before the hot loop
    int pitch = this->pitch;
    uint8_t epoch = this->epoch;

    size_t eltSize = ncclLLBufferLLSyncSlotSize<T>();
    char *bufPtr = (char*)buf + eltStart * eltSize;

    #pragma unroll 1
    while (true) {
      bool okAll = true;
      uint32_t flag;  // Single scalar, reused each iteration
      #pragma unroll
      for (int u = 0; u < MaxEltCount; u++) {
        if (u < MinEltCount || u < eltCount) {
          ncclLLBufferLoadLLSyncUnpacked<T>(bufPtr + u * eltStride * eltSize, pitch, elts[u], flag);
          okAll &= (flag == epoch);
        }
      }
      if (__builtin_expect(okAll, true)) break;
    }

    #if __cpp_if_constexpr
    if constexpr (Reset) {
    #else
    if (Reset) {
    #endif
      #pragma unroll
      for (int u = 0; u < MaxEltCount; u++) {
        if (u < MinEltCount || u < eltCount) {
          using OpType = typename ncclLLBufferEltOpTraits<T>::EltOpType;
          OpType zeroVal = ncclLLBufferEltOpTraits<T>::zeroVal;
          ncclLLBufferStoreLLSync</*Multimem=*/false, OpType>((void *)((char *)bufPtr + u * eltStride * eltSize),
                                          zeroVal, pitch, 0);
        }
      }
    }
  } else
  #if __cpp_if_constexpr
  if constexpr (Mode == ncclPoison) {
  #else
  if (Mode == ncclPoison) {
  #endif
    // ************ SyncMode Poison ************
    // Use size-appropriate slot size
    T* bufPtr = (T*)buf + eltStart;
    #pragma unroll 1
    while (true) {
      bool okAll = true;
      #pragma unroll
      for (int u = 0; u < MaxEltCount; ++u) {
        if (u < MinEltCount || u < eltCount) {
          elts[u] = ncclLLBufferLoadPoisonSync<T>(bufPtr + u * eltStride);
          okAll &= !ncclLLBufferIsPoisoned<T>(elts[u]);
        }
      }
      if (__builtin_expect(okAll, true)) break;
    }

    #if __cpp_if_constexpr
    if constexpr (Reset) {
    #else
    if (Reset) {
    #endif
      #pragma unroll
      for (int u = 0; u < MaxEltCount; u++) {
        if (u < MinEltCount || u < eltCount) {
          T poison = ncclLLBufferGetPoisonVal<T>();
          ncclLLBufferStorePoisonSync</*Multimem=*/false, T>(bufPtr + u * eltStride, poison);
        }
      }
    }
  } else {
    // ************ SyncMode LL128 ************
    // TODO
  }
}

// ==================== Receive and Reduce ====================

/**
 * Receive and reduce from the buffer with unrolling.
 * Reduce data from slotCount slots starting from eltStart with stride slotStride.
 * Return: the reduced result in the accumulator type
 */
template<ncclLLSyncMode Mode, bool Multimem>
template<int Unroll, typename T, bool Reset, typename EltToAcc, typename Reduce>
NCCL_DEVICE_INLINE auto ncclLLBuffer<Mode, Multimem>::recvReduce(
    int eltStart, int eltCount, int eltStride,
    EltToAcc eltToAcc, Reduce reduce
  ) const -> decltype(eltToAcc(nccl::utility::declval<T>())) {
  using Acc = decltype(eltToAcc(nccl::utility::declval<T>()));
  Acc acc;
  int i = 0;

  #pragma unroll 1
  for (; i + Unroll <= eltCount; i += Unroll) {
    T elts[Unroll];
    this->template recvUnrolled</*MinEltCount=*/Unroll, /*MaxEltCount=*/Unroll, T, Reset>(eltStart + i * eltStride, Unroll, eltStride, elts);
    Acc acc0 = eltToAcc(elts[0]);
    acc = i==0 ? acc0 : reduce(acc, acc0);
    #pragma unroll
    for (int j=1; j < Unroll; j++) acc = reduce(acc, eltToAcc(elts[j]));
  }

  // Tail handling
  if (i < eltCount) {
    T elts[Unroll];
    this->template recvUnrolled</*MinEltCount=*/1, /*MaxEltCount=*/Unroll, T, Reset>(eltStart + i * eltStride, eltCount - i, eltStride, elts);
    Acc acc0 = eltToAcc(elts[0]);
    acc = i==0 ? acc0 : reduce(acc, acc0);
    #pragma unroll
    for (int j = 1; j < Unroll - 1; ++j) {
      if (i + j < eltCount) acc = reduce(acc, eltToAcc(elts[j]));
    }
  }

  return acc;
}

// ==================== Reset Operations ====================

/**
 * Reset the element at the given index as per the sync mode.
 * If the sync mode is LL or LL128, the element is reset to 0.
 * If the sync mode is Poison, the element is set to the poison value using size-appropriate instruction.
 */
template<ncclLLSyncMode Mode, bool Multimem>
template<typename T>
NCCL_DEVICE_INLINE void ncclLLBuffer<Mode, Multimem>::reset(int elt) const {
  char* buf = this->buf.localPtr() + this->currentBufferOffset;
  #if __cpp_if_constexpr
  if constexpr (Mode == ncclLL || Mode == ncclLL128) {
  #else
  if (Mode == ncclLL || Mode == ncclLL128) {
  #endif
    using OpType = typename ncclLLBufferEltOpTraits<T>::EltOpType;
    OpType data = ncclLLBufferEltOpTraits<T>::zeroVal;
    ncclLLBufferStoreLLSync</*Multimem=*/false, OpType>((void *)((char *)buf + elt * ncclLLBufferLLSyncSlotSize<T>()), data, this->pitch, 0);
  } else {
    // ncclPoison: use size-appropriate poison instruction
    T poison = ncclLLBufferGetPoisonVal<T>();
    ncclLLBufferStorePoisonSync</*Multimem=*/false, T>((void *)(((T*)buf) + elt), poison);
  }
}

template<ncclLLSyncMode Mode, bool Multimem>
template<typename T>
NCCL_DEVICE_INLINE void ncclLLBuffer<Mode, Multimem>::resetRange(int eltStart, int eltCount, int eltStride) const {
  #pragma unroll
  for (int i = 0; i < eltCount; ++i) {
    this->reset<T>(eltStart + i * eltStride);
  }
}

// ==================== Flag Management ====================

/**
 * Advance the epoch flag.
 */
template<ncclLLSyncMode Mode, bool Multimem>
NCCL_DEVICE_INLINE void ncclLLBuffer<Mode, Multimem>::advanceEpoch() {
  this->epoch += (this->epoch == (uint8_t) -1u) ? 3 : 1;
  this->currentBufferOffset = this->calcBufferOffset();
}

template<ncclLLSyncMode Mode, bool Multimem>
NCCL_DEVICE_INLINE void ncclLLBuffer<Mode, Multimem>::resetEpoch() {
  this->epoch = 2;
  this->currentBufferOffset = this->calcBufferOffset();
}

#endif // __CUDACC__

#endif // _NCCL_DEVICE_LL_BUFFER__FUNCS_H_
