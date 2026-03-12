/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_LL_BUFFER__TYPES_H_
#define _NCCL_DEVICE_LL_BUFFER__TYPES_H_
#include "../ll_buffer.h"
#include "../ptr.h"
#include <type_traits>

// Everything below requires CUDA compiler (uses CUDA types like uint2, uint4)

#if __CUDACC__

// Tag types for operation dispatch (empty structs, zero runtime cost)
struct ncclLLOp8   {};
struct ncclLLOp16  {};   // 16-bit operations (1-2 byte types)
struct ncclLLOp32  {};   // 32-bit operations (3-4 byte types)
struct ncclLLOp64  {};   // 64-bit operations (5-8 byte types)
struct ncclLLOp128 {};  // 128-bit operations (9-16 byte types)

// ===================== LL mode element types =====================

// Retrieve the type of a single primitive operation for LL Sync mode
// For instance, if the category is 0, the element type is uint8_t,
// but the operation needs to be uint16_t since we need to include
// the flag in the operation.
template<int Category> struct ncclLLBufferEltOpTypeByCategory;

template<> struct ncclLLBufferEltOpTypeByCategory<0> { // 1 byte
  static constexpr uint8_t zeroVal = 0;
  using EltOpType = uint8_t;
};

template<> struct ncclLLBufferEltOpTypeByCategory<1> { // 2 bytes
  static constexpr uint16_t zeroVal = 0;
  using EltOpType = uint16_t;
};

template<> struct ncclLLBufferEltOpTypeByCategory<2> { // 4 bytes
  static constexpr uint32_t zeroVal = 0;
  using EltOpType = uint32_t;
};

template<> struct ncclLLBufferEltOpTypeByCategory<3> { // 5-8 bytes
  static constexpr uint2 zeroVal = {0, 0};
  using EltOpType = uint2;
};

template<> struct ncclLLBufferEltOpTypeByCategory<4> { // 9-16 bytes
  static constexpr uint4 zeroVal = {0, 0, 0, 0};
  using EltOpType = uint4;
};

constexpr int ncclLLBufferEltOpCategory(int size) {
  return (size == 1) ? 0 : (size == 2) ? 1 : (size == 4) ? 2 : (size == 8) ? 3 : 4;
}

template<typename T>
struct ncclLLBufferEltOpTraits : ncclLLBufferEltOpTypeByCategory<ncclLLBufferEltOpCategory(sizeof(T))> {};


// ==================== Poison Value Traits ====================
//
// Similar to ncclLLBufferMemOpByCategory, this uses size-based defaults with
// explicit overrides for floating-point types that need specific NaN patterns.
//
// How it works:
// 1. ncclLLBufferPoisonByCategory<N> provides default poison values by size category
// 2. ncclLLBufferPoisonTraits<T> inherits from the appropriate category
// 3. Floating-point types have explicit specializations to use NaN values
//
// Usage: ncclLLBufferPoisonTraits<float>::poisonBits32 → 0x7FC00000 (FP32 qNaN)
//        ncclLLBufferPoisonTraits<int32_t>::poisonBits32 → 0xDEADBEEF (default for 4-byte)

// Base trait templates by size category (default poison patterns)
template<int Category> struct ncclLLBufferPoisonByCategory;

template<> struct ncclLLBufferPoisonByCategory<0> {  // 1 byte
  static constexpr uint32_t poisonBits32 = 0x7F7F7F7Fu;
  static constexpr uint8_t poison = 0x7Fu;
  using PoisonType = uint8_t;
};

template<> struct ncclLLBufferPoisonByCategory<1> {  // 2 bytes
  static constexpr uint32_t poisonBits32 = 0x7E007E00u;
  static constexpr uint16_t poison = 0x7E00u;
  using PoisonType = uint16_t;
};

template<> struct ncclLLBufferPoisonByCategory<2> {  // 4 bytes
  static constexpr uint32_t poisonBits32 = 0xFFFAFFFAu;
  static constexpr uint32_t poison = 0xFFFAFFFAu;
  using PoisonType = uint32_t;
};


template<> struct ncclLLBufferPoisonByCategory<3> {  // > 8 bytes
  static constexpr uint32_t poisonBits32 = 0xFFFAFFFAu;
  static constexpr uint64_t poison = 0xFFFAFFFAFFFAFFFAu;
  using PoisonType = uint64_t;
};

template<> struct ncclLLBufferPoisonByCategory<4> {  // > 16 bytes
  static constexpr uint32_t poisonBits32 = 0xFFFAFFFAu;
  static constexpr uint4 poison = {0xFFFAFFFA, 0xFFFAFFFA, 0xFFFAFFFA, 0xFFFAFFFA};
  using PoisonType = uint4;
};

// Helper to get poison category (reuses ncclLLMemOpCategory)
constexpr int ncclLLBufferPoisonCategory(int size) {
  return (size == 1) ? 0 : (size == 2) ? 1 : (size == 4) ? 2 : (size == 8) ? 3 : (size == 16) ? 4 : 2;
}

// Primary template - uses size-based default poison value
// Any type without an explicit specialization gets the default pattern
template<typename T>
struct ncclLLBufferPoisonTraits : ncclLLBufferPoisonByCategory<ncclLLBufferPoisonCategory(sizeof(T))> {};

// ==================== Floating-Point Type Overrides ====================
// These types need specific NaN patterns instead of the default

// ==================== Poison Memory Operation Traits ====================
// These traits select the appropriate vector width for load/store operations
// based on the data type size.
//
// How this works:
// 1. Tag types (ncclLLOp16, ncclLLOp32, etc.) are empty structs used for
//    function overload dispatch - the compiler selects which overload to
//    call based on the tag type passed as an argument.
//
// 2. ncclLLMemOpTraits<T> provides compile-time information about type T:
//    - OpTag: which tag type to use for dispatch
//    - slotBytes: how many bytes the slot should be
//    - StoreType: the underlying storage type (uint16_t, uint32_t, etc.)
//
// 3. When you call ncclLLStorePoison<float>(ptr, data):
//    - Traits = ncclLLMemOpTraits<float> → OpTag = ncclLLOp32
//    - Calls ncclLLStorePoisonImpl(ptr, data, ncclLLOp32{})
//    - Compiler selects the 32-bit overload of ncclLLStorePoisonImpl
//
// This avoids runtime if-else branches - all decisions happen at compile time.

// ==================== Internal Buffer State ====================

// Helper to compute which category a size falls into
// Returns: 0 for <=2, 1 for <=4, 2 for <=8, 3 for <=16
constexpr int ncclLLBufferMemOpCategory(int size) {
  return (size == 1) ? 0 : (size == 2) ? 1 : (size == 4) ? 2 : (size == 8) ? 3 : 4;
}

// Helper to compute slot bytes from size
constexpr int ncclLLSlotBytes(int size) {
  return (size <= 2) ? 2 : (size <= 4) ? 4 : (size <= 8) ? 8 : 16;
}

// Base trait templates for each category
template<int Category> struct ncclLLBufferMemOpByCategory;

template<> struct ncclLLBufferMemOpByCategory<0> {  // 1 byte
  using OpTag = ncclLLOp8;
  using StoreType = uint8_t;
  static constexpr int slotBytes = 1;
};


template<> struct ncclLLBufferMemOpByCategory<1> {  // 2 bytes
  using OpTag = ncclLLOp16;
  using StoreType = uint16_t;
  static constexpr int slotBytes = 2;
};

template<> struct ncclLLBufferMemOpByCategory<2> {  // 3-4 bytes
  using OpTag = ncclLLOp32;
  using StoreType = uint32_t;
  static constexpr int slotBytes = 4;
};

template<> struct ncclLLBufferMemOpByCategory<3> {  // 5-8 bytes
  using OpTag = ncclLLOp64;
  using StoreType = uint2;
  static constexpr int slotBytes = 8;
};

template<> struct ncclLLBufferMemOpByCategory<4> {  // 9-16 bytes
  using OpTag = ncclLLOp128;
  using StoreType = uint4;
  static constexpr int slotBytes = 16;
};

// Main trait template - derives from the appropriate category
// Usage: ncclLLMemOpTraits<float>::OpTag → ncclLLOp32
//        ncclLLMemOpTraits<double>::slotBytes → 8
template<typename T>
struct ncclLLBufferMemOpTraits : ncclLLBufferMemOpByCategory<ncclLLBufferMemOpCategory(sizeof(T))> {};

// ==================== Poison Value Helpers ====================
// These helper functions handle type conversions to avoid ambiguous operators
// when working with __half or __nv_bfloat16 types and uint16_t poison values.

// Get the poison value as the correct type T (using bit reinterpretation)
// Usage: T poisonVal = ncclLLGetPoisonVal<T>();
template<typename T>
NCCL_DEVICE_INLINE T ncclLLBufferGetPoisonVal() {
  using PoisonType = typename ncclLLBufferPoisonTraits<T>::PoisonType;
  static_assert(sizeof(T) == sizeof(PoisonType), "Type size mismatch");
  union {
    PoisonType bits;
    T typed;
  } u;
  u.bits = ncclLLBufferPoisonTraits<T>::poison;
  return u.typed;
}

// Check if a value is poisoned (bit-level comparison)
// Usage: if (ncclLLIsPoisoned(value)) { ... }
template<typename T>
NCCL_DEVICE_INLINE bool ncclLLBufferIsPoisonedImpl(T val, std::true_type) {
  using PoisonType = typename ncclLLBufferPoisonTraits<T>::PoisonType;
  union { T typed; PoisonType bits; };
  typed = val;
  return bits == ncclLLBufferPoisonTraits<T>::poison;
}

template<typename T>
NCCL_DEVICE_INLINE bool ncclLLBufferIsPoisonedImpl(T val, std::false_type) {
  using nccl::utility::divUp;
  union { T typed; uint32_t bits[divUp(sizeof(T), 4)]; };
  typed = val;
  return bits[0] == ncclLLBufferPoisonTraits<T>::poisonBits32;
}

template<typename T>
NCCL_DEVICE_INLINE bool ncclLLBufferIsPoisoned(T val) {
  return ncclLLBufferIsPoisonedImpl(val, std::integral_constant<bool, (sizeof(T) <= 4)>{});
}

// Minimal internal struct for ncclLLBuffer - optimized for register residency
// Removed unused fields: Coop coop, ncclDevComm const& comm, nRanks (use team.nRanks)
// Multimem is now a template parameter (compile-time) instead of runtime field
// Total size: ~40 bytes
template<ncclLLSyncMode Mode, bool Multimem>
struct ncclLLBuffer_internal {
  ncclSymPtr<char> buf;             // 16 bytes - symmetric buffer pointer
  ncclMultimemHandle mmHandle;      // 8 bytes - multimem handle (nullptr if not available)
  uint32_t currentBufferOffset;     // 4 bytes - pre-computed buffer offset
  int pitch;                        // 4 bytes - bytes per CTA per epoch
  int block;                        // 4 bytes - block index for offset calculation
  uint8_t roundRobinFactor;         // 1 byte - number of buffers for multiple buffering
  uint8_t epoch;                    // 1 byte - current flag value (for LL/LL128 modes)
  uint32_t subBuffer;         // 4 bytes - current sub-buffer index
  // Total: 42 Bytes
  

  /**
   * Check if the buffer is a reduction buffer.
   * A reduction buffer is a buffer that is used for multiple buffering,
   * and a LL buffer is deemed a reduction buffer if the roundRobinFactor is greater than 0.
   * @return true if the buffer is a reduction buffer, false otherwise.
   */
  NCCL_DEVICE_INLINE bool isReductionBuffer() const
  {
    return roundRobinFactor > 0;
  }

  NCCL_DEVICE_INLINE uint32_t calcBufferOffset() const {
    if (isReductionBuffer())
    {
      return block * pitch * roundRobinFactor + (subBuffer % roundRobinFactor) * pitch;
    }
    return 0;
  }
};
#endif

#endif // _NCCL_DEVICE_LL_BUFFER__TYPES_H_
