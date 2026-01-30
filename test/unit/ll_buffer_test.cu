/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*************************************************************************
 * Unit test for ncclLLBuffer API with configurable sync modes
 *
 * This implements an allreduce using:
 * - Symmetric memory for inter-rank communication
 * - ncclLLBuffer with configurable sync mode (Poison or LL)
 * - Double buffering via multipleBufferingFactor
 * - bcast() for data distribution and recvReduce() for reduction
 * - Random input values per rank/element for thorough testing
 *
 * Run: mpirun -np 4 ./ll_buffer_test [-t <type>] [-m <mode>]
 *
 * Supported types: float (default), double, int, int32, uint32, int64, uint64,
 *                  int8, uint8, half, bfloat16, fp8e4m3, fp8e5m2
 *
 * Supported sync modes: poison (default), ll
 ************************************************************************/

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#ifdef __CUDA_FP8_TYPES_EXIST__
#include <cuda_fp8.h>
#endif
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdint>
#include <type_traits>

// Include NCCL device headers
#include "nccl_device.h"

// Include common test utilities
#include "common.h"

#define MULTIPLE_BUFFERING_FACTOR 2

// Default values for configurable parameters
constexpr int DEFAULT_NUM_ELEMENTS = 32;
constexpr int DEFAULT_NUM_BLOCKS = 1;

//------------------------------------------------------------------------------
// Data type enumeration and helpers
//------------------------------------------------------------------------------
enum TestDataType {
  TEST_INT8,
  TEST_UINT8,
  TEST_INT32,
  TEST_UINT32,
  TEST_INT64,
  TEST_UINT64,
  TEST_HALF,
  TEST_FLOAT,
  TEST_DOUBLE,
  TEST_BFLOAT16,
#ifdef __CUDA_FP8_TYPES_EXIST__
  TEST_FP8E4M3,
  TEST_FP8E5M2,
#endif
  TEST_NUM_TYPES
};

const char* dataTypeToString(TestDataType type) {
  switch (type) {
    case TEST_INT8:     return "int8";
    case TEST_UINT8:    return "uint8";
    case TEST_INT32:    return "int32";
    case TEST_UINT32:   return "uint32";
    case TEST_INT64:    return "int64";
    case TEST_UINT64:   return "uint64";
    case TEST_HALF:     return "half";
    case TEST_FLOAT:    return "float";
    case TEST_DOUBLE:   return "double";
    case TEST_BFLOAT16: return "bfloat16";
#ifdef __CUDA_FP8_TYPES_EXIST__
    case TEST_FP8E4M3:  return "fp8e4m3";
    case TEST_FP8E5M2:  return "fp8e5m2";
#endif
    default:            return "unknown";
  }
}

ncclDataType_t toNcclDataType(TestDataType type) {
  switch (type) {
    case TEST_INT8:     return ncclInt8;
    case TEST_UINT8:    return ncclUint8;
    case TEST_INT32:    return ncclInt32;
    case TEST_UINT32:   return ncclUint32;
    case TEST_INT64:    return ncclInt64;
    case TEST_UINT64:   return ncclUint64;
    case TEST_HALF:     return ncclHalf;
    case TEST_FLOAT:    return ncclFloat;
    case TEST_DOUBLE:   return ncclDouble;
    case TEST_BFLOAT16: return ncclBfloat16;
#ifdef __CUDA_FP8_TYPES_EXIST__
    case TEST_FP8E4M3:  return ncclFloat8e4m3;
    case TEST_FP8E5M2:  return ncclFloat8e5m2;
#endif
    default:            return ncclFloat;
  }
}

size_t dataTypeSize(TestDataType type) {
  switch (type) {
    case TEST_INT8:     return sizeof(int8_t);
    case TEST_UINT8:    return sizeof(uint8_t);
    case TEST_INT32:    return sizeof(int32_t);
    case TEST_UINT32:   return sizeof(uint32_t);
    case TEST_INT64:    return sizeof(int64_t);
    case TEST_UINT64:   return sizeof(uint64_t);
    case TEST_HALF:     return sizeof(__half);
    case TEST_FLOAT:    return sizeof(float);
    case TEST_DOUBLE:   return sizeof(double);
    case TEST_BFLOAT16: return sizeof(__nv_bfloat16);
#ifdef __CUDA_FP8_TYPES_EXIST__
    case TEST_FP8E4M3:  return sizeof(__nv_fp8_e4m3);
    case TEST_FP8E5M2:  return sizeof(__nv_fp8_e5m2);
#endif
    default:            return sizeof(float);
  }
}

TestDataType parseDataType(const char* str) {
  if (strcmp(str, "int8") == 0 || strcmp(str, "char") == 0) return TEST_INT8;
  if (strcmp(str, "uint8") == 0)    return TEST_UINT8;
  if (strcmp(str, "int32") == 0 || strcmp(str, "int") == 0) return TEST_INT32;
  if (strcmp(str, "uint32") == 0)   return TEST_UINT32;
  if (strcmp(str, "int64") == 0)    return TEST_INT64;
  if (strcmp(str, "uint64") == 0)   return TEST_UINT64;
  if (strcmp(str, "half") == 0 || strcmp(str, "float16") == 0) return TEST_HALF;
  if (strcmp(str, "float") == 0 || strcmp(str, "float32") == 0) return TEST_FLOAT;
  if (strcmp(str, "double") == 0 || strcmp(str, "float64") == 0) return TEST_DOUBLE;
  if (strcmp(str, "bfloat16") == 0) return TEST_BFLOAT16;
#ifdef __CUDA_FP8_TYPES_EXIST__
  if (strcmp(str, "fp8e4m3") == 0 || strcmp(str, "float8e4m3") == 0) return TEST_FP8E4M3;
  if (strcmp(str, "fp8e5m2") == 0 || strcmp(str, "float8e5m2") == 0) return TEST_FP8E5M2;
#endif
  return TEST_FLOAT;  // Default
}

//------------------------------------------------------------------------------
// Sync mode enumeration and helpers
//------------------------------------------------------------------------------
enum TestSyncMode {
  SYNC_POISON,
  SYNC_LL
};

const char* syncModeToString(TestSyncMode mode) {
  switch (mode) {
    case SYNC_POISON: return "poison";
    case SYNC_LL:     return "ll";
    default:          return "unknown";
  }
}

TestSyncMode parseSyncMode(const char* str) {
  if (strcmp(str, "poison") == 0) return SYNC_POISON;
  if (strcmp(str, "ll") == 0)     return SYNC_LL;
  return SYNC_POISON;  // Default
}

//------------------------------------------------------------------------------
// Accumulator type enumeration and helpers
//------------------------------------------------------------------------------
enum TestAccumType {
  ACCUM_SAME,    // Same as data type (default NCCL behavior for most types)
  ACCUM_FLOAT,   // Use float for accumulation (NCCL default for half/bf16/fp8)
  ACCUM_DOUBLE   // Use double for accumulation (higher precision)
};

const char* accumTypeToString(TestAccumType type) {
  switch (type) {
    case ACCUM_SAME:   return "same";
    case ACCUM_FLOAT:  return "float";
    case ACCUM_DOUBLE: return "double";
    default:           return "unknown";
  }
}

TestAccumType parseAccumType(const char* str) {
  if (strcmp(str, "same") == 0 || strcmp(str, "default") == 0) return ACCUM_SAME;
  if (strcmp(str, "float") == 0 || strcmp(str, "float32") == 0) return ACCUM_FLOAT;
  if (strcmp(str, "double") == 0 || strcmp(str, "float64") == 0) return ACCUM_DOUBLE;
  return ACCUM_SAME;  // Default
}

void printUsage(const char* progName) {
  printf("Usage: %s [-t <type>] [-m <mode>] [-acc <accum>] [-n <elems>] [-b <blocks>] [-mm] [-a] [-h]\n", progName);
  printf("  -t <type>   Data type to test. Default: float\n");
  printf("              Supported types:\n");
  printf("                int8, uint8, int32 (int), uint32,\n");
  printf("                int64, uint64, half (float16),\n");
  printf("                float (float32), double (float64), bfloat16\n");
#ifdef __CUDA_FP8_TYPES_EXIST__
  printf("                fp8e4m3 (float8e4m3), fp8e5m2 (float8e5m2) [SM90+ only]\n");
#endif
  printf("  -m <mode>   Sync mode to use. Default: poison\n");
  printf("              Supported modes: poison, ll\n");
  printf("  -acc <type> Accumulator type for reduction. Default: same\n");
  printf("              Supported types:\n");
  printf("                same   - Use same type as data (identity conversion)\n");
  printf("                float  - Accumulate in float (NCCL default for half/bf16/fp8)\n");
  printf("                double - Accumulate in double (highest precision)\n");
  printf("  -n <elems>  Number of elements to allreduce. Default: %d\n", DEFAULT_NUM_ELEMENTS);
  printf("  -b <blocks> Number of CUDA thread blocks (CTAs). Default: %d\n", DEFAULT_NUM_BLOCKS);
  printf("  -mm         Enable multimem buffer (requires H100+ with NVSwitch)\n");
  printf("  -a          Test all data types\n");
  printf("  -h          Show this help message\n");
}

//------------------------------------------------------------------------------
// Device helper function for type-safe addition
// FP8 types don't have native operator+, so we need to go through __half
//------------------------------------------------------------------------------
template<typename T>
__device__ __forceinline__ T deviceAdd(T a, T b) {
  return a + b;
}

#ifdef __CUDA_FP8_TYPES_EXIST__
template<>
__device__ __forceinline__ __nv_fp8_e4m3 deviceAdd<__nv_fp8_e4m3>(__nv_fp8_e4m3 a, __nv_fp8_e4m3 b) {
  return __nv_fp8_e4m3(__hadd(__half(a), __half(b)));
}

template<>
__device__ __forceinline__ __nv_fp8_e5m2 deviceAdd<__nv_fp8_e5m2>(__nv_fp8_e5m2 a, __nv_fp8_e5m2 b) {
  return __nv_fp8_e5m2(__hadd(__half(a), __half(b)));
}
#endif

//------------------------------------------------------------------------------
// Device helper functions for accumulator type conversion
//------------------------------------------------------------------------------

// Default: use constructor-style conversion
template<typename To, typename From>
__device__ __forceinline__ To castType(From val) {
  return To(val);
}

// Specializations for __half
template<> __device__ __forceinline__ float castType<float, __half>(__half val) {
  return __half2float(val);
}
template<> __device__ __forceinline__ __half castType<__half, float>(float val) {
  return __float2half_rn(val);
}
template<> __device__ __forceinline__ double castType<double, __half>(__half val) {
  return static_cast<double>(__half2float(val));
}
template<> __device__ __forceinline__ __half castType<__half, double>(double val) {
  return __float2half_rn(static_cast<float>(val));
}

// Specializations for __nv_bfloat16
template<> __device__ __forceinline__ float castType<float, __nv_bfloat16>(__nv_bfloat16 val) {
  return __bfloat162float(val);
}
template<> __device__ __forceinline__ __nv_bfloat16 castType<__nv_bfloat16, float>(float val) {
  return __float2bfloat16_rn(val);
}
template<> __device__ __forceinline__ double castType<double, __nv_bfloat16>(__nv_bfloat16 val) {
  return static_cast<double>(__bfloat162float(val));
}
template<> __device__ __forceinline__ __nv_bfloat16 castType<__nv_bfloat16, double>(double val) {
  return __float2bfloat16_rn(static_cast<float>(val));
}

#ifdef __CUDA_FP8_TYPES_EXIST__
// FP8 types use constructor for conversion
template<> __device__ __forceinline__ float castType<float, __nv_fp8_e4m3>(__nv_fp8_e4m3 val) {
  return float(val);
}
template<> __device__ __forceinline__ __nv_fp8_e4m3 castType<__nv_fp8_e4m3, float>(float val) {
  return __nv_fp8_e4m3(val);
}
template<> __device__ __forceinline__ float castType<float, __nv_fp8_e5m2>(__nv_fp8_e5m2 val) {
  return float(val);
}
template<> __device__ __forceinline__ __nv_fp8_e5m2 castType<__nv_fp8_e5m2, float>(float val) {
  return __nv_fp8_e5m2(val);
}
template<> __device__ __forceinline__ double castType<double, __nv_fp8_e4m3>(__nv_fp8_e4m3 val) {
  return static_cast<double>(float(val));
}
template<> __device__ __forceinline__ __nv_fp8_e4m3 castType<__nv_fp8_e4m3, double>(double val) {
  return __nv_fp8_e4m3(static_cast<float>(val));
}
template<> __device__ __forceinline__ double castType<double, __nv_fp8_e5m2>(__nv_fp8_e5m2 val) {
  return static_cast<double>(float(val));
}
template<> __device__ __forceinline__ __nv_fp8_e5m2 castType<__nv_fp8_e5m2, double>(double val) {
  return __nv_fp8_e5m2(static_cast<float>(val));
}
#endif

//------------------------------------------------------------------------------
// Kernel: One-shot AllReduce using ncclLLBuffer
//------------------------------------------------------------------------------
template<ncclLLSyncMode Mode, bool Multimem, typename T, typename Acc>
__global__ void allReduceKernel(
    ncclDevComm devComm,
    T* inputBuf,
    T* outputBuf,
    ncclSymPtr<char> scratchSymPtr,
    int nRanks,
    int myRank,
    int numElements
) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

  ncclTeam team = ncclTeamWorld(devComm);

  size_t bytesPerCtaPerEpoch = nRanks * blockDim.x * sizeof(T);

  // Multimem requires SM 90+; disable at compile time for older architectures
#if __CUDA_ARCH__ >= 900
  constexpr bool UseMultimem = Multimem;
#else
  constexpr bool UseMultimem = false;
#endif

  ncclLLBuffer<Mode, UseMultimem> llBuf(
    scratchSymPtr,
    /*bytesPerCtaPerEpoch=*/ bytesPerCtaPerEpoch,
    /*block=*/ blockIdx.x,
    /*roundRobinFactor=*/ MULTIPLE_BUFFERING_FACTOR,
    /*mmHandle=*/ UseMultimem ? devComm.lsaMultimem : ncclMultimemHandle{}
  );

  for (int i = tid; i < numElements; i += nthreads) {
    T myData = inputBuf[i];
    int slot = i % nthreads % blockDim.x + myRank * blockDim.x;
    llBuf.template bcast<4, T>(team, slot, myData);

    Acc result = llBuf.template recvReduce</*Unroll=*/4, T, /*Reset=*/true>(
      /*eltStart=*/ i % nthreads % blockDim.x,
      /*eltCount=*/ nRanks,
      /*eltStride=*/ blockDim.x,
      /*eltToAcc=*/ [](T val) -> Acc { return castType<Acc, T>(val); },
      /*reduce=*/ [](Acc a, Acc b) -> Acc { return deviceAdd(a, b); }
    );

    outputBuf[i] = castType<T, Acc>(result);
    llBuf.advanceEpoch();
  }
}

//------------------------------------------------------------------------------
// Random number generator helpers
//------------------------------------------------------------------------------

inline float generateRandomFloat(int rank, int index, unsigned int seed = 12345) {
  unsigned int state = seed + rank * 1000003 + index;
  state = state * 1103515245 + 12345;
  return 0.1f + 0.9f * (float)(state & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

inline int generateRandomInt(int rank, int index, int maxVal = 3, unsigned int seed = 12345) {
  unsigned int state = seed + rank * 1000003 + index;
  state = state * 1103515245 + 12345;
  return 1 + (state % maxVal);
}

template<typename T>
T generateTestValue(int rank, int index) {
  #if __cpp_if_constexpr
  if constexpr (std::is_integral<T>::value) {
  #else
  if (std::is_integral<T>::value)  {
  #endif
    int randVal = generateRandomInt(rank, index);
    return static_cast<T>(randVal);
  } else {
    float randVal = generateRandomFloat(rank, index);
    return static_cast<T>(randVal);
  }
}

template<>
__half generateTestValue<__half>(int rank, int index) {
  float randVal = generateRandomFloat(rank, index);
  return __float2half(randVal);
}

template<>
__nv_bfloat16 generateTestValue<__nv_bfloat16>(int rank, int index) {
  float randVal = generateRandomFloat(rank, index);
  return __float2bfloat16(randVal);
}

#ifdef __CUDA_FP8_TYPES_EXIST__
template<>
__nv_fp8_e4m3 generateTestValue<__nv_fp8_e4m3>(int rank, int index) {
  float randVal = generateRandomFloat(rank, index);
  return __nv_fp8_e4m3(randVal);
}

template<>
__nv_fp8_e5m2 generateTestValue<__nv_fp8_e5m2>(int rank, int index) {
  float randVal = generateRandomFloat(rank, index);
  return __nv_fp8_e5m2(randVal);
}
#endif

//------------------------------------------------------------------------------
// Value comparison helpers
//------------------------------------------------------------------------------

template<typename T>
bool compareValues(T a, T b, T tolerance) {
  T diff = (a > b) ? (a - b) : (b - a);
  return diff <= tolerance;
}

template<>
bool compareValues<float>(float a, float b, float tolerance) {
  return fabsf(a - b) <= tolerance;
}

template<>
bool compareValues<double>(double a, double b, double tolerance) {
  return fabs(a - b) <= tolerance;
}

template<>
bool compareValues<__half>(__half a, __half b, __half tolerance) {
  return fabsf(__half2float(a) - __half2float(b)) <= __half2float(tolerance);
}

template<>
bool compareValues<__nv_bfloat16>(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 tolerance) {
  return fabsf(__bfloat162float(a) - __bfloat162float(b)) <= __bfloat162float(tolerance);
}

#ifdef __CUDA_FP8_TYPES_EXIST__
template<>
bool compareValues<__nv_fp8_e4m3>(__nv_fp8_e4m3 a, __nv_fp8_e4m3 b, __nv_fp8_e4m3 tolerance) {
  return fabsf(float(a) - float(b)) <= float(tolerance);
}

template<>
bool compareValues<__nv_fp8_e5m2>(__nv_fp8_e5m2 a, __nv_fp8_e5m2 b, __nv_fp8_e5m2 tolerance) {
  return fabsf(float(a) - float(b)) <= float(tolerance);
}
#endif

template<typename T>
double toDouble(T val) {
  return static_cast<double>(val);
}

template<>
double toDouble<__half>(__half val) {
  return static_cast<double>(__half2float(val));
}

template<>
double toDouble<__nv_bfloat16>(__nv_bfloat16 val) {
  return static_cast<double>(__bfloat162float(val));
}

#ifdef __CUDA_FP8_TYPES_EXIST__
template<>
double toDouble<__nv_fp8_e4m3>(__nv_fp8_e4m3 val) {
  return static_cast<double>(float(val));
}

template<>
double toDouble<__nv_fp8_e5m2>(__nv_fp8_e5m2 val) {
  return static_cast<double>(float(val));
}
#endif

template<typename T>
T getTolerance() {
  return static_cast<T>(0);
}

template<>
float getTolerance<float>() {
  return 1e-5f;
}

template<>
double getTolerance<double>() {
  return 1e-10;
}

template<>
__half getTolerance<__half>() {
  return __float2half(1e-2f);
}

template<>
__nv_bfloat16 getTolerance<__nv_bfloat16>() {
  return __float2bfloat16(0.05f);
}

#ifdef __CUDA_FP8_TYPES_EXIST__
template<>
__nv_fp8_e4m3 getTolerance<__nv_fp8_e4m3>() {
  return __nv_fp8_e4m3(0.5f);
}

template<>
__nv_fp8_e5m2 getTolerance<__nv_fp8_e5m2>() {
  return __nv_fp8_e5m2(1.0f);
}
#endif

//------------------------------------------------------------------------------
// Compute expected sum
//------------------------------------------------------------------------------

template<typename T>
T computeExpectedSum(int nRanks, int index) {
  #if __cpp_if_constexpr
  if constexpr (std::is_integral<T>::value) {
  #else
  if (std::is_integral<T>::value) {
  #endif
    int64_t sum = 0;
    for (int r = 0; r < nRanks; r++) {
      sum += generateRandomInt(r, index);
    }
    return static_cast<T>(sum);
  } else {
    double sum = 0.0;
    for (int r = 0; r < nRanks; r++) {
      sum += static_cast<double>(generateRandomFloat(r, index));
    }
    return static_cast<T>(sum);
  }
}

template<>
__half computeExpectedSum<__half>(int nRanks, int index) {
  float sum = 0.0f;
  for (int r = 0; r < nRanks; r++) {
    __half val = __float2half(generateRandomFloat(r, index));
    sum = __half2float(__float2half(sum + __half2float(val)));
  }
  return __float2half(sum);
}

template<>
__nv_bfloat16 computeExpectedSum<__nv_bfloat16>(int nRanks, int index) {
  float sum = 0.0f;
  for (int r = 0; r < nRanks; r++) {
    __nv_bfloat16 val = __float2bfloat16(generateRandomFloat(r, index));
    sum = __bfloat162float(__float2bfloat16(sum + __bfloat162float(val)));
  }
  return __float2bfloat16(sum);
}

#ifdef __CUDA_FP8_TYPES_EXIST__
template<>
__nv_fp8_e4m3 computeExpectedSum<__nv_fp8_e4m3>(int nRanks, int index) {
  float sum = 0.0f;
  for (int r = 0; r < nRanks; r++) {
    __nv_fp8_e4m3 val = __nv_fp8_e4m3(generateRandomFloat(r, index));
    sum = float(__nv_fp8_e4m3(sum + float(val)));
  }
  return __nv_fp8_e4m3(sum);
}

template<>
__nv_fp8_e5m2 computeExpectedSum<__nv_fp8_e5m2>(int nRanks, int index) {
  float sum = 0.0f;
  for (int r = 0; r < nRanks; r++) {
    __nv_fp8_e5m2 val = __nv_fp8_e5m2(generateRandomFloat(r, index));
    sum = float(__nv_fp8_e5m2(sum + float(val)));
  }
  return __nv_fp8_e5m2(sum);
}
#endif

//------------------------------------------------------------------------------
// Templated test runner
//------------------------------------------------------------------------------

template<ncclLLSyncMode Mode, typename T, typename Acc>
int runTestWithModeAndAcc(int worldSize, int worldRank, ncclDataType_t ncclType, int numElements, int numBlocks, bool useMultimem) {
  int numDevices;
  CUDACHECK(cudaGetDeviceCount(&numDevices));
  int deviceId = worldRank % numDevices;
  CUDACHECK(cudaSetDevice(deviceId));

  printf("Rank %d/%d using GPU %d\n", worldRank, worldSize, deviceId);

  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  ncclUniqueId ncclId;
  if (worldRank == 0) {
    NCCLCHECK(ncclGetUniqueId(&ncclId));
  }
  MPICHECK(MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD));

  ncclComm_t comm;
  NCCLCHECK(ncclCommInitRank(&comm, worldSize, ncclId, worldRank));

  T* inputBuf;
  T* outputBuf;
  CUDACHECK(cudaMalloc(&inputBuf, numElements * sizeof(T)));
  CUDACHECK(cudaMalloc(&outputBuf, numElements * sizeof(T)));

  T* hostInput = (T*)malloc(numElements * sizeof(T));
  for (int i = 0; i < numElements; i++) {
    hostInput[i] = generateTestValue<T>(worldRank, i);
  }
  CUDACHECK(cudaMemcpyAsync(inputBuf, hostInput, numElements * sizeof(T),
                            cudaMemcpyHostToDevice, stream));

  // Buffer size must match the kernel's bytesPerCtaPerEpoch calculation
  // The kernel uses: bytesPerCtaPerEpoch = nRanks * blockDim.x * sizeof(T)
  // So we must allocate accordingly, not based on numElements
  // For LL mode, the pitch is doubled internally (to store flags), so we need 2x the buffer
  constexpr int BLOCK_SIZE = 512;  // Must match the kernel launch configuration
  size_t bytesPerCtaPerEpoch = worldSize * BLOCK_SIZE * sizeof(T);
  // LL mode doubles the pitch for flag storage
  size_t effectiveBytesPerCta = (Mode == ncclLL) ? bytesPerCtaPerEpoch * 2 : bytesPerCtaPerEpoch;
  size_t scratchBufSize = effectiveBytesPerCta * MULTIPLE_BUFFERING_FACTOR * numBlocks;

  if (worldRank == 0) {
    printf("Intermediate buffer size: %zu bytes\n", scratchBufSize);
    printf("  - Element size for type: %zu bytes\n", sizeof(T));
    printf("  - Elements per rank: %d\n", numElements);
    printf("  - Block size: %d\n", BLOCK_SIZE);
    printf("  - Bytes per CTA per epoch: %zu\n", bytesPerCtaPerEpoch);
    printf("  - Number of blocks: %d\n", numBlocks);
    printf("  - Sync mode: %s\n", Mode == ncclPoison ? "poison" : "ll");
    printf("  - Multimem: %s\n", useMultimem ? "enabled" : "disabled");
  }

  void* scratchBufRaw;
  NCCLCHECK(ncclMemAlloc(&scratchBufRaw, scratchBufSize));

  ncclWindow_t scratchWindow;
  NCCLCHECK(ncclCommWindowRegister(comm, scratchBufRaw, scratchBufSize, &scratchWindow, NCCL_WIN_COLL_SYMMETRIC));

  ncclSymPtr<char> scratchSymPtr(scratchWindow, 0);

  ncclDevResourceRequirements_t lsaBarReq = {};
  ncclLsaBarrierHandle_t lsaBarHandle;
  NCCLCHECK(ncclLsaBarrierCreateRequirement(
    ncclTeamWorld(comm),
    numBlocks,
    &lsaBarHandle,
    &lsaBarReq
  ));

  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  lsaBarReq.next = reqs.resourceRequirementsList;
  reqs.resourceRequirementsList = &lsaBarReq;
  reqs.lsaMultimem = useMultimem;

  ncclDevComm devComm;
  NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));

  #if __cpp_if_constexpr
  if constexpr (Mode == ncclPoison) {
  #else
  if (Mode == ncclPoison) {
  #endif
    NCCLCHECK(ncclLLPoisonBufferAsync(scratchBufRaw, scratchBufSize, ncclType, stream));
  } else {
    CUDACHECK(cudaMemsetAsync(scratchBufRaw, 0, scratchBufSize, stream));
  }

  CUDACHECK(cudaStreamSynchronize(stream));
  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

  printf("Rank %d: Launching allreduce kernel...\n", worldRank);

  // Dispatch based on useMultimem (Multimem is now a template parameter)
  if (useMultimem) {
    allReduceKernel<Mode, /*Multimem=*/true, T, Acc><<<numBlocks, 512, 0, stream>>>(
      devComm,
      inputBuf,
      outputBuf,
      scratchSymPtr,
      worldSize,
      worldRank,
      numElements
    );
  } else {
    allReduceKernel<Mode, /*Multimem=*/false, T, Acc><<<numBlocks, 512, 0, stream>>>(
      devComm,
      inputBuf,
      outputBuf,
      scratchSymPtr,
      worldSize,
      worldRank,
      numElements
    );
  }

  T* hostOutput = (T*)malloc(numElements * sizeof(T));

  CUDACHECK(cudaMemcpyAsync(hostOutput, outputBuf, numElements * sizeof(T),
                            cudaMemcpyDeviceToHost, stream));

  CUDACHECK(cudaStreamSynchronize(stream));

  // Validation
  T tolerance = getTolerance<T>();
  bool success = true;
  int errorCount = 0;
  const int maxErrors = 8;

  printf("Rank %d: Validating allreduce results...\n", worldRank);

  for (int i = 0; i < numElements; i++) {
    T expectedSum = computeExpectedSum<T>(worldSize, i);
    if (!compareValues(hostOutput[i], expectedSum, tolerance)) {
      if (errorCount < maxErrors) {
        printf("Rank %d: ERROR at index %d: expected %f, got %f (diff=%f)\n",
               worldRank, i, toDouble(expectedSum), toDouble(hostOutput[i]),
               toDouble(expectedSum) - toDouble(hostOutput[i]));
      }
      success = false;
      errorCount++;
    }
  }

  if (errorCount > maxErrors) {
    printf("Rank %d: ... and %d more errors\n", worldRank, errorCount - maxErrors);
  }

  if (success) {
    printf("Rank %d: SUCCESS - allreduce verified for all %d elements\n",
           worldRank, numElements);
  } else {
    printf("Rank %d: FAILED - %d/%d elements incorrect\n",
           worldRank, errorCount, numElements);
  }

  // Cleanup
  free(hostInput);
  free(hostOutput);
  CUDACHECK(cudaStreamDestroy(stream));
  NCCLCHECK(ncclCommWindowDeregister(comm, scratchWindow));
  NCCLCHECK(ncclMemFree(scratchBufRaw));
  CUDACHECK(cudaFree(inputBuf));
  CUDACHECK(cudaFree(outputBuf));
  NCCLCHECK(ncclCommDestroy(comm));

  return success ? 0 : 1;
}

template<ncclLLSyncMode Mode, typename T>
int runTestWithMode(int worldSize, int worldRank, ncclDataType_t ncclType, TestAccumType accumType, int numElements, int numBlocks, bool useMultimem) {
  switch (accumType) {
    case ACCUM_FLOAT:
      return runTestWithModeAndAcc<Mode, T, float>(worldSize, worldRank, ncclType, numElements, numBlocks, useMultimem);
    case ACCUM_DOUBLE:
      return runTestWithModeAndAcc<Mode, T, double>(worldSize, worldRank, ncclType, numElements, numBlocks, useMultimem);
    case ACCUM_SAME:
    default:
      return runTestWithModeAndAcc<Mode, T, T>(worldSize, worldRank, ncclType, numElements, numBlocks, useMultimem);
  }
}

template<typename T>
int runTest(int worldSize, int worldRank, ncclDataType_t ncclType, TestSyncMode syncMode, TestAccumType accumType, int numElements, int numBlocks, bool useMultimem) {
  if (syncMode == SYNC_LL) {
    return runTestWithMode<ncclLL, T>(worldSize, worldRank, ncclType, accumType, numElements, numBlocks, useMultimem);
  } else {
    return runTestWithMode<ncclPoison, T>(worldSize, worldRank, ncclType, accumType, numElements, numBlocks, useMultimem);
  }
}

//------------------------------------------------------------------------------
// Run test for a single data type
//------------------------------------------------------------------------------
int runTestForType(TestDataType dataType, int worldSize, int worldRank, TestSyncMode syncMode, TestAccumType accumType, int numElements, int numBlocks, bool useMultimem) {
  ncclDataType_t ncclType = toNcclDataType(dataType);

  if (worldRank == 0) {
    printf("----------------------------------------------\n");
    printf("Testing data type: %s (size: %zu bytes), sync mode: %s\n",
           dataTypeToString(dataType), dataTypeSize(dataType),
           syncModeToString(syncMode));
    printf("  Elements: %d, Blocks: %d\n", numElements, numBlocks);
    printf("----------------------------------------------\n");
  }

  int result;

  switch (dataType) {
    case TEST_INT8:
      result = runTest<int8_t>(worldSize, worldRank, ncclType, syncMode, accumType, numElements, numBlocks, useMultimem);
      break;
    case TEST_UINT8:
      result = runTest<uint8_t>(worldSize, worldRank, ncclType, syncMode, accumType, numElements, numBlocks, useMultimem);
      break;
    case TEST_INT32:
      result = runTest<int32_t>(worldSize, worldRank, ncclType, syncMode, accumType, numElements, numBlocks, useMultimem);
      break;
    case TEST_UINT32:
      result = runTest<uint32_t>(worldSize, worldRank, ncclType, syncMode, accumType, numElements, numBlocks, useMultimem);
      break;
    case TEST_INT64:
      result = runTest<int64_t>(worldSize, worldRank, ncclType, syncMode, accumType, numElements, numBlocks, useMultimem);
      break;
    case TEST_UINT64:
      result = runTest<uint64_t>(worldSize, worldRank, ncclType, syncMode, accumType, numElements, numBlocks, useMultimem);
      break;
    case TEST_HALF:
      result = runTest<__half>(worldSize, worldRank, ncclType, syncMode, accumType, numElements, numBlocks, useMultimem);
      break;
    case TEST_FLOAT:
      result = runTest<float>(worldSize, worldRank, ncclType, syncMode, accumType, numElements, numBlocks, useMultimem);
      break;
    case TEST_DOUBLE:
      result = runTest<double>(worldSize, worldRank, ncclType, syncMode, accumType, numElements, numBlocks, useMultimem);
      break;
    case TEST_BFLOAT16:
      result = runTest<__nv_bfloat16>(worldSize, worldRank, ncclType, syncMode, accumType, numElements, numBlocks, useMultimem);
      break;
#ifdef __CUDA_FP8_TYPES_EXIST__
    case TEST_FP8E4M3:
      result = runTest<__nv_fp8_e4m3>(worldSize, worldRank, ncclType, syncMode, accumType, numElements, numBlocks, useMultimem);
      break;
    case TEST_FP8E5M2:
      result = runTest<__nv_fp8_e5m2>(worldSize, worldRank, ncclType, syncMode, accumType, numElements, numBlocks, useMultimem);
      break;
#endif
    default:
      result = runTest<float>(worldSize, worldRank, ncclType, syncMode, accumType, numElements, numBlocks, useMultimem);
      break;
  }

  return result;
}

//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  TestDataType dataType = TEST_FLOAT;
  TestSyncMode syncMode = SYNC_POISON;
  TestAccumType accumType = ACCUM_SAME;
  int numElements = DEFAULT_NUM_ELEMENTS;
  int numBlocks = DEFAULT_NUM_BLOCKS;
  bool testAll = false;
  bool useMultimem = false;

  for (int i = 1; i < argc; i++) {
    if ((strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--type") == 0) && i + 1 < argc) {
      dataType = parseDataType(argv[i + 1]);
      i++;
    } else if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--mode") == 0) && i + 1 < argc) {
      syncMode = parseSyncMode(argv[i + 1]);
      i++;
    } else if ((strcmp(argv[i], "-acc") == 0 || strcmp(argv[i], "--accumulator") == 0) && i + 1 < argc) {
      accumType = parseAccumType(argv[i + 1]);
      i++;
    } else if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--elements") == 0) && i + 1 < argc) {
      numElements = atoi(argv[i + 1]);
      if (numElements <= 0) numElements = DEFAULT_NUM_ELEMENTS;
      i++;
    } else if ((strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--blocks") == 0) && i + 1 < argc) {
      numBlocks = atoi(argv[i + 1]);
      if (numBlocks <= 0) numBlocks = DEFAULT_NUM_BLOCKS;
      i++;
    } else if (strcmp(argv[i], "-mm") == 0 || strcmp(argv[i], "--multimem") == 0) {
      useMultimem = true;
    } else if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--all") == 0) {
      testAll = true;
    } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      printUsage(argv[0]);
      return 0;
    }
  }

  MPICHECK(MPI_Init(&argc, &argv));

  int worldSize, worldRank;
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));

  if (worldRank == 0) {
    printf("==============================================\n");
    printf("ncclLLBuffer Unit Test - AllReduce\n");
    printf("==============================================\n");
    printf("Number of ranks: %d\n", worldSize);
    printf("Elements per rank: %d\n", numElements);
    printf("Number of blocks (CTAs): %d\n", numBlocks);
    printf("Double buffering factor: %d\n", MULTIPLE_BUFFERING_FACTOR);
    printf("Sync mode: %s\n", syncModeToString(syncMode));
    printf("Multimem: %s\n", useMultimem ? "enabled" : "disabled");
    if (testAll) {
      printf("Testing ALL data types\n");
    } else {
      printf("Data type: %s (size: %zu bytes)\n",
             dataTypeToString(dataType), dataTypeSize(dataType));
    }
    printf("==============================================\n");
  }

  int globalResult = 0;

  if (testAll) {
    int passCount = 0;
    int failCount = 0;
    int skipCount = 0;

    TestDataType allTypes[] = {
      TEST_INT8,
      TEST_UINT8,
      TEST_INT32,
      TEST_UINT32,
      TEST_INT64,
      TEST_UINT64,
      TEST_HALF,
      TEST_FLOAT,
      TEST_DOUBLE,
      TEST_BFLOAT16,
#ifdef __CUDA_FP8_TYPES_EXIST__
      TEST_FP8E4M3,
      TEST_FP8E5M2,
#endif
    };
    int numTypes = sizeof(allTypes) / sizeof(allTypes[0]);

    for (int t = 0; t < numTypes; t++) {
      TestDataType currentType = allTypes[t];

      if (useMultimem) {
        bool is8bit = (currentType == TEST_INT8 || currentType == TEST_UINT8
#ifdef __CUDA_FP8_TYPES_EXIST__
                       || currentType == TEST_FP8E4M3 || currentType == TEST_FP8E5M2
#endif
                      );
        bool is16bit = (currentType == TEST_HALF || currentType == TEST_BFLOAT16);

        bool skipType = false;
        if (syncMode == SYNC_POISON && (is8bit || is16bit)) {
          skipType = true;
        } else if (syncMode == SYNC_LL && is8bit) {
          skipType = true;
        }

        if (skipType) {
          skipCount++;
          if (worldRank == 0) {
            printf("  --> %s: SKIPPED (unsupported with multimem in %s mode)\n",
                   dataTypeToString(currentType), syncModeToString(syncMode));
          }
          continue;
        }
      }

      // Skip double when using float accumulator (reduces accuracy)
      if (accumType == ACCUM_FLOAT && currentType == TEST_DOUBLE) {
        skipCount++;
        if (worldRank == 0) {
          printf("  --> %s: SKIPPED (float accumulator reduces accuracy for double)\n",
                 dataTypeToString(currentType));
        }
        continue;
      }

      MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

      int result = runTestForType(currentType, worldSize, worldRank, syncMode, accumType, numElements, numBlocks, useMultimem);

      int typeResult;
      MPICHECK(MPI_Allreduce(&result, &typeResult, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));

      if (typeResult == 0) {
        passCount++;
        if (worldRank == 0) {
          printf("  --> %s: PASSED\n", dataTypeToString(currentType));
        }
      } else {
        failCount++;
        globalResult = 1;
        if (worldRank == 0) {
          printf("  --> %s: FAILED\n", dataTypeToString(currentType));
        }
      }
    }

    if (worldRank == 0) {
      printf("==============================================\n");
      int testedTypes = numTypes - skipCount;
      printf("SUMMARY: %d/%d types passed", passCount, testedTypes);
      if (skipCount > 0) {
        printf(" (%d skipped)", skipCount);
      }
      printf("\n");
      if (failCount > 0) {
        printf("FAILED TYPES: %d\n", failCount);
      }
    }
  } else {
    int result = runTestForType(dataType, worldSize, worldRank, syncMode, accumType, numElements, numBlocks, useMultimem);
    MPICHECK(MPI_Allreduce(&result, &globalResult, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
  }

  if (worldRank == 0) {
    printf("==============================================\n");
    if (globalResult == 0) {
      printf("ALL RANKS PASSED\n");
    } else {
      printf("SOME RANKS FAILED\n");
    }
    printf("==============================================\n");
  }

  MPICHECK(MPI_Finalize());

  return globalResult;
}
