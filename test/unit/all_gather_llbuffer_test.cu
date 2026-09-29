/*************************************************************************
 * See LICENSE.txt for license information
 ************************************************************************/

// Public-API AllGather regression; Linux, one process per GPU. No MPI required.
// Build: make -f test/unit/Makefile.all_gather CUDA_ARCH=sm_103
// Checked run: make -f test/unit/Makefile.all_gather check CUDA_ARCH=sm_103
// Run with LD_LIBRARY_PATH pointing to the library under test and:
//   NCCL_CUMEM_ENABLE=1 NCCL_WIN_ENABLE=1 NCCL_NVLS_ENABLE=1 NCCL_GIN_ENABLE=0
//   NCCL_GRAPH_MIXING_SUPPORT=1 NCCL_SYM_KERNEL=AllGather_LLBufferMC
//   NCCL_SYM_LLBUFFER_SYNC=0 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING
//   timeout --signal=TERM --kill-after=10 120 ./build/all_gather_llbuffer_test --ranks 8
// Require actual "AllGather [Symmetric]: ... -> Kernel AllGather_LLBufferMC" dispatch;
// a fallback pass is not evidence for this regression. --help lists test options.
// Poison-mode data excludes reserved sentinel patterns. This is not a benchmark.
#include <cuda_runtime.h>
#include <nccl.h>
#include <pthread.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../../examples/common/include/nccl_utils.h"

constexpr size_t NCCL_TEST_GUARD = 128;
constexpr unsigned char NCCL_TEST_CANARY = 201;

struct ncclTestCase {
  size_t bytes;
  size_t offset;
};

struct ncclTestFailure {
  unsigned long long count;
  unsigned long long epoch;
  size_t index;
  int caseId;
  int region;
  unsigned int actual;
  unsigned int expected;
};

struct ncclTestOptions {
  int ranks = 8;
  int eager = 3;
  int graphs = 16;
  int delayCycles = 50000;
  bool inPlace = false;
  std::vector<ncclTestCase> cases = {
    {8, 0}, {1, 0}, {2, 1}, {7, 3}, {9, 5}, {14, 1}, {42, 2},
    {40960, 0}, {40962, 7}, {327680, 32}, {2621442, 3}
  };
};

static void ncclTestBarrier(pthread_barrier_t* barrier) {
  int code = pthread_barrier_wait(barrier);
  if (code != 0 && code != PTHREAD_BARRIER_SERIAL_THREAD) {
    fprintf(stderr, "pthread_barrier_wait: %s\n", strerror(code));
    exit(EXIT_FAILURE);
  }
}

__device__ unsigned char ncclTestValue(size_t index, unsigned long long epoch, int rank) {
  // Every byte is 1..113: neither payloads nor zero-initialized partial-pack
  // padding contain this protocol's reserved poison values.
  return static_cast<unsigned char>((index + (epoch % 113) * 3 + rank * 7) % 113 + 1);
}

__global__ void ncclTestAdvance(unsigned long long* epoch, int delayCycles) {
  ++*epoch;
  unsigned long long start = clock64();
  while (clock64() - start < static_cast<unsigned long long>(delayCycles)) {}
}

__global__ void ncclTestPrepare(unsigned char* input, unsigned char* output,
    unsigned long long* epoch, size_t bytes, size_t start,
    int rank, int ranks, bool inPlace) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = tid; i < start + bytes + NCCL_TEST_GUARD; i += stride) {
    input[i] = i >= start && i < start + bytes
        ? ncclTestValue(i - start, *epoch, rank) : NCCL_TEST_CANARY;
  }
  size_t ownStart = start + rank * bytes;
  for (size_t i = tid; i < start + ranks * bytes + NCCL_TEST_GUARD; i += stride) {
    output[i] = inPlace && i >= ownStart && i < ownStart + bytes
        ? ncclTestValue(i - ownStart, *epoch, rank) : NCCL_TEST_CANARY;
  }
}

__device__ void ncclTestCheckByte(ncclTestFailure* failure, unsigned char actual,
    unsigned char expected, size_t index, int region,
    int caseId, unsigned long long epoch) {
  if (actual != expected && atomicAdd(&failure->count, 1ull) == 0) {
    failure->epoch = epoch;
    failure->index = index;
    failure->caseId = caseId;
    failure->region = region;
    failure->actual = actual;
    failure->expected = expected;
  }
}

__global__ void ncclTestVerify(unsigned char* input, unsigned char* output,
    unsigned long long* epoch, ncclTestFailure* failure,
    size_t bytes, size_t start, int rank, int ranks, int caseId) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = tid; i < start + bytes + NCCL_TEST_GUARD; i += stride) {
    unsigned char expected = i >= start && i < start + bytes
        ? ncclTestValue(i - start, *epoch, rank) : NCCL_TEST_CANARY;
    ncclTestCheckByte(failure, input[i], expected, i, 0, caseId, *epoch);
  }
  for (size_t i = tid; i < start + ranks * bytes + NCCL_TEST_GUARD; i += stride) {
    unsigned char expected = NCCL_TEST_CANARY;
    if (i >= start && i < start + ranks * bytes) {
      size_t index = i - start;
      expected = ncclTestValue(index % bytes, *epoch, static_cast<int>(index / bytes));
    }
    ncclTestCheckByte(failure, output[i], expected, i, 1, caseId, *epoch);
  }
}

struct ncclTestRank {
  int rank;
  ncclComm_t comm;
  cudaStream_t stream;
  unsigned char* input;
  unsigned char* output;
  unsigned long long* epoch;
  ncclTestFailure* failure;
};

static void ncclTestEnqueue(ncclTestRank& state, const ncclTestOptions& options, int caseId) {
  auto test = options.cases[caseId];
  size_t start = NCCL_TEST_GUARD + test.offset;
  int ranks = options.ranks;
  int delay = state.rank == ranks - 1 ? options.delayCycles : 0;
  bool inPlace = options.inPlace;
  void* advanceArgs[] = {&state.epoch, &delay};
  CUDACHECK(cudaLaunchKernel(reinterpret_cast<const void*>(ncclTestAdvance),
          dim3(1), dim3(1), advanceArgs, 0, state.stream));
  void* prepareArgs[] = {&state.input, &state.output, &state.epoch, &test.bytes,
          &start, &state.rank, &ranks, &inPlace
      };
  CUDACHECK(cudaLaunchKernel(reinterpret_cast<const void*>(ncclTestPrepare),
          dim3(32), dim3(256), prepareArgs, 0, state.stream));
  unsigned char* send = inPlace ? state.output + start + state.rank * test.bytes
      : state.input + start;
  NCCLCHECK(ncclAllGather(send, state.output + start, test.bytes, ncclUint8,
          state.comm, state.stream));
  void* verifyArgs[] = {&state.input, &state.output, &state.epoch, &state.failure,
          &test.bytes, &start, &state.rank, &ranks, &caseId
      };
  CUDACHECK(cudaLaunchKernel(reinterpret_cast<const void*>(ncclTestVerify),
          dim3(32), dim3(256), verifyArgs, 0, state.stream));
}

static void ncclTestInspect(ncclTestRank& state, const ncclTestOptions& options,
    const char* phase) {
  CUDACHECK(cudaStreamSynchronize(state.stream));
  ncclTestFailure failure;
  CUDACHECK(cudaMemcpy(&failure, state.failure, sizeof(failure), cudaMemcpyDeviceToHost));
  if (failure.count != 0) {
    auto test = options.cases[failure.caseId];
    fprintf(stderr, "FAIL rank=%d phase=%s bytes=%zu offset=%zu epoch=%llu "
        "region=%s index=%zu actual=%u expected=%u mismatches=%llu\n",
        state.rank, phase, test.bytes, test.offset, failure.epoch,
        failure.region == 0 ? "input" : "output", failure.index,
        failure.actual, failure.expected, failure.count);
    // A failed rank must terminate the process, not strand other ranks at a
    // barrier. The external timeout additionally bounds kernel hangs.
    exit(EXIT_FAILURE);
  }
}

static void ncclTestRunRank(int rank, const ncclTestOptions& options, ncclUniqueId id,
    pthread_barrier_t* barrier) {
  CUDACHECK(cudaSetDevice(rank));
  ncclTestRank state{};
  state.rank = rank;
  NCCLCHECK(ncclCommInitRank(&state.comm, options.ranks, id, rank));
  CUDACHECK(cudaStreamCreateWithFlags(&state.stream, cudaStreamNonBlocking));
  size_t capacity = 0;
  for (auto test : options.cases) {
    capacity = std::max(capacity, test.offset + options.ranks * test.bytes + 2 * NCCL_TEST_GUARD);
  }
  // Equal allocation sizes and collective registration on every rank.
  NCCLCHECK(ncclMemAlloc(reinterpret_cast<void**>(&state.input), capacity));
  NCCLCHECK(ncclMemAlloc(reinterpret_cast<void**>(&state.output), capacity));
  NCCLCHECK(ncclMemAlloc(reinterpret_cast<void**>(&state.epoch), sizeof(*state.epoch)));
  NCCLCHECK(ncclMemAlloc(reinterpret_cast<void**>(&state.failure), sizeof(*state.failure)));
  // Initialization and every producer/verifier use the same nonblocking stream.
  CUDACHECK(cudaMemsetAsync(state.epoch, 0, sizeof(*state.epoch), state.stream));
  CUDACHECK(cudaMemsetAsync(state.failure, 0, sizeof(*state.failure), state.stream));
  ncclWindow_t inputWindow, outputWindow;
  NCCLCHECK(ncclCommWindowRegister(state.comm, state.input, capacity, &inputWindow,
          NCCL_WIN_COLL_SYMMETRIC));
  NCCLCHECK(ncclCommWindowRegister(state.comm, state.output, capacity, &outputWindow,
          NCCL_WIN_COLL_SYMMETRIC));
  for (size_t c = 0; c < options.cases.size(); ++c) {
    for (int repeat = 0; repeat < options.eager; ++repeat) {
      ncclTestBarrier(barrier);
      ncclTestEnqueue(state, options, static_cast<int>(c));
      ncclTestInspect(state, options, "eager");
    }
  }
  if (options.graphs > 0) {
    ncclTestBarrier(barrier);
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    CUDACHECK(cudaStreamBeginCapture(state.stream, cudaStreamCaptureModeThreadLocal));
    for (size_t c = 0; c < options.cases.size(); ++c) {
      ncclTestEnqueue(state, options, static_cast<int>(c));
    }
    CUDACHECK(cudaStreamEndCapture(state.stream, &graph));
    CUDACHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    ncclTestBarrier(barrier);
    // Intentionally queue all replays without CPU/rank barriers between
    // launches. Each call's GPU verifier runs before its buffers are reused.
    for (int repeat = 0; repeat < options.graphs; ++repeat) {
      CUDACHECK(cudaGraphLaunch(executable, state.stream));
    }
    ncclTestInspect(state, options, "queued-graphs");
    CUDACHECK(cudaGraphExecDestroy(executable));
    CUDACHECK(cudaGraphDestroy(graph));
  }
  unsigned long long epoch;
  CUDACHECK(cudaMemcpy(&epoch, state.epoch, sizeof(epoch), cudaMemcpyDeviceToHost));
  size_t checks = options.cases.size() * (options.eager + options.graphs);
  if (epoch != checks) {
    fprintf(stderr, "FAIL rank=%d epoch=%llu expected_calls=%zu\n", rank, epoch, checks);
    exit(EXIT_FAILURE);
  }
  printf("PASS rank=%d checks=%zu eager=%d graph_replays=%d in_place=%d\n",
      rank, checks, options.eager, options.graphs, options.inPlace);
  ncclTestBarrier(barrier);
  NCCLCHECK(ncclCommWindowDeregister(state.comm, outputWindow));
  NCCLCHECK(ncclCommWindowDeregister(state.comm, inputWindow));
  NCCLCHECK(ncclMemFree(state.failure));
  NCCLCHECK(ncclMemFree(state.epoch));
  NCCLCHECK(ncclMemFree(state.output));
  NCCLCHECK(ncclMemFree(state.input));
  CUDACHECK(cudaStreamDestroy(state.stream));
  NCCLCHECK(ncclCommDestroy(state.comm));
}

static long ncclTestNumber(const char* value, long low, long high) {
  errno = 0;
  char* end;
  long number = strtol(value, &end, 10);
  if (errno || end == value || *end || number < low || number > high) {
    fprintf(stderr, "Invalid numeric argument: %s (range %ld..%ld)\n", value, low, high);
    exit(EXIT_FAILURE);
  }
  return number;
}

static void ncclTestStopChildren(std::vector<pid_t>& children) {
  // Only unreaped children are tracked, so their PIDs cannot be reused.
  for (pid_t pid : children) {
    if (pid > 0 && kill(pid, SIGKILL) != 0 && errno != ESRCH) perror("kill");
  }
  for (pid_t& pid : children) {
    if (pid <= 0) continue;
    pid_t result;
    do { result = waitpid(pid, nullptr, 0); } while (result < 0 && errno == EINTR);
    if (result < 0) perror("waitpid during cleanup");
    pid = 0;
  }
}

int main(int argc, char** argv) {
  ncclTestOptions options;
  size_t bytes = 8, offset = 0;
  bool singleCase = false;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--in-place")) {
      options.inPlace = true;
    } else if (!strcmp(argv[i], "--help")) {
      printf("Usage: %s [--ranks N] [--bytes N --offset N] [--eager N] "
          "[--graphs N] [--delay-cycles N] [--in-place]\n", argv[0]);
      return 0;
    } else if (i + 1 < argc) {
      const char* key = argv[i++];
      if (!strcmp(key, "--ranks")) options.ranks = ncclTestNumber(argv[i], 2, 32);
      else if (!strcmp(key, "--eager")) options.eager = ncclTestNumber(argv[i], 1, 10000);
      else if (!strcmp(key, "--graphs")) options.graphs = ncclTestNumber(argv[i], 0, 10000);
      else if (!strcmp(key, "--delay-cycles"))
        options.delayCycles = ncclTestNumber(argv[i], 0, 10000000);
      else if (!strcmp(key, "--bytes")) {
        bytes = ncclTestNumber(argv[i], 1, 64 * 1024 * 1024);
        singleCase = true;
      } else if (!strcmp(key, "--offset")) {
        offset = ncclTestNumber(argv[i], 0, 4096);
        singleCase = true;
      } else {
        fprintf(stderr, "Unknown option: %s\n", key);
        return EXIT_FAILURE;
      }
    } else {
      fprintf(stderr, "Missing value or unknown option: %s\n", argv[i]);
      return EXIT_FAILURE;
    }
  }
  if (singleCase) options.cases = {{bytes, offset}};
  // Use one process/device to avoid the default single-process registration
  // fallback in this branch's symmetric scheduler.
  // Fork BEFORE any CUDA/NCCL call; no initialized CUDA state is inherited.
  struct ncclTestShared {
    ncclUniqueId id;
    pthread_barrier_t barrier;
  };
  auto* shared = static_cast<ncclTestShared*>(mmap(nullptr, sizeof(ncclTestShared),
              PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0));
  if (shared == MAP_FAILED) {
    perror("mmap");
    return EXIT_FAILURE;
  }
  pthread_barrierattr_t attr;
  int code = pthread_barrierattr_init(&attr);
  if (!code) code = pthread_barrierattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
  if (!code) code = pthread_barrier_init(&shared->barrier, &attr, options.ranks);
  if (!code) code = pthread_barrierattr_destroy(&attr);
  if (code) {
    fprintf(stderr, "Process-shared barrier initialization: %s\n", strerror(code));
    return EXIT_FAILURE;
  }
  std::vector<pid_t> children;
  for (int rank = 0; rank < options.ranks; ++rank) {
    pid_t child = fork();
    if (child == 0) {
      int devices, version;
      CUDACHECK(cudaGetDeviceCount(&devices));
      if (devices < options.ranks) {
        fprintf(stderr, "Need %d visible GPUs; found %d\n", options.ranks, devices);
        exit(EXIT_FAILURE);
      }
      CUDACHECK(cudaSetDevice(rank));
      if (rank == 0) {
        NCCLCHECK(ncclGetVersion(&version));
        NCCLCHECK(ncclGetUniqueId(&shared->id));
        printf("CONFIG nccl=%d ranks=%d cases=%zu eager=%d graph_replays=%d in_place=%d\n",
            version, options.ranks, options.cases.size(), options.eager, options.graphs,
            options.inPlace);
        fflush(stdout);
      }
      ncclTestBarrier(&shared->barrier);
      ncclTestRunRank(rank, options, shared->id, &shared->barrier);
      exit(EXIT_SUCCESS);
    }
    if (child < 0) {
      perror("fork");
      ncclTestStopChildren(children);
      return EXIT_FAILURE;
    }
    children.push_back(child);
  }
  for (int remaining = options.ranks; remaining > 0; --remaining) {
    int status;
    pid_t child;
    do { child = waitpid(-1, &status, 0); } while (child < 0 && errno == EINTR);
    if (child < 0) {
      perror("waitpid");
      ncclTestStopChildren(children);
      return EXIT_FAILURE;
    }
    for (pid_t& pid : children) if (pid == child) pid = 0;
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      ncclTestStopChildren(children);
      return EXIT_FAILURE;
    }
  }
  code = pthread_barrier_destroy(&shared->barrier);
  if (code) {
    fprintf(stderr, "pthread_barrier_destroy: %s\n", strerror(code));
    return EXIT_FAILURE;
  }
  if (munmap(shared, sizeof(*shared)) != 0) {
    perror("munmap");
    return EXIT_FAILURE;
  }
  printf("ALL_PASS ranks=%d total_checks=%zu\n", options.ranks,
      options.ranks * options.cases.size() * (options.eager + options.graphs));
  return 0;
}
