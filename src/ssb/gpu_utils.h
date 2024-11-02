#pragma once

#define SETUP_TIMING() cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);

#define TIME_FUNC(f,t) { \
    cudaEventRecord(start, 0); \
    f; \
    cudaEventRecord(stop, 0); \
    cudaEventSynchronize(stop); \
    cudaEventElapsedTime(&t, start,stop); \
}

#define CLEANUP(vec) if(vec)CubDebugExit(g_allocator.DeviceFree(vec))

#define ALLOCATE(vec,size) CubDebugExit(g_allocator.DeviceAllocate((void**)&vec, size))

__device__ inline void prefetch_l1(const void *p) {
  asm("prefetch.global.L1 [%0];" : : "l"(p));
}

__device__ inline int noncached_read(const int* ptr) {
    int value;
    asm("ld.global.nc.s32 %0, [%1];" : "=r"(value) : "l"(ptr));
    return value;
}

__device__ inline void prefetch_l2(const void *p) {
  asm("prefetch.global.L2 [%0];" : : "l"(p));
}

enum QueryVariant {
  Vector = 0,
  VectorOpt = 1,
  CompiledBatchToSM = 2,
  CompiledBatchToSMLocals = 3,
  CompiledBatchToGPU = 4,
  VectorSMEM = 5,
  VectorOptSMEM = 6
};

enum Prefetch {
  NONE = 0,
  L1 = 1,
  L2 = 2
};

enum ReadCacheMode {
  PERSIST = 0,
  STREAM = 1
};

template<ReadCacheMode CacheMode = PERSIST>
__device__ inline int readAs(const int *p) {
  if constexpr(CacheMode == ReadCacheMode::STREAM)
    return noncached_read(p);
  else 
    return *p;
}


template<Prefetch PrefSetting>
__device__ inline void prefetchPtrTo(const void *p) {
  // Prefetch *cache line* containing address
  if constexpr(PrefSetting == Prefetch::L1){
    prefetch_l1(p); // Any different from L2?
  } else {
    prefetch_l2(p);
  }
}


int getSMCount() {
    int device;
    CubDebugExit(cudaGetDevice(&device)); 
    int numSMs;
    CubDebugExit(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device));
    return numSMs;
}

template<QueryVariant QImpl>
__device__ inline int getStart(const int batchSize) {
  int start;
  if constexpr (QImpl == QueryVariant::CompiledBatchToSM || QImpl == QueryVariant::CompiledBatchToSMLocals){
    start = threadIdx.x; // offset within batch
  } else {
    start = blockIdx.x * blockDim.x + threadIdx.x; // global TID
  }
  return start;
}

template<QueryVariant QImpl>
__device__ inline int getStep() {
  int step;
  if constexpr (QImpl == QueryVariant::CompiledBatchToSM || QImpl == QueryVariant::CompiledBatchToSMLocals){
    step = blockDim.x;
  } else {
    step = gridDim.x * blockDim.x;
  }
  return step;
}

template<QueryVariant QImpl>
__device__ inline int getNumBatches(const int colLimit, const int batchSize) {
  int numIters;
  if constexpr (QImpl == QueryVariant::CompiledBatchToSM || QImpl == QueryVariant::CompiledBatchToSMLocals){
    numIters = 1; // Scheduler will "schedule" a batch
  } else {
    numIters = (colLimit + batchSize - 1)/batchSize;
  }
  return numIters;
}

template<QueryVariant QImpl>
__device__ inline int getBatchStart(const int batchId, const int batchSize) {
  int batchStart;
  if constexpr (QImpl == QueryVariant::CompiledBatchToSM || QImpl == QueryVariant::CompiledBatchToSMLocals){
    batchStart = blockIdx.x * batchSize; // Scheduler will "schedule" a batch
  } else {
    batchStart = batchId * batchSize;
  }
  return batchStart;
}

__device__ inline int getLimit(const int batchSize, const int start, const int colLimit) {
  int limit;
  int currentEnd = start + batchSize;
  if(currentEnd < colLimit){
    limit = currentEnd;
  } else {
    limit = colLimit;
  }
  return limit;
}

template<QueryVariant QImpl>
__host__ inline int getBatchSizeCompiled() {
  int batchSize;
  if constexpr (QImpl == QueryVariant::CompiledBatchToSM || QImpl == QueryVariant::CompiledBatchToSMLocals){
    batchSize = 20000;
  } else {
    batchSize = 32'000'000;
  } 
  return batchSize;
}

template<QueryVariant QImpl, class T>
__host__ inline std::pair<int, int> getLaunchConfigCompiled(T kernel, const int numSMs, const int batchSize, const int numBatches) {
  int gridSize, blockSize;
  if constexpr (QImpl == QueryVariant::CompiledBatchToSM || QImpl == QueryVariant::CompiledBatchToSMLocals){
    gridSize = numBatches;
    blockSize = std::min(batchSize, 1024);
  } else {
    // Compare against CompiledBatchToSM
    gridSize = (batchSize * numBatches) / getBatchSizeCompiled<QueryVariant::CompiledBatchToSM>();
    blockSize = std::min(batchSize, 1024);
    // HeavyDB way:
    // cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, kernel, 0, 0); 
    // // Adjust for more *possible* parallelism
    // if(gridSize == numSMs){
    //   gridSize *= 100;
    // }
  } 
  return {gridSize, blockSize};
}

template<QueryVariant QImpl, typename Func>
__device__ void scanLoop(int numTuplesTotal, int batchSize, Func body) {
    const int startWithinBatch = getStart<QImpl>(batchSize);
    const int step = getStep<QImpl>();
    const int numBatchesToVisit = getNumBatches<QImpl>(numTuplesTotal, batchSize); // 1 for QueryVariant::CompiledBatchToSM
    for(int batchId = 0; batchId < numBatchesToVisit; batchId++) {
        int batchStart = getBatchStart<QImpl>(batchId, batchSize);
        const int limit = (batchStart + batchSize > numTuplesTotal) ? numTuplesTotal : batchStart + batchSize;
        for(int offset = batchStart + startWithinBatch; offset < limit; offset += step) {
          body(offset); 
        }
    }
}




template<typename T>
T* loadToGPU(T* src, int numEntries, cub::CachingDeviceAllocator& g_allocator) {
  T* dest;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&dest, sizeof(T) * numEntries));
  CubDebugExit(cudaMemcpy(dest, src, sizeof(T) * numEntries, cudaMemcpyHostToDevice));
  return dest;
}

#define TILE_SIZE (BLOCK_THREADS * ITEMS_PER_THREAD)

#define CHECK_ERROR() { \
  cudaDeviceSynchronize(); \
  cudaError_t error = cudaGetLastError(); \
  if(error != cudaSuccess) \
  { \
    printf("CUDA error: %s\n", cudaGetErrorString(error)); \
    exit(-1); \
  } \
}
