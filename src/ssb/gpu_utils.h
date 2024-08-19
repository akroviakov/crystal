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
