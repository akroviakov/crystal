// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <iostream>
#include <stdio.h>
#include <curand.h>

#include <cuda.h>
#include <cub/util_allocator.cuh>
#include "cub/test/test_util.h"

#include "crystal/crystal.cuh"

#include "gpu_utils.h"
#include "ssb_utils.h"

using namespace std;

/**
 * Globals, constants and typedefs
 */
bool                    g_verbose = false;  // Whether to display input/output to console
cub::CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

constexpr int batchSize{20000};

enum QueryVariant {
  Vector = 0,
  Compiled = 1,
  Compiled_Multi = 2
};

enum Parallelism {
  BatchToSM = 0,
  BatchToGPU = 1
};
/*
  OmniSci(HeavyDB) parallelism. https://github.com/heavyai/heavydb/blob/72c90bc290b79dd30240da41c103a00720f6b050/QueryEngine/JoinHashTable/Runtime/HashJoinRuntime.cpp#L779):
  Typical OmniSci BatchToGPU loop example:
  int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  int32_t step = blockDim.x * gridDim.x;
  or in terms of their LLVM:
  define i32 @pos_start_impl() {
    %threadIdx = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
    %blockIdx = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
    %blockDim = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
    %1 = mul nsw i32 %blockIdx, %blockDim
    %2 = add nsw i32 %threadIdx, %1
    ret i32 %2
  }

  define i32 @pos_step_impl() {
    %blockDim = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
    %gridDim = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
    %1 = mul nsw i32 %blockDim, %gridDim
    ret i32 %1
  }
*/

template<int BLOCK_THREADS, int ITEMS_PER_THREAD, Parallelism ParModel>
__global__ void QueryKernelCompiled(int* lo_orderdate, int* lo_discount, int* lo_quantity, int* lo_extendedprice,
    int lo_num_entries, unsigned long long* revenue, int batchId = -1, int numBatches = 1) {

  long long sum = 0;
  if constexpr (ParModel == Parallelism::BatchToSM){
    int blockIndex = (batchId == -1) ? blockIdx.x : batchId + blockIdx.x;
    int tile_offset = blockIndex * TILE_SIZE;
    int num_tiles = (lo_num_entries + TILE_SIZE - 1) / TILE_SIZE;
    int num_tile_items = TILE_SIZE;
    if (blockIndex == num_tiles - 1) {
      num_tile_items = lo_num_entries - tile_offset;
    }
    
    // Crystal parallelism:
    for(int i = 0; i < ITEMS_PER_THREAD; i++){
      if(threadIdx.x + i * BLOCK_THREADS < num_tile_items){
        int offset = tile_offset + threadIdx.x + BLOCK_THREADS * i;
        if(offset < lo_num_entries){
          if(lo_orderdate[offset] > 19930000 && lo_orderdate[offset] < 19940000 && 
              lo_quantity[offset] < 25 && lo_discount[offset] >= 1 && lo_discount[offset] <= 3){
            sum += lo_discount[offset] * lo_extendedprice[offset];
          }
        }
      }
    }
  } else { // Omnisci
    const int step{gridDim.x * blockDim.x}; // number of threads in kernel
    const int globalThreadIdx{threadIdx.x + blockIdx.x * blockDim.x};
    int batchStart = (batchId == -1) ? 0 : batchId * batchSize;
    const int limit = (batchId == -1) ? lo_num_entries : batchStart + batchSize * numBatches; // Multibatch kernel vs single batch kernel
    for(; batchStart < limit; batchStart += batchSize){
      for(int threadOffsetInBatch = globalThreadIdx; threadOffsetInBatch < batchSize; threadOffsetInBatch += step) {
        int offset = batchStart + threadOffsetInBatch;
        if(offset < lo_num_entries) {
          if(lo_orderdate[offset] > 19930000 && lo_orderdate[offset] < 19940000 && 
            lo_quantity[offset] < 25 && lo_discount[offset] >= 1 && lo_discount[offset] <= 3) {
            sum += lo_discount[offset] * lo_extendedprice[offset];
          }
        }
      }
    }
  }
  __syncthreads();
  static __shared__ long long buffer[32];
  unsigned long long aggregate = BlockSum<long long, BLOCK_THREADS, ITEMS_PER_THREAD>(sum, (long long*)buffer);
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(revenue, aggregate);
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void QueryKernel(int* lo_orderdate, int* lo_discount, int* lo_quantity, int* lo_extendedprice,
    int lo_num_entries, unsigned long long* revenue) {
  // Load a segment of consecutive items that are blocked across threads
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];

  long long sum = 0;

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (lo_num_entries + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = lo_num_entries - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_orderdate + tile_offset, items, num_tile_items);
  BlockPredGT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 19930000, selection_flags, num_tile_items);
  BlockPredAndLT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 19940000, selection_flags, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_quantity + tile_offset, items, num_tile_items);
  BlockPredAndLT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 25, selection_flags, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_discount + tile_offset, items, num_tile_items);
  BlockPredAndGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 1, selection_flags, num_tile_items);
  BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 3, selection_flags, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_extendedprice + tile_offset, items2, num_tile_items);

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if ((threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items))
      if (selection_flags[ITEM])
        sum += items[ITEM] * items2[ITEM];
  }

  __syncthreads();

  static __shared__ long long buffer[32];
  unsigned long long aggregate = BlockSum<long long, BLOCK_THREADS, ITEMS_PER_THREAD>(sum, (long long*)buffer);
  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(revenue, aggregate);
  }
}


template<QueryVariant QImpl, Parallelism ParModel = BatchToSM>
float runQuery(int* lo_orderdate, int* lo_discount, int* lo_quantity, int* lo_extendedprice, 
    int lo_num_entries, cub::CachingDeviceAllocator&  g_allocator) {
  SETUP_TIMING();

  float time_query;
  chrono::high_resolution_clock::time_point st, finish;
  st = chrono::high_resolution_clock::now();

  cudaEventRecord(start, 0);

  unsigned long long* d_sum = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_sum, sizeof(long long)));

  cudaMemset(d_sum, 0, sizeof(long long));

  // Run

  if constexpr(QImpl == QueryVariant::Vector){
    constexpr int numThreads{128};
    constexpr int elemPerThread = 4;
    constexpr int tile_items = numThreads*elemPerThread;
    constexpr int numBlocks = (LO_LEN + tile_items - 1)/tile_items;
    QueryKernel<numThreads,elemPerThread><<<numBlocks, numThreads>>>(lo_orderdate, 
          lo_discount, lo_quantity, lo_extendedprice, lo_num_entries, d_sum);
  } else {
    /* DATA INFO */
    constexpr int numBatches{(LO_LEN + batchSize - 1) / batchSize};
    /* END DATA INFO */

    /* KERNEL INFO */
    constexpr int SMEM_B_PER_BLOCK{256};
    constexpr int REG_B_PER_THREAD{16 * 32};
    /* END KERNEL INFO */

    /* DEVICE INFO */
    constexpr int SM_COUNT{56};
    constexpr int REG_FILE_B_PER_SM{256 * 1024};
    constexpr int SMEM_B_MAX_PER_SM{64 * 1024};
    constexpr int MAX_THREADS_PER_SM{2048};
    constexpr int MAX_BLOCKS_PER_SM{32};
    constexpr int MAX_BLOCK_SIZE{1024};
    /* END DEVICE INFO */

    /* CALCULATION INFO */
    constexpr int factor{3};
    constexpr int NUM_BLOCKS_PER_SM{SMEM_B_MAX_PER_SM/SMEM_B_PER_BLOCK};
    constexpr int NUM_BLOCKS_TOTAL{SM_COUNT * NUM_BLOCKS_PER_SM};
    constexpr int NUM_THREADS{REG_FILE_B_PER_SM / REG_B_PER_THREAD - 32};

    constexpr int blocksPerSM_SMEM = SMEM_B_MAX_PER_SM / SMEM_B_PER_BLOCK;
    constexpr int blocksPerSM_Threads = MAX_THREADS_PER_SM / MAX_BLOCK_SIZE;

    /* END CALCULATION INFO */
    // constexpr int numBlocks{11722};
    // constexpr int numThreads{128};
    // constexpr int elemPerThread{4};
    constexpr int numThreads{1024};
    constexpr int elemPerThread{batchSize / numThreads + 1};
    int blockSize;   // The launch configurator returned block size 
    int minGridSize; // The minimum grid size needed to achieve the 
                    // maximum occupancy for a full device launch 
    int gridSize;    // The actual grid size needed, based on input size 

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                QueryKernelCompiled<numThreads,elemPerThread,ParModel>, 0, 0); 
    gridSize = (LO_LEN + blockSize - 1) / blockSize; 
    cout << "GRID SIZE: " << gridSize << ", block size" << blockSize << "\n";
    int gridSize_{0}; 
    int blockSize_{0};
    if constexpr(ParModel == Parallelism::BatchToGPU){
      gridSize_ = gridSize;
      blockSize_ = blockSize;
    } else {
      gridSize_ = numBatches;
      blockSize_ = numThreads;
    }
    // Round up according to array size 
    if constexpr(QImpl == QueryVariant::Compiled){
      QueryKernelCompiled<numThreads,elemPerThread,ParModel><<<numBatches, numThreads>>>(lo_orderdate, 
          lo_discount, lo_quantity, lo_extendedprice, lo_num_entries, d_sum);
    } else {
      int batchId{0};
      constexpr int scheduleFactor{4};
      for(; batchId < numBatches; batchId+=scheduleFactor){
        QueryKernelCompiled<numThreads,elemPerThread,ParModel><<<scheduleFactor, numThreads>>>(lo_orderdate, 
          lo_discount, lo_quantity, lo_extendedprice, lo_num_entries, d_sum, batchId, scheduleFactor);
      }
      for(; batchId < numBatches; batchId++){
        QueryKernelCompiled<numThreads,elemPerThread,ParModel><<<1, numThreads>>>(lo_orderdate, 
          lo_discount, lo_quantity, lo_extendedprice, lo_num_entries, d_sum, batchId);
      }
    }
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_query, start,stop);

  unsigned long long revenue;
  CubDebugExit(cudaMemcpy(&revenue, d_sum, sizeof(long long), cudaMemcpyDeviceToHost));

  finish = chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = finish - st;

  // cout << "Num blocks: " << num_blocks << ", Tile size = " << tile_items << endl;
  cout << "Revenue: " << revenue << endl;
  cout << "Time Taken Total: " << diff.count() * 1000 << endl;

  CLEANUP(d_sum);

  return time_query;
}

/**
 * Main
 */
int main(int argc, char** argv)
{
  int num_trials          = 3;

  // Initialize command line
  CommandLineArgs args(argc, argv);
  args.GetCmdLineArgument("t", num_trials);

  // Print usage
  if (args.CheckCmdLineFlag("help"))
  {
      printf("%s "
          "[--t=<num trials>] "
          "[--v] "
          "\n", argv[0]);
      exit(0);
  }

  // Initialize device
  CubDebugExit(args.DeviceInit());

  int *h_lo_orderdate = loadColumn<int>("lo_orderdate", LO_LEN);
  int *h_lo_discount = loadColumn<int>("lo_discount", LO_LEN);
  int *h_lo_quantity = loadColumn<int>("lo_quantity", LO_LEN);
  int *h_lo_extendedprice = loadColumn<int>("lo_extendedprice", LO_LEN);
  int *h_d_datekey = loadColumn<int>("d_datekey", D_LEN);
  int *h_d_year = loadColumn<int>("d_year", D_LEN);

  cout << "** LOADED DATA **" << endl;
  cout << "LO_LEN " << LO_LEN << endl;

  int *d_lo_orderdate = loadToGPU<int>(h_lo_orderdate, LO_LEN, g_allocator);
  int *d_lo_discount = loadToGPU<int>(h_lo_discount, LO_LEN, g_allocator);
  int *d_lo_quantity = loadToGPU<int>(h_lo_quantity, LO_LEN, g_allocator);
  int *d_lo_extendedprice = loadToGPU<int>(h_lo_extendedprice, LO_LEN, g_allocator);
  int *d_d_datekey = loadToGPU<int>(h_d_datekey, D_LEN, g_allocator);
  int *d_d_year = loadToGPU<int>(h_d_year, D_LEN, g_allocator);

  cout << "** LOADED DATA TO GPU **" << endl;

  cout << "** COMPILED MULTI TEST **" << endl;
  for (int t = 0; t < num_trials+1; t++) {
    float time_query;
    time_query = runQuery<QueryVariant::Compiled_Multi>(d_lo_orderdate, d_lo_discount, d_lo_quantity, d_lo_extendedprice, LO_LEN, g_allocator);
    cout<< "{"
        << "\"type\":comp_multi" 
        << ",\"query\":11" 
        << ",\"time_query\":" << time_query
        << "}" << endl;
  }

  cout << "** COMPILED MULTI TEST (OMNISCI) **" << endl;
  for (int t = 0; t < num_trials+1; t++) {
    float time_query;
    time_query = runQuery<QueryVariant::Compiled_Multi, Parallelism::BatchToGPU>(d_lo_orderdate, d_lo_discount, d_lo_quantity, d_lo_extendedprice, LO_LEN, g_allocator);
    cout<< "{"
        << "\"type\":comp_multi_omnisci" 
        << ",\"query\":11" 
        << ",\"time_query\":" << time_query
        << "}" << endl;
  }

  cout << "** VECTOR TEST **" << endl;
  for (int t = 0; t < num_trials+1; t++) {
    float time_query;
    time_query = runQuery<QueryVariant::Vector>(d_lo_orderdate, d_lo_discount, d_lo_quantity, d_lo_extendedprice, LO_LEN, g_allocator);
    cout<< "{"
        << "\"type\":vec" 
        << ",\"query\":11" 
        << ",\"time_query\":" << time_query
        << "}" << endl;
  }

  cout << "** COMPILED TEST **" << endl;
  for (int t = 0; t < num_trials; t++) {
    float time_query;
    time_query = runQuery<QueryVariant::Compiled>(d_lo_orderdate, d_lo_discount, d_lo_quantity, d_lo_extendedprice, LO_LEN, g_allocator);
    cout<< "{"
        << "\"type\":comp" 
        << ",\"query\":11" 
        << ",\"time_query\":" << time_query
        << "}" << endl;
  }

  cout << "** COMPILED TEST (OMNISCI)**" << endl;
  for (int t = 0; t < num_trials; t++) {
    float time_query;
    time_query = runQuery<QueryVariant::Compiled, Parallelism::BatchToGPU>(d_lo_orderdate, d_lo_discount, d_lo_quantity, d_lo_extendedprice, LO_LEN, g_allocator);
    cout<< "{"
        << "\"type\":comp_omnisci" 
        << ",\"query\":11" 
        << ",\"time_query\":" << time_query
        << "}" << endl;
  }

  return 0;
}

