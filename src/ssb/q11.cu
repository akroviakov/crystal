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

int externalBatchSize{0};

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

__device__ __forceinline__ int getSMemIndex(int threadId, int pdist, int numColumns, int columnId, int offset) {
    int baseIndex = threadId * pdist * numColumns;
    int columnBaseIndex = columnId * pdist;
    int smemIndex = baseIndex + columnBaseIndex + offset;
    return smemIndex;
}

template<QueryVariant QImpl, Prefetch PrefetchSetting=Prefetch::NONE>
__global__ void QueryKernelCompiled(const int* lo_orderdate, const int* lo_discount, const int* lo_quantity, const int* lo_extendedprice,
    int lo_num_entries, unsigned long long* revenue, const int batchSize, unsigned long long* locals = nullptr, int batchId = -1, int numBatches = 1) {

  unsigned long long sumd;
  unsigned long long* sum;
  if constexpr(QImpl == QueryVariant::CompiledBatchToSMLocals){
    const int globalTID = blockDim.x * blockIdx.x + threadIdx.x; 
    sum = &locals[globalTID];
  } else {
    sum = &sumd;
  }
  *sum = 0;

  const int startWithinBatch = getStart<QImpl>(batchSize);
  const int step = getStep<QImpl>();
  const int numBatchesToVisit = getNumBatches<QImpl>(lo_num_entries, batchSize); // 1 for QueryVariant::CompiledBatchToSM
  for(int batchId = 0; batchId < numBatchesToVisit; batchId++){
    int batchStart = getBatchStart<QImpl>(batchId, batchSize);
    const int limit = (batchStart + batchSize > lo_num_entries) ? lo_num_entries : batchStart + batchSize;
    int prefetchCursor{0};
    if constexpr (PrefetchSetting != Prefetch::NONE){
      prefetchPtrTo<PrefetchSetting>(&lo_orderdate[batchStart + startWithinBatch + (prefetchCursor++)*step]);
      prefetchPtrTo<PrefetchSetting>(&lo_orderdate[batchStart + startWithinBatch + (prefetchCursor++)*step]);
      prefetchPtrTo<PrefetchSetting>(&lo_orderdate[batchStart + startWithinBatch + (prefetchCursor++)*step]);
      prefetchPtrTo<PrefetchSetting>(&lo_orderdate[batchStart + startWithinBatch + (prefetchCursor++)*step]);
      prefetchPtrTo<PrefetchSetting>(&lo_orderdate[batchStart + startWithinBatch + (prefetchCursor++)*step]);
      prefetchPtrTo<PrefetchSetting>(&lo_orderdate[batchStart + startWithinBatch + (prefetchCursor++)*step]);
    }
    for(int i = batchStart + startWithinBatch; i < limit; i += step){
      if constexpr (PrefetchSetting != Prefetch::NONE){
        if((prefetchCursor + 1 ) * step < limit){
          prefetchPtrTo<PrefetchSetting>(&lo_orderdate[batchStart + startWithinBatch + (prefetchCursor++)*step]);
        }
      }
      int orderdate = lo_orderdate[i];
      if(orderdate > 19930000 && orderdate < 19940000){
        int quantity = lo_quantity[i];
        if(lo_quantity[i] < 25 ){
          int discount = lo_discount[i];
          if(discount >= 1 && discount <= 3){
            *sum += discount * lo_extendedprice[i];
          }
        }
      }
    }
  }
  __syncthreads();
  static __shared__ long long buffer[32];
  unsigned long long aggregate = BlockSum<long long>(*sum, (long long*)buffer);
  __syncthreads(); 
  if (threadIdx.x == 0) {
    atomicAdd(revenue, aggregate);
  }
}

template<QueryVariant Impl, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void QueryKernel(int* lo_orderdate, int* lo_discount, int* lo_quantity, int* lo_extendedprice,
    int lo_num_entries, unsigned long long* revenue) {

  __shared__ int shared_items[BLOCK_THREADS *(ITEMS_PER_THREAD+1)];
  __shared__ int shared_selection_flags[BLOCK_THREADS *(ITEMS_PER_THREAD+1)];
  __shared__ int shared_items2[BLOCK_THREADS *(ITEMS_PER_THREAD+1)];

  int local_items[ITEMS_PER_THREAD];
  int local_selection_flags[ITEMS_PER_THREAD];
  int local_items2[ITEMS_PER_THREAD];

  int* items;
  int* selection_flags;
  int* items2;

  // Load a segment of consecutive items that are blocked across threads
  if constexpr (Impl == QueryVariant::VectorSMEM || Impl == QueryVariant::VectorOptSMEM) {
    items = shared_items + threadIdx.x * (ITEMS_PER_THREAD+1);
    selection_flags = shared_selection_flags + threadIdx.x * (ITEMS_PER_THREAD+1);
    items2 = shared_items2 + threadIdx.x * (ITEMS_PER_THREAD+1);
  } else {
    items = local_items;
    selection_flags = local_selection_flags;
    items2 = local_items2;
  } 
  // Let the compiler decide which allocations to optimize away, we want convenient testing



  unsigned long long sum = 0;

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (lo_num_entries + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = lo_num_entries - tile_offset;
  }
  if constexpr(Impl == QueryVariant::Vector || Impl == QueryVariant::VectorSMEM){
    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_orderdate + tile_offset, items, num_tile_items);
    BlockPredGT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 19930000, selection_flags, num_tile_items);
    BlockPredAndLT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 19940000, selection_flags, num_tile_items);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_quantity + tile_offset, items, num_tile_items);
    BlockPredAndLT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 25, selection_flags, num_tile_items);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_discount + tile_offset, items, num_tile_items);
    BlockPredAndGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 1, selection_flags, num_tile_items);
    BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 3, selection_flags, num_tile_items);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_extendedprice + tile_offset, items2, num_tile_items);
  } else {
    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_orderdate + tile_offset,items, num_tile_items);
    BlockPredGT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 19930000, selection_flags, num_tile_items);
    BlockPredAndLT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 19940000, selection_flags, num_tile_items);

    BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_quantity + tile_offset, items, num_tile_items, selection_flags);
    BlockPredAndLT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 25, selection_flags, num_tile_items);

    BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_discount + tile_offset, items, num_tile_items, selection_flags);
    BlockPredAndGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 1, selection_flags, num_tile_items);
    BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 3, selection_flags, num_tile_items);

    BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_extendedprice + tile_offset, items2, num_tile_items, selection_flags);
  }
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


template<QueryVariant QImpl>
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
  unsigned long long* locals;
  if constexpr(QImpl == QueryVariant::Vector || QImpl == QueryVariant::VectorOpt || QImpl == QueryVariant::VectorSMEM || QImpl == QueryVariant::VectorOptSMEM){
    constexpr int numThreads{128};
    constexpr int elemPerThread = 4;
    constexpr int tile_items = numThreads*elemPerThread;
    constexpr int numBlocks = (LO_LEN + tile_items - 1)/tile_items;
    QueryKernel<QImpl, numThreads,elemPerThread><<<numBlocks, numThreads>>>(lo_orderdate, 
          lo_discount, lo_quantity, lo_extendedprice, lo_num_entries, d_sum);
  } else {
    /* DATA INFO */
    const int batchSize = externalBatchSize ? externalBatchSize : getBatchSizeCompiled<QImpl>();
    const int numBatches = (lo_num_entries + batchSize - 1)/batchSize;
    cout << "batchSize = " << batchSize << ", numBatches = " << numBatches << "\n";
    /* END DATA INFO */

    /* DEVICE INFO */
    constexpr int REG_FILE_B_PER_SM{256 * 1024};
    constexpr int SMEM_B_MAX_PER_SM{64 * 1024};
    constexpr int MAX_THREADS_PER_SM{2048};
    constexpr int MAX_BLOCKS_PER_SM{32};
    constexpr int MAX_BLOCK_SIZE{1024};
    /* END DEVICE INFO */

    // constexpr int numBlocks{11722};
    // constexpr int numThreads{128};
    // constexpr int elemPerThread{4};
    // Round up according to array size 
    auto [gridSize, blockSize] = getLaunchConfigCompiled<QImpl>(QueryKernelCompiled<QImpl>, getSMCount(), batchSize, numBatches);
    cout << "Launch config : <<<" << gridSize << ", " << blockSize << ">>>\n";

    if constexpr(QImpl == QueryVariant::CompiledBatchToSMLocals){
      CubDebugExit(g_allocator.DeviceAllocate((void**)&locals, gridSize*blockSize*sizeof(unsigned long long)));
    } 
    QueryKernelCompiled<QImpl><<<gridSize, blockSize>>>(lo_orderdate, lo_discount, lo_quantity, lo_extendedprice, lo_num_entries, d_sum, batchSize, locals);

    // } else {
    //   int batchId{0};
    //   constexpr int scheduleFactor{4};
    //   for(; batchId < numBatches; batchId+=scheduleFactor){
    //     QueryKernelCompiled<ParModel><<<scheduleFactor, numThreads>>>(lo_orderdate, 
    //       lo_discount, lo_quantity, lo_extendedprice, lo_num_entries, d_sum, batchSize, batchId, scheduleFactor);
    //   }
    //   for(; batchId < numBatches; batchId++){
    //     QueryKernelCompiled<ParModel><<<1, numThreads>>>(lo_orderdate, 
    //       lo_discount, lo_quantity, lo_extendedprice, lo_num_entries, d_sum, batchSize, batchId);
    //   }
    // }
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_query, start,stop);
  if constexpr(QImpl == QueryVariant::CompiledBatchToSMLocals){
    CLEANUP(locals);
  }
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
  int num_trials          = 10;
  // Initialize command line
  CommandLineArgs args(argc, argv);
  args.GetCmdLineArgument("t", num_trials);
  args.GetCmdLineArgument("batchSize", externalBatchSize);
  string dataSetPath;
  args.GetCmdLineArgument("dataSetPath", dataSetPath);

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
  int *h_lo_orderdate = loadColumn<int>(dataSetPath, "lo_orderdate", LO_LEN);
  int *h_lo_discount = loadColumn<int>(dataSetPath,"lo_discount", LO_LEN);
  int *h_lo_quantity = loadColumn<int>(dataSetPath,"lo_quantity", LO_LEN);
  int *h_lo_extendedprice = loadColumn<int>(dataSetPath,"lo_extendedprice", LO_LEN);
  int *h_d_datekey = loadColumn<int>(dataSetPath, "d_datekey", D_LEN);
  int *h_d_year = loadColumn<int>(dataSetPath, "d_year", D_LEN);

  cout << "** LOADED DATA **" << endl;
  cout << "LO_LEN " << LO_LEN << endl;

  int *d_lo_orderdate = loadToGPU<int>(h_lo_orderdate, LO_LEN, g_allocator);
  int *d_lo_discount = loadToGPU<int>(h_lo_discount, LO_LEN, g_allocator);
  int *d_lo_quantity = loadToGPU<int>(h_lo_quantity, LO_LEN, g_allocator);
  int *d_lo_extendedprice = loadToGPU<int>(h_lo_extendedprice, LO_LEN, g_allocator);
  int *d_d_datekey = loadToGPU<int>(h_d_datekey, D_LEN, g_allocator);
  int *d_d_year = loadToGPU<int>(h_d_year, D_LEN, g_allocator);

  cout << "** LOADED DATA TO GPU **" << endl;

  cout << "** VECTOR TEST **" << endl;
  for (int t = 0; t < num_trials; t++) {
    float time_query;
    time_query = runQuery<QueryVariant::Vector>(d_lo_orderdate, d_lo_discount, d_lo_quantity, d_lo_extendedprice, LO_LEN, g_allocator);
    cout<< "{"
        << "\"type\":Vector" 
        << ",\"query\":11" 
        << ",\"time_query\":" << time_query
        << ",\"batch_size\":" << externalBatchSize
        << "}" << endl;
  }
  cout << "** VECTOR-OPT TEST **" << endl;
  for (int t = 0; t < num_trials; t++) {
    float time_query;
    time_query = runQuery<QueryVariant::VectorOpt>(d_lo_orderdate, d_lo_discount, d_lo_quantity, d_lo_extendedprice, LO_LEN, g_allocator);
    cout<< "{"
        << "\"type\":VectorOpt" 
        << ",\"query\":11" 
        << ",\"time_query\":" << time_query
        << ",\"batch_size\":" << externalBatchSize
        << "}" << endl;
  }

  cout << "** VectorSMEM TEST **" << endl;
  for (int t = 0; t < num_trials; t++) {
    float time_query;
    time_query = runQuery<QueryVariant::VectorSMEM>(d_lo_orderdate, d_lo_discount, d_lo_quantity, d_lo_extendedprice, LO_LEN, g_allocator);
    cout<< "{"
        << "\"type\":VectorSMEM" 
        << ",\"query\":11" 
        << ",\"time_query\":" << time_query
        << ",\"batch_size\":" << externalBatchSize
        << "}" << endl;
  }

  cout << "** VectorOptSMEM TEST **" << endl;
  for (int t = 0; t < num_trials; t++) {
    float time_query;
    time_query = runQuery<QueryVariant::VectorOptSMEM>(d_lo_orderdate, d_lo_discount, d_lo_quantity, d_lo_extendedprice, LO_LEN, g_allocator);
    cout<< "{"
        << "\"type\":VectorOptSMEM" 
        << ",\"query\":11" 
        << ",\"time_query\":" << time_query
        << ",\"batch_size\":" << externalBatchSize
        << "}" << endl;
  }

  cout << "** CompiledBatchToSM TEST **" << endl;
  for (int t = 0; t < num_trials; t++) {
    float time_query;
    time_query = runQuery<QueryVariant::CompiledBatchToSM>(d_lo_orderdate, d_lo_discount, d_lo_quantity, d_lo_extendedprice, LO_LEN, g_allocator);
    cout<< "{"
        << "\"type\":CompiledBatchToSM" 
        << ",\"query\":11" 
        << ",\"time_query\":" << time_query
        << ",\"batch_size\":" << externalBatchSize
        << "}" << endl;
  }
  
  // cout << "** CompiledBatchToSMLocals TEST **" << endl;
  // for (int t = 0; t < num_trials; t++) {
  //   float time_query;
  //   time_query = runQuery<QueryVariant::CompiledBatchToSMLocals>(d_lo_orderdate, d_lo_discount, d_lo_quantity, d_lo_extendedprice, LO_LEN, g_allocator);
  //   cout<< "{"
  //       << "\"type\":CompiledBatchToSMLocals" 
  //       << ",\"query\":11" 
  //       << ",\"time_query\":" << time_query
  //       << ",\"batch_size\":" << externalBatchSize
  //       << "}" << endl;
  // }

  cout << "** CompiledBatchToGPU TEST **" << endl;
  for (int t = 0; t < num_trials; t++) {
    float time_query;
    time_query = runQuery<QueryVariant::CompiledBatchToGPU>(d_lo_orderdate, d_lo_discount, d_lo_quantity, d_lo_extendedprice, LO_LEN, g_allocator);
    cout<< "{"
        << "\"type\":CompiledBatchToGPU" 
        << ",\"query\":11" 
        << ",\"time_query\":" << time_query
        << ",\"batch_size\":" << externalBatchSize
        << "}" << endl;
  }

  return 0;
}

