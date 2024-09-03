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

template<QueryVariant QImpl, Prefetch PrefetchSetting = Prefetch::NONE>
__global__ void DeviceSelectIfCompiled(int* lo_orderdate, int* lo_discount, int* lo_quantity, int* lo_extendedprice,
    int lo_num_entries, unsigned long long* revenue, int batchSize, unsigned long long* locals = nullptr) {
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
      if(orderdate > 19940204 && orderdate < 19940210){
        int quantity = lo_quantity[i];
        if(quantity >= 26 && quantity <= 35){
          int discount = lo_discount[i];
          if(discount >= 5 && discount<= 7){
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

template<int BLOCK_THREADS, int ITEMS_PER_THREAD, QueryVariant QImpl>
__global__ void DeviceSelectIf(int* lo_orderdate, int* lo_discount, int* lo_quantity, int* lo_extendedprice,
    int lo_num_entries, unsigned long long* revenue) {
  // Load a segment of consecutive items that are blocked across threads
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];

  unsigned long long sum = 0;

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (lo_num_entries + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = lo_num_entries - tile_offset;
  }
  if constexpr (QImpl == QueryVariant::Vector){
    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_orderdate + tile_offset, items, num_tile_items);
    BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 19940204, selection_flags, num_tile_items);
    BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 19940210, selection_flags, num_tile_items);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_quantity + tile_offset, items, num_tile_items);
    BlockPredAndGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 26, selection_flags, num_tile_items);
    BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 35, selection_flags, num_tile_items);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_discount + tile_offset, items, num_tile_items);
    BlockPredAndGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 5, selection_flags, num_tile_items);
    BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 7, selection_flags, num_tile_items);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_extendedprice + tile_offset, items2, num_tile_items);
  } else {
    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_orderdate + tile_offset,
                                                    items, num_tile_items);
    BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        items, 19940204, selection_flags, num_tile_items);
    BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        items, 19940210, selection_flags, num_tile_items);

    BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        lo_quantity + tile_offset, items, num_tile_items, selection_flags);
    BlockPredAndGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        items, 26, selection_flags, num_tile_items);
    BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        items, 35, selection_flags, num_tile_items);

    BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        lo_discount + tile_offset, items, num_tile_items, selection_flags);
    BlockPredAndGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        items, 5, selection_flags, num_tile_items);
    BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        items, 7, selection_flags, num_tile_items);

    BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        lo_extendedprice + tile_offset, items2, num_tile_items, selection_flags);
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


template<QueryVariant QImpl, Prefetch PrefSetting = Prefetch::NONE>
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
  unsigned long long* locals;

  // Run
  if constexpr(QImpl == QueryVariant::Vector || QImpl == QueryVariant::VectorOpt){
    int tile_items = 128*4;
    TIME_FUNC((DeviceSelectIf<128,4,QImpl><<<(lo_num_entries + tile_items - 1)/tile_items, 128>>>(lo_orderdate, 
            lo_discount, lo_quantity, lo_extendedprice, lo_num_entries, d_sum)), time_query);
  } else {
    const int batchSize = getBatchSizeCompiled<QImpl>();
    const int numBatches = (lo_num_entries + batchSize - 1)/batchSize;
    auto [gridSize, blockSize] = getLaunchConfigCompiled<QImpl>(DeviceSelectIfCompiled<QImpl>, getSMCount(), batchSize, numBatches);
    if constexpr(QImpl == QueryVariant::CompiledBatchToSMLocals){
      std::cerr << "WOW\n";
      CubDebugExit(g_allocator.DeviceAllocate((void**)&locals, gridSize*blockSize*sizeof(unsigned long long)));
    } 
    TIME_FUNC((DeviceSelectIfCompiled<QImpl, PrefSetting><<<gridSize, blockSize>>>(lo_orderdate, 
        lo_discount, lo_quantity, lo_extendedprice, lo_num_entries, d_sum, batchSize, locals)), time_query);
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

  int *h_lo_orderdate = loadColumn<int>(dataSetPath,"lo_orderdate", LO_LEN);
  int *h_lo_discount = loadColumn<int>(dataSetPath,"lo_discount", LO_LEN);
  int *h_lo_quantity = loadColumn<int>(dataSetPath,"lo_quantity", LO_LEN);
  int *h_lo_extendedprice = loadColumn<int>(dataSetPath,"lo_extendedprice", LO_LEN);
  int *h_d_datekey = loadColumn<int>(dataSetPath,"d_datekey", D_LEN);
  int *h_d_year = loadColumn<int>(dataSetPath,"d_year", D_LEN);

  cout << "** LOADED DATA **" << endl;

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
        << ",\"query\":13" 
        << ",\"time_query\":" << time_query
        << "}" << endl;
  }

  cout << "** VECTOR-OPT TEST **" << endl;
  for (int t = 0; t < num_trials; t++) {
    float time_query;
    time_query = runQuery<QueryVariant::VectorOpt>(d_lo_orderdate, d_lo_discount, d_lo_quantity, d_lo_extendedprice, LO_LEN, g_allocator);
    cout<< "{"
        << "\"type\":VectorOpt" 
        << ",\"query\":13" 
        << ",\"time_query\":" << time_query
        << "}" << endl;
  }

  cout << "** CompiledBatchToSM TEST **" << endl;
  for (int t = 0; t < num_trials; t++) {
    float time_query;
    time_query = runQuery<QueryVariant::CompiledBatchToSM>(d_lo_orderdate, d_lo_discount, d_lo_quantity, d_lo_extendedprice, LO_LEN, g_allocator);
    cout<< "{"
        << "\"type\":CompiledBatchToSM" 
        << ",\"query\":13" 
        << ",\"time_query\":" << time_query
        << "}" << endl;
  }
  // cout << "** CompiledBatchToSMPrefL2 TEST **" << endl;
  // for (int t = 0; t < num_trials; t++) {
  //   float time_query;
  //   time_query = runQuery<QueryVariant::CompiledBatchToSM, Prefetch::L2>(d_lo_orderdate, d_lo_discount, d_lo_quantity, d_lo_extendedprice, LO_LEN, g_allocator);
  //   cout<< "{"
  //       << "\"type\":CompiledBatchToSMPrefL2" 
  //       << ",\"query\":13" 
  //       << ",\"time_query\":" << time_query
  //       << "}" << endl;
  // }

  // cout << "** CompiledBatchToSMPrefLocals TEST **" << endl;
  // for (int t = 0; t < num_trials; t++) {
  //   float time_query;
  //   time_query = runQuery<QueryVariant::CompiledBatchToSMLocals>(d_lo_orderdate, d_lo_discount, d_lo_quantity, d_lo_extendedprice, LO_LEN, g_allocator);
  //   cout<< "{"
  //       << "\"type\":CompiledBatchToSMLocals" 
  //       << ",\"query\":13" 
  //       << ",\"time_query\":" << time_query
  //       << "}" << endl;
  // }
  cout << "** CompiledBatchToGPU  TEST **" << endl;
  for (int t = 0; t < num_trials; t++) {
    float time_query;
    time_query = runQuery<QueryVariant::CompiledBatchToGPU>(d_lo_orderdate, d_lo_discount, d_lo_quantity, d_lo_extendedprice, LO_LEN, g_allocator);
    cout<< "{"
        << "\"type\":CompiledBatchToGPU" 
        << ",\"query\":13" 
        << ",\"time_query\":" << time_query
        << "}" << endl;
  }
  return 0;
}
