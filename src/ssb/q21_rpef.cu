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

constexpr int PDIST{3};
#define vsmem(index,col)  orderDate[index+(PDIST+col)*threadIdx.x]
#define GET_PREFETCHED_VAL(smem, currPrefIdx, col, numCols) \
    (smem[threadIdx.x * PDIST * numCols + (col * PDIST) + currPrefIdx])
// __device__ __forceinline__ int& getPrefetchedVal(const int* smem, int currPrefIdx, int col, int numCols){
//   return smem[threadIdx.x*(PDIST+numCols) + currPrefIdx + col];
// }

__device__ __forceinline__ int getSMemIndex(int threadId, int pdist, int numColumns, int columnId, int offset) {
    int baseIndex = threadId * pdist * numColumns;
    int columnBaseIndex = columnId * pdist;
    int smemIndex = baseIndex + columnBaseIndex + offset;
    return smemIndex;
}

__global__ void probeCompiled(int* lo_orderdate, int* lo_partkey, int* lo_suppkey, int* lo_revenue, int lo_len,
    int* ht_s, int s_len,
    int* ht_p, int p_len,
    int* ht_d, int d_len,
    int* res, const int batchSize) {
  // Load a tile striped across threads

  const int threadsInBlock{blockDim.x};
  const int numRowsPerThread{(batchSize + threadsInBlock - 1) / threadsInBlock};
  const int blockIndex =  blockIdx.x;
  const int batchOffset = blockIndex * batchSize;
  const int numBatches = (lo_len + batchSize - 1) / batchSize;
  const int numBatchRows = (blockIndex == numBatches - 1) ? lo_len - batchOffset : batchSize;
  
  int k{0};

  __shared__ int orderDate[2*PDIST* 1024];
  for (k=0; k<PDIST; ++k) { 
    orderDate[getSMemIndex(threadIdx.x, PDIST, 2, 0, k)] = noncached_read(&lo_suppkey[batchOffset + threadIdx.x + k * threadsInBlock]);
    orderDate[getSMemIndex(threadIdx.x, PDIST, 2, 1, k)] = noncached_read(&lo_partkey[batchOffset + threadIdx.x + k * threadsInBlock]);
  }

  for (int i = 0; i < numRowsPerThread; i++, k++) {
    int ctr_mod= i%PDIST;
    int suppkey = orderDate[getSMemIndex(threadIdx.x, PDIST, 2, 0, ctr_mod)]; // vsmem(ctr_mod,0);
    int partkey = orderDate[getSMemIndex(threadIdx.x, PDIST, 2, 1, ctr_mod)];
    if(k < numRowsPerThread){
      orderDate[getSMemIndex(threadIdx.x, PDIST, 2, 0, ctr_mod)] = noncached_read(&lo_suppkey[batchOffset + threadIdx.x + k * threadsInBlock]);
      orderDate[getSMemIndex(threadIdx.x, PDIST, 2, 1, ctr_mod)] = noncached_read(&lo_partkey[batchOffset + threadIdx.x + k * threadsInBlock]);
    }
    const int localIndex = threadIdx.x + i * threadsInBlock;
    if(localIndex < numBatchRows){
      int offset = batchOffset + localIndex;
      int hash_s = HASH(lo_suppkey[offset], s_len, 0);
      if(ht_s[hash_s]){
        int hash_p = HASH(lo_partkey[offset], p_len, 0);
        uint64_t slot = *reinterpret_cast<uint64_t*>(&ht_p[hash_p << 1]);
        if(slot){
          int brand = (slot >> 32);
          int hash_d = HASH(lo_orderdate[offset], d_len, 19920101);
          uint64_t slot_d = *reinterpret_cast<uint64_t*>(&ht_d[hash_d << 1]);
          if(slot_d){
            int year = (slot_d >> 32);
            int hash_res = (brand * 7 + (year - 1992)) % ((1998-1992+1) * (5*5*40));
            res[hash_res * 4] = year;
            res[hash_res * 4 + 1] = brand;
            atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash_res * 4 + 2]), (long long)(lo_revenue[offset]));
          }
        }
      }
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe(int* lo_orderdate, int* lo_partkey, int* lo_suppkey, int* lo_revenue, int lo_len,
    int* ht_s, int s_len,
    int* ht_p, int p_len,
    int* ht_d, int d_len,
    int* res) {
  // Load a tile striped across threads
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int brand[ITEMS_PER_THREAD];
  int year[ITEMS_PER_THREAD];
  int revenue[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (lo_len + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = lo_len - tile_offset;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  // BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_suppkey + tile_offset, items, num_tile_items);
  // BlockProbeAndPHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, selection_flags, ht_s, s_len, num_tile_items);

  // BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_partkey + tile_offset, items, num_tile_items);
  // BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, brand, selection_flags,
  //     ht_p, p_len, num_tile_items);

  // BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_orderdate + tile_offset, items, num_tile_items);
  // BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, year, selection_flags,
  //     ht_d, d_len, 19920101, num_tile_items);

  // BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_revenue + tile_offset, revenue, num_tile_items);


  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_suppkey + tile_offset,
                                                  items, num_tile_items);
  BlockProbeAndPHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, selection_flags, ht_s, s_len, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags)) { return; }

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_partkey + tile_offset, items, num_tile_items, selection_flags);
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, brand, selection_flags, ht_p, p_len, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags)) { return; }

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_orderdate + tile_offset, items, num_tile_items, selection_flags);
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, year, selection_flags, ht_d, d_len, 19920101, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags)) { return; }

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_revenue + tile_offset, revenue, num_tile_items, selection_flags);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags)) { return; }

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS * ITEM)) < num_tile_items) {
      if (selection_flags[ITEM]) {
        int hash = (brand[ITEM] * 7 +  (year[ITEM] - 1992)) % ((1998-1992+1) * (5*5*40));
        res[hash * 4] = year[ITEM];
        res[hash * 4 + 1] = brand[ITEM];
        atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 4 + 2]), (long long)(revenue[ITEM]));
      }
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_s_Compiled(int *filter_col, int *dim_key, int num_tuples, int *hash_table, int num_slots) {
  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  for(int i = 0; i < ITEMS_PER_THREAD; i++){
    if(threadIdx.x + i * BLOCK_THREADS < num_tile_items){
      int offset = tile_offset + threadIdx.x + BLOCK_THREADS * i;
      if(offset < num_tuples && filter_col[offset] == 1){
        int hash = HASH(dim_key[offset], num_slots, 0);
        atomicCAS(&hash_table[hash], 0, dim_key[offset]);
      }
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_p_Compiled(int *filter_col, int *dim_key, int *dim_val, int num_tuples, int *hash_table, int num_slots) {
  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  for(int i = 0; i < ITEMS_PER_THREAD; i++){
    if(threadIdx.x + i * BLOCK_THREADS < num_tile_items){
      int offset = tile_offset + threadIdx.x + BLOCK_THREADS * i;
      if(offset < num_tuples && filter_col[offset] == 1){
        int hash = HASH(dim_key[offset], num_slots, 0);
        atomicCAS(&hash_table[hash << 1], 0, dim_key[offset]);
        hash_table[(hash << 1) + 1] = dim_val[offset];
      }
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_d_Compiled(int *dim_key, int *dim_val, int num_tuples, int *hash_table, int num_slots, int val_min) {
  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  for(int i = 0; i < ITEMS_PER_THREAD; i++){
    if(threadIdx.x + i * BLOCK_THREADS < num_tile_items){
      int offset = tile_offset + threadIdx.x + BLOCK_THREADS * i;
      int hash = HASH(dim_key[offset], num_slots, val_min);
      atomicCAS(&hash_table[hash << 1], 0, dim_key[offset]);
      hash_table[(hash << 1) + 1] = dim_val[offset];
    }
  }
}


template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_s(int *filter_col, int *dim_key, int num_tuples, int *hash_table, int num_slots) {
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col + tile_offset, items, num_tile_items);
  BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 1, selection_flags, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items, num_tile_items);
  BlockBuildSelectivePHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, selection_flags, 
      hash_table, num_slots, num_tile_items);
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_p(int *filter_col, int *dim_key, int *dim_val, int num_tuples, int *hash_table, int num_slots) {
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col + tile_offset, items, num_tile_items);
  BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 1, selection_flags, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items, num_tile_items);
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items2, num_tile_items);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, items2, selection_flags, 
      hash_table, num_slots, num_tile_items);
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_d(int *dim_key, int *dim_val, int num_tuples, int *hash_table, int num_slots, int val_min) {
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items, num_tile_items);
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items2, num_tile_items);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, items2, selection_flags, 
      hash_table, num_slots, val_min, num_tile_items);
}

enum QueryVariant {
    Vector = 0,
    Compiled = 1,
    Compiled_Multi = 2
};

template<QueryVariant QImpl>
float runQuery(int* lo_orderdate, int* lo_partkey, int* lo_suppkey, int* lo_revenue, int lo_len,
    int* p_partkey, int* p_brand1, int* p_category, int p_len,
    int *d_datekey, int* d_year, int d_len,
    int *s_suppkey, int* s_region, int s_len,
    cub::CachingDeviceAllocator&  g_allocator) {
  SETUP_TIMING();
  cudaEvent_t stop_build; cudaEventCreate(&stop_build);

  float time_query;
  float time_build;
  chrono::high_resolution_clock::time_point st, finish;
  st = chrono::high_resolution_clock::now();
  cudaEventRecord(start, 0);

  int *ht_d, *ht_p, *ht_s;
  int d_val_len = 19981230 - 19920101 + 1;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&ht_d, 2 * d_val_len * sizeof(int)));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&ht_p, 2 * p_len * sizeof(int)));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&ht_s, 2 * s_len * sizeof(int)));

  CubDebugExit(cudaMemset(ht_d, 0, 2 * d_val_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_p, 0, 2 * p_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_s, 0, 2 * s_len * sizeof(int)));

  int tile_items = 128*4;
  int d_val_min = 19920101;

  if constexpr(QImpl == QueryVariant::Vector){
      build_hashtable_s<128,4><<<(s_len + tile_items - 1)/tile_items, 128>>>(s_region, s_suppkey, s_len, ht_s, s_len);
    /*CHECK_ERROR();*/
      build_hashtable_p<128,4><<<(p_len + tile_items - 1)/tile_items, 128>>>(p_category, p_partkey, p_brand1, p_len, ht_p, p_len);
    /*CHECK_ERROR();*/
      build_hashtable_d<128,4><<<(d_len + tile_items - 1)/tile_items, 128>>>(
        d_datekey, d_year, d_len, ht_d, d_val_len, d_val_min);
    /*CHECK_ERROR();*/
  } else {
    if constexpr(QImpl == QueryVariant::Compiled){
      // build_hashtable_s_Compiled<128,4><<<(s_len + tile_items - 1)/tile_items, 128>>>(s_region, s_suppkey, s_len, ht_s, s_len);
      // build_hashtable_p_Compiled<128,4><<<(p_len + tile_items - 1)/tile_items, 128>>>(p_category, p_partkey, p_brand1, p_len, ht_p, p_len);
      // build_hashtable_d_Compiled<128,4><<<(d_len + tile_items - 1)/tile_items, 128>>>(d_datekey, d_year, d_len, ht_d, d_val_len, d_val_min);
      
      constexpr int batchSize{20000};
      constexpr int numBatchesS{(S_LEN + batchSize - 1) / batchSize};
      constexpr int numBatchesP{(P_LEN + batchSize - 1) / batchSize};
      constexpr int numBatchesD{(D_LEN + batchSize - 1) / batchSize};
      constexpr int numThreads{1024};
      constexpr int elemPerThread{batchSize / numThreads + 1};
      build_hashtable_s_Compiled<numThreads,elemPerThread><<<numBatchesS, numThreads>>>(s_region, s_suppkey, s_len, ht_s, s_len);
      build_hashtable_p_Compiled<numThreads,elemPerThread><<<numBatchesP, numThreads>>>(p_category, p_partkey, p_brand1, p_len, ht_p, p_len);
      build_hashtable_d_Compiled<numThreads,elemPerThread><<<numBatchesD, numThreads>>>(d_datekey, d_year, d_len, ht_d, d_val_len, d_val_min);
    }
  }
  cudaEventRecord(stop_build, 0);
  cudaEventSynchronize(stop_build);
  cudaEventElapsedTime(&time_build, start, stop_build);

  int *res;
  int res_size = ((1998-1992+1) * (5 * 5 * 40));
  int res_array_size = res_size * 4;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&res, res_array_size * sizeof(int)));

  CubDebugExit(cudaMemset(res, 0, res_array_size * sizeof(int)));
  if constexpr(QImpl == QueryVariant::Vector){
    probe<128,4><<<(lo_len + tile_items - 1)/tile_items, 128>>>(lo_orderdate,
          lo_partkey, lo_suppkey, lo_revenue, lo_len, ht_s, s_len, ht_p, p_len, ht_d, d_val_len, res);
  } else {
    if constexpr(QImpl == QueryVariant::Compiled){
      constexpr int batchSize{20000};
      constexpr int numBatches{(LO_LEN + batchSize - 1) / batchSize};
      constexpr int numThreads{1024};
      constexpr int elemPerThread{batchSize / numThreads + 1};
      probeCompiled<<<numBatches, numThreads>>>(lo_orderdate,
          lo_partkey, lo_suppkey, lo_revenue, lo_len, ht_s, s_len, ht_p, p_len, ht_d, d_val_len, res, 20000);
      // probeCompiled<128,4><<<(lo_len + tile_items - 1)/tile_items, 128>>>(lo_orderdate,
          // lo_partkey, lo_suppkey, lo_revenue, lo_len, ht_s, s_len, ht_p, p_len, ht_d, d_val_len, res);
    }
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_query, start, stop);

  int* h_res = new int[res_array_size];
  CubDebugExit(cudaMemcpy(h_res, res, res_array_size * sizeof(int), cudaMemcpyDeviceToHost));
  finish = chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = finish - st;

  int res_count = 0;
  for (int i=0; i<res_size; i++) {
    if (h_res[4*i] != 0) {
      cout << h_res[4*i] << " " << h_res[4*i + 1] << " " << reinterpret_cast<unsigned long long*>(&h_res[4*i + 2])[0]  << endl;
      res_count += 1;
    }
  }

  cout << "Res Count: " << res_count << endl;
  cout << "Time Build: " << time_build << endl;
  cout << "Time Taken Total: " << diff.count() * 1000 << endl;

  delete[] h_res;

  CLEANUP(res);
  CLEANUP(ht_d);
  CLEANUP(ht_p);
  CLEANUP(ht_s);

  return time_query;
}

/**
 * Main
 */
int main(int argc, char** argv)
{
  int num_trials          = 8;

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
  int *h_lo_partkey = loadColumn<int>(dataSetPath,"lo_partkey", LO_LEN);
  int *h_lo_suppkey = loadColumn<int>(dataSetPath,"lo_suppkey", LO_LEN);
  int *h_lo_revenue = loadColumn<int>(dataSetPath,"lo_revenue", LO_LEN);

  int *h_p_partkey = loadColumn<int>(dataSetPath,"p_partkey", P_LEN);
  int *h_p_brand1 = loadColumn<int>(dataSetPath,"p_brand1", P_LEN);
  int *h_p_category = loadColumn<int>(dataSetPath,"p_category", P_LEN);

  int *h_d_datekey = loadColumn<int>(dataSetPath,"d_datekey", D_LEN);
  int *h_d_year = loadColumn<int>(dataSetPath,"d_year", D_LEN);

  int *h_s_suppkey = loadColumn<int>(dataSetPath,"s_suppkey", S_LEN);
  int *h_s_region = loadColumn<int>(dataSetPath,"s_region", S_LEN);

  int *d_lo_orderdate = loadToGPU<int>(h_lo_orderdate, LO_LEN, g_allocator);
  int *d_lo_partkey = loadToGPU<int>(h_lo_partkey, LO_LEN, g_allocator);
  int *d_lo_suppkey = loadToGPU<int>(h_lo_suppkey, LO_LEN, g_allocator);
  int *d_lo_revenue = loadToGPU<int>(h_lo_revenue, LO_LEN, g_allocator);

  int *d_d_datekey = loadToGPU<int>(h_d_datekey, D_LEN, g_allocator);
  int *d_d_year = loadToGPU<int>(h_d_year, D_LEN, g_allocator);

  int *d_p_partkey = loadToGPU<int>(h_p_partkey, P_LEN, g_allocator);
  int *d_p_brand1  = loadToGPU<int>(h_p_brand1, P_LEN, g_allocator);
  int *d_p_category = loadToGPU<int>(h_p_category, P_LEN, g_allocator);

  int *d_s_suppkey = loadToGPU<int>(h_s_suppkey, S_LEN, g_allocator);
  int *d_s_region = loadToGPU<int>(h_s_region, S_LEN, g_allocator);

  cout << "** VECTOR TEST **" << endl;
  for (int t = 0; t < num_trials; t++) {
    float time_query;
    time_query = runQuery<QueryVariant::Vector>(
        d_lo_orderdate, d_lo_partkey, d_lo_suppkey, d_lo_revenue, LO_LEN,
        d_p_partkey, d_p_brand1, d_p_category, P_LEN,
        d_d_datekey, d_d_year, D_LEN,
        d_s_suppkey, d_s_region, S_LEN,
        g_allocator);
    cout<< "{"
        << "\"type\":vec" 
        << ",\"query\":21" 
        << ",\"time_query\":" << time_query
        << "}" << endl;
  }

  cout << "** COMPILED TEST **" << endl;
  for (int t = 0; t < num_trials; t++) {
    float time_query;
    time_query = runQuery<QueryVariant::Compiled>(
        d_lo_orderdate, d_lo_partkey, d_lo_suppkey, d_lo_revenue, LO_LEN,
        d_p_partkey, d_p_brand1, d_p_category, P_LEN,
        d_d_datekey, d_d_year, D_LEN,
        d_s_suppkey, d_s_region, S_LEN,
        g_allocator);
    cout<< "{"
        << "\"type\":comp" 
        << ",\"query\":21" 
        << ",\"time_query\":" << time_query
        << "}" << endl;
  }
  return 0;
}

