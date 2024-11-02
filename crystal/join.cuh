#pragma once

#define HASH(X,Y,Z) ((X-Z) % Y)

template<typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeDirectAndPHT_1(
    int tid,
    K* __restrict__ items,
    int* __restrict__ selection_flags,
    K* ht,
    int ht_len,
    K keys_min
) {
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        if (selection_flags[ITEM]) {
            int hash = HASH(items[ITEM], ht_len, keys_min);

            K slot = ht[hash];
            if (slot != 0) {
                selection_flags[ITEM] = 1;
            } else {
                selection_flags[ITEM] = 0;
            }
        }
    }
}

template<typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeDirectAndPHT_1(
    int tid,
    K* __restrict__ items,
    int* __restrict__ selection_flags,
    K* ht,
    int ht_len,
    K keys_min,
    int num_items
) {
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        if (tid + (ITEM * BLOCK_THREADS) < num_items) {
            if (selection_flags[ITEM]) {
                int hash = HASH(items[ITEM], ht_len, keys_min);

                K slot = ht[hash];
                if (slot != 0) {
                    selection_flags[ITEM] = 1;
                } else {
                    selection_flags[ITEM] = 0;
                }
            }
        }
    }
}

template<typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeAndPHT_1(
    K* __restrict__ items,
    int* __restrict__ selection_flags,
    K* ht,
    int ht_len,
    K keys_min,
    int num_items
) {
    if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
        BlockProbeDirectAndPHT_1<K, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, selection_flags, ht, ht_len, keys_min);
    } else {
        BlockProbeDirectAndPHT_1<K, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, selection_flags, ht, ht_len, keys_min, num_items);
    }
}

template<typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeAndPHT_1(
    K* __restrict__ items,
    int* __restrict__ selection_flags,
    K* ht,
    int ht_len,
    int num_items
) {
    BlockProbeAndPHT_1<K, BLOCK_THREADS, ITEMS_PER_THREAD>(items, selection_flags, ht, ht_len, 0, num_items);
}

template<typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeDirectAndPHT_2(
    int tid,
    K* __restrict__ keys,
    V* __restrict__ res,
    int* __restrict__ selection_flags,
    K* ht,
    int ht_len,
    K keys_min
) {
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        if (selection_flags[ITEM]) {
            int hash = HASH(keys[ITEM], ht_len, keys_min);

            uint64_t slot = *reinterpret_cast<uint64_t*>(&ht[hash << 1]);
            if (slot != 0) {
                res[ITEM] = (slot >> 32);
            } else {
                selection_flags[ITEM] = 0;
            }
        }
    }
}

template<typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeDirectAndPHT_2(
    int tid,
    K* __restrict__ items,
    V* __restrict__ res,
    int* __restrict__ selection_flags,
    K* ht,
    int ht_len,
    K keys_min,
    int num_items
) {
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        if (tid + (ITEM * BLOCK_THREADS) < num_items) {
            if (selection_flags[ITEM]) {
                int hash = HASH(items[ITEM], ht_len, keys_min);

                uint64_t slot = *reinterpret_cast<uint64_t*>(&ht[hash << 1]);
                if (slot != 0) {
                    res[ITEM] = (slot >> 32);
                } else {
                    selection_flags[ITEM] = 0;
                }
            }
        }
    }
}

template<typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeAndPHT_2(
    K* __restrict__ keys,
    V* __restrict__ res,
    int* __restrict__ selection_flags,
    K* ht,
    int ht_len,
    K keys_min,
    int num_items
) {
    if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
        BlockProbeDirectAndPHT_2<K, V, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, keys, res, selection_flags, ht, ht_len, keys_min);
    } else {
        BlockProbeDirectAndPHT_2<K, V, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, keys, res, selection_flags, ht, ht_len, keys_min, num_items);
    }
}

template<typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeAndPHT_2(
    K* __restrict__ keys,
    V* __restrict__ res,
    int* __restrict__ selection_flags,
    K* ht,
    int ht_len,
    int num_items
) {
    BlockProbeAndPHT_2<K, V, BLOCK_THREADS, ITEMS_PER_THREAD>(keys, res, selection_flags, ht, ht_len, 0, num_items);
}

template<typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildDirectSelectivePHT_1(
    int tid,
    K* __restrict__ keys,
    int* __restrict__ selection_flags,
    K* ht,
    int ht_len,
    K keys_min
) {
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        if (selection_flags[ITEM]) {
            int hash = HASH(keys[ITEM], ht_len, keys_min);

            K old = atomicCAS(&ht[hash], 0, keys[ITEM]);
        }
    }
}

template<typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildDirectSelectivePHT_1(
    int tid,
    K* __restrict__ items,
    int* __restrict__ selection_flags,
    K* ht,
    int ht_len,
    K keys_min,
    int num_items
) {
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        if (tid + (ITEM * BLOCK_THREADS) < num_items) {
            if (selection_flags[ITEM]) {
                int hash = HASH(items[ITEM], ht_len, keys_min);

                K old = atomicCAS(&ht[hash], 0, items[ITEM]);
            }
        }
    }
}

template<typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildSelectivePHT_1(
    K* __restrict__ keys,
    int* __restrict__ selection_flags,
    K* ht,
    int ht_len,
    K keys_min,
    int num_items
) {
    if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
        BlockBuildDirectSelectivePHT_1<K, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, keys, selection_flags, ht, ht_len, keys_min);
    } else {
        BlockBuildDirectSelectivePHT_1<K, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, keys, selection_flags, ht, ht_len, keys_min, num_items);
    }
}

template<typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildSelectivePHT_1(
    K* __restrict__ keys,
    int* __restrict__ selection_flags,
    K* ht,
    int ht_len,
    int num_items
) {
    BlockBuildSelectivePHT_1<K, BLOCK_THREADS, ITEMS_PER_THREAD>(keys, selection_flags, ht, ht_len, 0, num_items);
}

template<typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildDirectSelectivePHT_2(
    int tid,
    K* __restrict__ keys,
    V* __restrict__ res,
    int* __restrict__ selection_flags,
    K* ht,
    int ht_len,
    K keys_min
) {
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (selection_flags[ITEM]) {
      int hash = HASH(keys[ITEM], ht_len, keys_min);

      K old = atomicCAS(&ht[hash << 1], 0, keys[ITEM]);
      ht[(hash << 1) + 1] = res[ITEM];
    }
  }
}

template<typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildDirectSelectivePHT_2(
    int tid,
    K* __restrict__ keys,
    V* __restrict__ res,
    int* __restrict__ selection_flags,
    K* ht,
    int ht_len,
    K keys_min,
    int num_items
    ) {
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      if (selection_flags[ITEM]) {
        int hash = HASH(keys[ITEM], ht_len, keys_min);

        K old = atomicCAS(&ht[hash << 1], 0, keys[ITEM]);
        ht[(hash << 1) + 1] = res[ITEM];
      }
    }
  }
}

template<typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildSelectivePHT_2(
    K* __restrict__ keys,
    V* __restrict__ res,
    int* __restrict__ selection_flags,
    K* ht,
    int ht_len,
    K keys_min,
    int num_items
) {
    if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
        BlockBuildDirectSelectivePHT_2<K, V, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, keys, res, selection_flags, ht, ht_len, keys_min);
    } else {
        BlockBuildDirectSelectivePHT_2<K, V, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, keys, res, selection_flags, ht, ht_len, keys_min, num_items);
    }
}

template<typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildSelectivePHT_2(
    K* __restrict__ keys,
    V* __restrict__ res,
    int* __restrict__ selection_flags,
    K* ht,
    int ht_len,
    int num_items
) {
    BlockBuildSelectivePHT_2<K, V, BLOCK_THREADS, ITEMS_PER_THREAD>(keys, res, selection_flags, ht, ht_len, 0, num_items);
}
