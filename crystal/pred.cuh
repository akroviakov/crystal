#pragma once

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void InitFlags(
    int* selection_flags
    ) {
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    selection_flags[ITEM] = 1;
  }
}

template<typename T, typename SelectOp, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredDirect(
    int tid,
    T* __restrict__  items,
    SelectOp select_op,
    int* selection_flags
    ) {
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    selection_flags[ITEM] = select_op(items[ITEM]);
  }
}

template<typename T, typename SelectOp, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredDirect(
    int tid,
    T* __restrict__  items,
    SelectOp select_op,
    int* __restrict__  selection_flags,
    int num_items
    ) {
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      selection_flags[ITEM] = select_op(items[ITEM]);
    }
  }
}

template<typename T, typename SelectOp, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPred(
    T* __restrict__  items,
    SelectOp select_op,
    int* __restrict__  selection_flags,
    int num_items
    ) {

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockPredDirect<T, SelectOp, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, select_op, selection_flags);
  } else {
    BlockPredDirect<T, SelectOp, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, select_op, selection_flags, num_items);
  }
}

template<typename T, typename SelectOp, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredAndDirect(
    int tid,
    T* __restrict__  items,
    SelectOp select_op,
    int* selection_flags
    ) {
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    selection_flags[ITEM] = selection_flags[ITEM] && select_op(items[ITEM]);
  }
}

template<typename T, typename SelectOp, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredAndDirect(
    int tid,
    T* __restrict__  items,
    SelectOp select_op,
    int* __restrict__  selection_flags,
    int num_items
    ) {
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      selection_flags[ITEM] = selection_flags[ITEM] && select_op(items[ITEM]);
    }
  }
}

template<typename T, typename SelectOp, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredAnd(
    T* __restrict__  items,
    SelectOp select_op,
    int* __restrict__  selection_flags,
    int num_items
    ) {

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockPredAndDirect<T, SelectOp, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, select_op, selection_flags);
  } else {
    BlockPredAndDirect<T, SelectOp, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, select_op, selection_flags, num_items);
  }
}

template<typename T, typename SelectOp, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredOrDirect(
    int tid,
    T* __restrict__  items,
    SelectOp select_op,
    int* selection_flags
    ) {
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    selection_flags[ITEM] = selection_flags[ITEM] || select_op(items[ITEM]);
  }
}

template<typename T, typename SelectOp, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredOrDirect(
    int tid,
    T* __restrict__  items,
    SelectOp select_op,
    int* __restrict__  selection_flags,
    int num_items
    ) {
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      selection_flags[ITEM] = selection_flags[ITEM] || select_op(items[ITEM]);
    }
  }
}

template<typename T, typename SelectOp, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredOr(
    T* __restrict__  items,
    SelectOp select_op,
    int* __restrict__  selection_flags,
    int num_items
    ) {

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockPredOrDirect<T, SelectOp, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, select_op, selection_flags);
  } else {
    BlockPredOrDirect<T, SelectOp, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, select_op, selection_flags, num_items);
  }
}

template<typename T>
struct LessThan
{
  T compare;

  __device__ __forceinline__
  LessThan(T compare) : compare(compare) {}

  __device__ __forceinline__
  bool operator()(const T &a) const {
    return (a < compare);
  }
};

template<typename T>
struct GreaterThan
{
  T compare;

  __device__ __forceinline__
  GreaterThan(T compare) : compare(compare) {}

  __device__ __forceinline__
  bool operator()(const T &a) const {
    return (a > compare);
  }
};

template<typename T>
struct LessThanEq
{
  T compare;

  __device__ __forceinline__
  LessThanEq(T compare) : compare(compare) {}

  __device__ __forceinline__
  bool operator()(const T &a) const {
    return (a <= compare);
  }
};

template<typename T>
struct GreaterThanEq
{
  T compare;

  __device__ __forceinline__
  GreaterThanEq(T compare) : compare(compare) {}

  __device__ __forceinline__
  bool operator()(const T &a) const {
    return (a >= compare);
  }
};

template<typename T>
struct Eq
{
  T compare;

  __device__ __forceinline__
  Eq(T compare) : compare(compare) {}

  __device__ __forceinline__
  bool operator()(const T &a) const {
    return (a == compare);
  }
};

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredLT(
    T* __restrict__  items,
    T compare,
    int* __restrict__  selection_flags,
    int num_items
    ) {
  LessThan<T> select_op(compare);
  BlockPred<T, LessThan<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(items, select_op, selection_flags, num_items);
}

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredAndLT(
    T* __restrict__  items,
    T compare,
    int* __restrict__  selection_flags,
    int num_items
    ) {
  LessThan<T> select_op(compare);
  BlockPredAnd<T, LessThan<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(items, select_op, selection_flags, num_items);
}

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredGT(
    T* __restrict__  items,
    T compare,
    int* __restrict__  selection_flags,
    int num_items
    ) {
  GreaterThan<T> select_op(compare);
  BlockPred<T, GreaterThan<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(items, select_op, selection_flags, num_items);
}

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredAndGT(
    T* __restrict__  items,
    T compare,
    int* __restrict__  selection_flags,
    int num_items
    ) {
  GreaterThan<T> select_op(compare);
  BlockPredAnd<T, GreaterThan<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(items, select_op, selection_flags, num_items);
}

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredLTE(
    T* __restrict__  items,
    T compare,
    int* __restrict__  selection_flags,
    int num_items
    ) {
  LessThanEq<T> select_op(compare);
  BlockPred<T, LessThanEq<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(items, select_op, selection_flags, num_items);
}

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredAndLTE(
    T* __restrict__  items,
    T compare,
    int* __restrict__  selection_flags,
    int num_items
    ) {
  LessThanEq<T> select_op(compare);
  BlockPredAnd<T, LessThanEq<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(items, select_op, selection_flags, num_items);
}

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredGTE(
    T* __restrict__  items,
    T compare,
    int* __restrict__  selection_flags,
    int num_items
    ) {
  GreaterThanEq<T> select_op(compare);
  BlockPred<T, GreaterThanEq<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(items, select_op, selection_flags, num_items);
}

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredAndGTE(
    T* __restrict__  items,
    T compare,
    int* __restrict__  selection_flags,
    int num_items
    ) {
  GreaterThanEq<T> select_op(compare);
  BlockPredAnd<T, GreaterThanEq<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(items, select_op, selection_flags, num_items);
}

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredEQ(
    T* __restrict__  items,
    T compare,
    int* __restrict__  selection_flags,
    int num_items
    ) {
  Eq<T> select_op(compare);
  BlockPred<T, Eq<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(items, select_op, selection_flags, num_items);
}

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredAndEQ(
    T* __restrict__  items,
    T compare,
    int* __restrict__  selection_flags,
    int num_items
    ) {
  Eq<T> select_op(compare);
  BlockPredAnd<T, Eq<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(items, select_op, selection_flags, num_items);
}
