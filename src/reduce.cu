/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ************************************************************************/

#include <algorithm>

#include <nvToolsExt.h>

#include "core.h"
#include "common_kernel.h"
#include "copy_kernel.h"
#include "enqueue.h"
#include "reduce_kernel.h"

/* HIERARCHY
 *
 * The data is split into CHUNKS, and each CHUNK is split into NUM_SUBCHUNKS
 * SUBCHUNKS, where each SUBCHUNK is processed independently. A SUBCHUNK is
 * split into numUnroll UNROLLS and each thread performs UNROLL_COUNT
 * single-data-element operations inside an UNROLL. As the name suggests, the
 * UNROLL_COUNT operations within an UNROLL are unrolled.
 数据被分割成CHUNK，每个CHUNK又被分割成NUM_SUBCHUNKS、SUBCHUNKS，其中每个SUBCHUNK被
 独立处理。SUBCHUNK被拆分为numUnroll UNROLLS，每个线程在UNROLL中执行UNROLL_COUNT单数据
 元素操作。顾名思义，UNROLL中的UNROLL_COUNT操作是展开的。
*/

// Number of threads used to perform copies, etc. Must be multiple of 32.
// An additional thread is used to handle threadfences, so the CUDA blocks
// have dimension NUM_THREADS+1.
// 用于执行copy的线程数，必须是32的倍数
// 另外一个线程用于处理threadfences，因此cuda blocks的维度需是NUM_THREADS+1
#define NUM_THREADS     256

// Each thread unrolls the innermost loop of the copy or reduction operations
// to this many single-data-element instructions 
// "每个线程对这许多单数据元素指令展开复制或缩减操作的最内层循环" 啥意思？？
#define UNROLL_COUNT    8

#define UNROLL_SIZE     (UNROLL_COUNT * NUM_THREADS)

// To hide the latency associated with the synchronization between different
// subchunks, we interleave the independent subchunks so that more data can be
// transferred while the sync is in progress. This is the number of subchunks
// that are active at the same time
// 为了隐藏与不同子块之间的同步相关的延迟，我们交错使用独立的子块，
// 以便在同步进行时可以传输更多的数据。这是同时活动的子块的数量
#define NUM_SUBCHUNKS   4

// if this is called with CHUNK, it means that we just finished pushing the data
// of chunk CHUNK to the next GPU, so it can proceed with CHUNK
// We add 1 to chunk so that the initial flag of 0 doesn't allow the non-root
// GPUs to proceed before the flag is incremented from the upstream GPU. This
// is called by one particular consumer warp and so we select the first thread
// in the warp to set the flag.
/*
如果这是用CHUNK调用的，这意味着我们刚刚完成了将CHUNK CHUNK的数据推送到下一个GPU，
所以它可以继续处理CHUNK。我们将1添加到chunk中，这样初始标记0就不允许非根GPU在上游GPU
的标记递增之前继续运行。这是由一个特定的消费线程调用的，因此我们选择该线程中的第一个线程来设置标志。
*/
#define SIGNAL_NEW_DATA_AVAILABLE(chunk, subchunk)                              \
    do {                                                                        \
      __threadfence_system();                                                   \
      args.NextNewDataAvailableFlag[0] = NUM_SUBCHUNKS*(chunk) + subchunk + 1;  \
    } while (0)

// This is called by all producer threads, but only thread 0 spins on the flag,
#define WAIT_FOR_NEW_DATA(chunk, subchunk)                                      \
    do {                                                                        \
      if (tid == 0) {                                                           \
        Wait([=] {  /* Wait函数：在func为0时，一直等待*/                                                            \
          return ((volatile int *)args.ThisNewDataAvailableFlag)[0] >=          \
              NUM_SUBCHUNKS*(chunk) + subchunk + 1;                             \
        });                                                                     \
      }                                                                         \
      BAR(sync, 1, NUM_THREADS);                                                \
    } while (0)

// If this is called with CHUNK, it means that this GPU has just finished
// processing the chunk CHUNK and so the previous GPU can start with CHUNK + 1
#define SIGNAL_CHUNK_DONE(chunk, subchunk)                                      \
    do {                                                                        \
      args.PrevChunkDoneFlag[0] = NUM_SUBCHUNKS*(chunk) + subchunk + 1;         \
    } while (0)

// This is called by all producer threads, but only thread 0 spins on the flag,
// all threads synchronize after thread 0 is done spinning.
// 这个函数会被所有的producer threads调用，但是只有线程0会循环保持这个flat。
// 在所有的线程同步之后，线程0结束循环。
#define WAIT_FOR_CHUNK(chunk, subchunk)                                         \
    do {                                                                        \
      if (tid == 0) {                                                           \
        Wait([=] {                                                              \
          return ((volatile int *)args.ThisChunkDoneFlag)[0] >=                 \
              NUM_SUBCHUNKS*(chunk) + subchunk + 1 - NUM_SUBCHUNKS;             \
        });                                                                     \
      }                                                                         \
      BAR(sync, 1, NUM_THREADS);/*???*/                                                \
    } while (0)

// This is called by all producer threads, but only thread 0 spins on the flag,
// all threads synchronize after thread 0 is done spinning.
#define WAIT_FOR_NEW_DATA_AND_CHUNK(chunk, subchunk)                            \
    do {                                                                        \
      if (tid == 0) {                                                           \
        Wait([=] {                                                              \
          bool newDataAvailable =                                               \
              ((volatile int *)args.ThisNewDataAvailableFlag)[0] >=             \
                  NUM_SUBCHUNKS*(chunk) + subchunk + 1;                         \
          bool chunkDone =                                                      \
              ((volatile int *)args.ThisChunkDoneFlag)[0] >=                    \
                  NUM_SUBCHUNKS*(chunk)+subchunk + 1 - NUM_SUBCHUNKS;           \
          return newDataAvailable && chunkDone;                                 \
        });                                                                     \
      }                                                                         \
      BAR(sync, 1, NUM_THREADS);                                                \
    } while (0)

__device__ inline void getSliceSizeAndOffset(int *size, int *offset, int slice,
    int numSlices, int numBigSlices, int numSmallSlices, int bigSliceN,
    int smallSliceN, int lastSliceN) {
  if (slice < numBigSlices) {
    *size = bigSliceN;
    *offset = slice * bigSliceN;
  } else {
    *size = (slice < numBigSlices + numSmallSlices) ? smallSliceN
        : ((slice == numSlices - 1) ? lastSliceN : 0);
    *offset = numBigSlices * bigSliceN + (slice - numBigSlices) * smallSliceN;
  }

//  if (threadIdx.x == 0)
//    printf("[size=%d] [offset=%d] slice=%d numSlices=%d "
//        "numBigSlices=%d numSmallSlices=%d bigSliceN=%d smallSliceN=%d "
//        "lastSliceN=%d\n", *size, *offset, slice, numSlices, numBigSlices,
//        numSmallSlices, bigSliceN, smallSliceN, lastSliceN);
}

template<typename T>
struct ReduceKernelArgs {
  // general parameters
  int ThisId;
  int N;

  // some pre-computed sizes
  int SliceSize;
  int ChunkSize;
  int NumChunks;
  int BufferSliceStride;

  T ** ThisPtrToNextData;
  T ** PrevPtrToThisData;

  // local and remote data
  T * __restrict__ Output;
  const T * __restrict__ ThisData;
  volatile T * __restrict__ ThisBuffer;
  volatile T * __restrict__ NextBuffer;

  // local and remote flags
  volatile int * __restrict__ ThisNewDataAvailableFlag;
  volatile int * __restrict__ NextNewDataAvailableFlag;
  volatile int * __restrict__ ThisChunkDoneFlag;
  volatile int * __restrict__ PrevChunkDoneFlag;
};

__shared__ volatile void * nextData;
enum ReduceRole {BEGIN=0, MIDDLE=1, END=2};

template<int THREADS, int UNROLL, class FUNC, int ROLE, typename T>
__global__ void ReduceKernel(const ReduceKernelArgs<T> args) {// 注意这里是global函数
  if (args.N == 0) return;
  int tid = threadIdx.x;

  // First wait for args.PrevPtrToThisOutput to become nullptr to ensure that
  // the previous GPU is done with a previous collective operation.
  // 首先等待args.PrevPtrToThisOutput变为nullptr，以确保之前的GPU完成了之前的集合操作。
  if (tid == 0) {//如果是每个block中的第一个线程，所以说nccl是在
    Wait([=] {//Wait函数定义在common_kernel.h中。等待，直到func的如果为true
      return *((T * volatile *)args.PrevPtrToThisData) == nullptr; // Wait for previous processor to be done
    });

    *((T * volatile *)args.PrevPtrToThisData) = (T*)args.ThisData; // Tell Previous I'm starting
    Wait([=] {
      return *((T * volatile *)args.ThisPtrToNextData) != nullptr;  // Wait till I've been told next started
    });
  }
  __syncthreads();

  for (int chunk = 0; chunk < args.NumChunks; ++chunk) {//chunk就是数据被分割的单位
    // calculate slice size.  for all chunks except (possibly) the last one,
    // this will just be args.SliceSize. For the last one, it may be smaller
    // 计算出每一片sub-chunk的大小（除了最后一片），因为最后一个可能会更小。
    int bigSliceN   = args.SliceSize;
    int smallSliceN = 0;
    int lastSliceN  = 0;
    int numSlices   = NUM_SUBCHUNKS;  //slice等同于sub-chunk？？
    int numBigSlices   = numSlices;
    int numSmallSlices = 0;

    // last chunk
    if ((chunk + 1 == args.NumChunks) && (args.N % args.ChunkSize > 0))
      CalcLastChunk<THREADS, UNROLL, T>(&bigSliceN, &smallSliceN, &lastSliceN,
          &numSlices, &numBigSlices, &numSmallSlices, args.N, args.NumChunks,
          args.ChunkSize);//这个函数是在计算最后一片chunk的sub-chunk的大小？

    // this offset is only applied to Data pointers, not to Buffer pointers,
    // since we only have one buffer per chunk 每片chunk只有一个buffer
    //offset：偏移量
    int chunkOffset = chunk * args.ChunkSize;//chunk的offset？每次循环指针应该移动的位置？

    int offset;
    int sliceSize;

    if (tid < THREADS) {//
      for(int s=0; s<NUM_SUBCHUNKS; ++s) { //每个chunk又会被分割成很多个subChunk
        getSliceSizeAndOffset(&sliceSize, &offset, s, numSlices,
            numBigSlices, numSmallSlices, bigSliceN, smallSliceN, lastSliceN);
            //拿到每片sub-chunk的大小和偏移量？

        if (ROLE == BEGIN) {
          WAIT_FOR_CHUNK(chunk, s);//等待，在等待啥？
          //关于UNROLL，sub-chunk又会被拆分成UNROLLs
          Copy<UNROLL, THREADS>(//所以这一步是在拷贝什么?
              args.NextBuffer + (s * args.BufferSliceStride),
              args.ThisData + chunkOffset + offset,
              sliceSize);
        } else if (ROLE == MIDDLE) {
          WAIT_FOR_NEW_DATA_AND_CHUNK(chunk, s);

          Reduce<UNROLL, THREADS, FUNC>(
              args.NextBuffer + (s * args.BufferSliceStride),
              args.ThisData + chunkOffset + offset,
              args.ThisBuffer + (s * args.BufferSliceStride),
              sliceSize);
        } else { // ROLE == END
          WAIT_FOR_NEW_DATA(chunk, s);

          Reduce<UNROLL, THREADS, FUNC>(//这一步是在reduce什么？函数是定义在common_kernel.h中的reduce or copy函数
              args.Output + chunkOffset + offset,
              args.ThisData + chunkOffset + offset,
              args.ThisBuffer + (s * args.BufferSliceStride),
              sliceSize);
        }
        __syncthreads();
      }
    } else { // Consumer thread
      for(int s=0; s<NUM_SUBCHUNKS; ++s) {
        __syncthreads();
        if (ROLE != END)
          SIGNAL_NEW_DATA_AVAILABLE(chunk, s);

        // signal chunk done if we don't push into the receive buffer and this
        // is no the last chunk and this is not root
        if ((ROLE != BEGIN) && (chunk + 1 < args.NumChunks)) {
          SIGNAL_CHUNK_DONE(chunk, s);
        }
      }
    }
  }

  // reset flags
  if (tid == 0) {
    args.ThisNewDataAvailableFlag[0] = 0;
    args.ThisChunkDoneFlag[0] = 0;
    *args.ThisPtrToNextData = nullptr;
  }
}

template<class FUNC, typename T>
ncclResult_t ncclReduceWithTypeAndFunc(const void* sendbuff, void* recvbuff,
    const int count, const int root, ncclComm* comm, cudaStream_t stream) {
  if (count == 0)
    return ncclSuccess;

  int index = comm->ncclId;

  const int numUnroll = 4;
  int rootId = comm->ringFromUser[root];

  int nextId = (index + 1) % comm->nDev;
  int prevId = (index + comm->nDev - 1) % comm->nDev;

  // There is one slice per GPU, so a slice can be at most bufferN / numGPUs,
  // where bufferN is the number of elements of type T that fit into the buffer.
  // For efficiency, we want the slice size to be a multiple of UNROLL_SIZE
  /*
  每个GPU有一个片，所以一个片最多只能是bufferN / numgpu，其中bufferN是适合这个缓冲区
  的T类型元素的数量。为了提高效率，我们希望切片大小是UNROLL_SIZE的倍数
  */
  int bufferN = comm->buffSize / sizeof(T);//一个buffer中所能存储的T类型的个数
  // we only need buffer for k slices and k paddings
  int bufferNPerSlice = bufferN / NUM_SUBCHUNKS;//？？每个subChunk中所能存储的T类型的个数？？
  int maxSliceSize = (bufferNPerSlice / UNROLL_SIZE) * UNROLL_SIZE;

  ReduceKernelArgs<T> args;

  args.ThisId = index;
  args.N = count;

  args.SliceSize = numUnroll * UNROLL_SIZE * sizeof(PackType) / sizeof(T);

  if(!comm->useRemoteRecv) {
    // Proxy for QPI. Reduce never pushes directly to recv.
    // But larger transfers help QPI more than tag updates hurt P2P.
    args.SliceSize *= 8;
  }

  // make sure slice fits into the temporary buffer
  args.SliceSize = std::min(maxSliceSize, args.SliceSize);
  args.BufferSliceStride = args.SliceSize;
  args.ChunkSize = NUM_SUBCHUNKS * args.SliceSize;

  // avoid a case where we have one or more big chunks and one tiny one
  int remainder = args.N % args.ChunkSize;
  if ((args.N > args.ChunkSize) && (remainder > 0) &&
      (args.N < 5 * args.ChunkSize) && (2 * remainder < args.ChunkSize)) {
    args.SliceSize /= 2;
    args.ChunkSize = NUM_SUBCHUNKS * args.SliceSize;

    // round down so we end up with a big last chunk
    args.NumChunks = args.N / args.ChunkSize;
  } else {
    // round up
    args.NumChunks = (args.N + args.ChunkSize - 1) / args.ChunkSize;
  }

  args.ThisPtrToNextData = (T**)&(comm->ptrs[nextId].local->recvPtrs[0]);
  args.PrevPtrToThisData = (T**)&(comm->ptrs[prevId].remote->recvPtrs[0]);

  args.Output = (T*)recvbuff;
  args.ThisData = (const T*) sendbuff;
  args.ThisBuffer = (volatile T*)comm->ptrs[prevId].local->buff;
  args.NextBuffer = (volatile T*)comm->ptrs[nextId].remote->buff;

  args.ThisNewDataAvailableFlag = comm->ptrs[prevId].local->flags;
  args.NextNewDataAvailableFlag = comm->ptrs[nextId].remote->flags;

  args.ThisChunkDoneFlag = comm->ptrs[nextId].local->flags + 1;
  args.PrevChunkDoneFlag = comm->ptrs[prevId].remote->flags + 1;

  if (index == (rootId + 1) % comm->nDev) {
    ReduceKernel<NUM_THREADS, UNROLL_COUNT, FUNC, BEGIN, T>
        <<<1, NUM_THREADS + 1, 0, stream>>>(args);
  } else if (index == rootId) {
    ReduceKernel<NUM_THREADS, UNROLL_COUNT, FUNC, END, T>
        <<<1, NUM_THREADS + 1, 0, stream>>>(args);
  } else {
    ReduceKernel<NUM_THREADS, UNROLL_COUNT, FUNC, MIDDLE, T>
        <<<1, NUM_THREADS + 1, 0, stream>>>(args);
  }
  return ncclSuccess;
}

template <typename T>
ncclResult_t ncclReduceWithType(const void* sendbuff,
      void* recvbuff, int count, ncclRedOp_t op, int root,
      ncclComm* comm, cudaStream_t stream) {

  switch (op) {
    case ncclSum:
      return ncclReduceWithTypeAndFunc<FuncSum<T>, T>(
          sendbuff, recvbuff, count, root, comm, stream);
    case ncclProd:
      return ncclReduceWithTypeAndFunc<FuncProd<T>, T>(
          sendbuff, recvbuff, count, root, comm, stream);
    case ncclMax:
      return ncclReduceWithTypeAndFunc<FuncMax<T>, T>(
          sendbuff, recvbuff, count, root, comm, stream);
    case ncclMin:
      return ncclReduceWithTypeAndFunc<FuncMin<T>, T>(
          sendbuff, recvbuff, count, root, comm, stream);
  }
  return ncclInvalidOperation;
}


class ReduceFunctor {
public:
  ncclResult_t operator()(const void* sendbuff,
      void* recvbuff, int count, ncclDataType_t datatype, ncclRedOp_t op,
      int root, ncclComm* comm, cudaStream_t stream) {

    switch (datatype) {
    case ncclChar:
      return ncclReduceWithType<char>(sendbuff, recvbuff, count, op, root, comm, stream);
    case ncclInt:
      return ncclReduceWithType<int>(sendbuff, recvbuff, count, op, root, comm, stream);
#ifdef CUDA_HAS_HALF
    case ncclHalf:
      return ncclReduceWithType<half>(sendbuff, recvbuff, count, op, root, comm, stream);
#endif
    case ncclFloat:
      return ncclReduceWithType<float>(sendbuff, recvbuff, count, op, root, comm, stream);
    case ncclDouble:
      return ncclReduceWithType<double>(sendbuff, recvbuff, count, op, root, comm, stream);
    case ncclInt64:
      return ncclReduceWithType<long long>(sendbuff, recvbuff, count, op, root, comm, stream);
    case ncclUint64:
      return ncclReduceWithType<unsigned long long>(sendbuff, recvbuff, count, op, root, comm, stream);
    }
    return ncclInvalidType;
  }
};

extern "C" DSOGLOBAL
ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, int count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm,
    cudaStream_t stream) {
  return enqueue(ReduceFunctor(), sendbuff, recvbuff, count, datatype, op,
      root, comm, stream);
}

