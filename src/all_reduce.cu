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

#include <nvToolsExt.h>

#include "common_kernel.h"
#include "copy_kernel.h"
#include "core.h"
#include "enqueue.h"
#include "reduce_kernel.h"

/* HIERARCHY
 *
 * The data is split into CHUNKS, and each CHUNK is split into NUM_SUBCHUNKS
 * SUBCHUNKS, where each SUBCHUNK is an independent, complete reduction. Each
 * GPU has a buffer that can fit an entire CHUNK, so that all SUBCHUNKS can be
 * processed without checking that the buffer on the receiving GPU is empty. A
 * SUBCHUNK is split into NUM_GPUS SLICES and each GPU works on a different
 * SLICE at the same time. Before moving on the the next SLICE in the reduction
 * algorithm, the GPU has to check whether it has received the data from the
 * previous GPU it needs for this SLICE. To hide the latency of this
 * communication, each GPU processes all the SLICES of all the SUBCHUNKS in
 * sequence before moving on to the next SLICE. Each SLICE is split into a
 * certain number of UNROLLS (determined by the buffer size) and each thread
 * performs UNROLL_COUNT single-data-element operations inside an UNROLL. As the
 * name suggests, the UNROLL_COUNT operations within an UNROLL are unrolled.
 */

// Number of threads used to perform copies, etc. Must be multiple of 32.
// An additional thread is used to handle threadfences, so the CUDA blocks
// have dimension NUM_THREADS+1.
#define NUM_THREADS 256

// Each thread unrolls the innermost loop of the copy or reduction operations
// to this many single-data-element instructions
// 每个线程将复制或还原操作的最内层循环展开到这些单个数据元素指令
#define UNROLL_COUNT 8

#define UNROLL_SIZE (UNROLL_COUNT * NUM_THREADS)

// To hide the latency associated with the synchronization between different
// subchunks, we interleave the independent subchunks so that more data can be
// transferred while the sync is in progress. This is the number of subchunks
// that are active at the same time
// 为了隐藏与不同子块之间同步相关的延迟，我们交错独立子块，以便在同步进行时可以传输更多数据。这是同时处于活动状态的子块的数量
#define NUM_SUBCHUNKS 2

// If this is called with STEP, it means that we just finished processing the
// data for step STEP on this GPU, which is the data required on the next GPU
// for step STEP + 1, so we signal the next GPU that its data for step STEP + 1
// is available. This is called by one particular consumer warp and so we select
// the first thread in the warp to set the flag.
#define SIGNAL_NEW_DATA_AVAILABLE(chunk, subchunk, step)                  \
  do {                                                                    \
    __threadfence_system();                                               \
    args.NextNewDataAvailableFlag[0] =                                    \
        NUM_SUBCHUNKS * ((chunk) * (2 * args.NumGPUs - 2) + (step) + 1) + \
        subchunk;                                                         \
  } while (0)

// This is called by all producer threads, but only thread 0 spins on the flag,
#define WAIT_FOR_NEW_DATA(chunk, subchunk, step)                           \
  do {                                                                     \
    if (tid == 0) {                                                        \
      Wait([=] {                                                           \
        return ((volatile int *)args.ThisNewDataAvailableFlag)[0] >=       \
               2 * ((chunk) * (2 * args.NumGPUs - 2) + (step)) + subchunk; \
      });                                                                  \
    }                                                                      \
    BAR(sync, 1, NUM_THREADS);                                             \
  } while (0)

#define SIGNAL_CHUNK_DONE(chunk, subchunk)                              \
  do {                                                                  \
    args.PrevChunkDoneFlag[0] = NUM_SUBCHUNKS * (chunk) + subchunk + 1; \
  } while (0)

#define WAIT_FOR_CHUNK(chunk, subchunk)                                \
  do {                                                                 \
    if (tid == 0) {                                                    \
      Wait([=] {                                                       \
        return ((volatile int *)args.ThisChunkDoneFlag)[0] >=          \
               NUM_SUBCHUNKS * (chunk) + subchunk + 1 - NUM_SUBCHUNKS; \
      });                                                              \
    }                                                                  \
    BAR(sync, 1, NUM_THREADS);                                         \
  } while (0)

// 计算slice在数据中的大小和偏移量，这样就可以在CUDA kernel中
// 用这个偏移量来找到当前切片在输入数据中的位置，
__device__ inline void getSliceSizeAndOffset(int *size,
                                             int *offset,
                                             int slice,
                                             int numSlices,
                                             int numBigSlices,
                                             int numSmallSlices,
                                             int bigSliceN,
                                             int smallSliceN,
                                             int lastSliceN) {
  if (slice < numBigSlices) {
    *size = bigSliceN;
    *offset = slice * bigSliceN;
  } else {
    *size = (slice < numBigSlices + numSmallSlices)
                ? smallSliceN
                : ((slice == numSlices - 1) ? lastSliceN : 0);
    *offset = numBigSlices * bigSliceN + (slice - numBigSlices) * smallSliceN;
  }
}

template <typename T>
struct AllReduceKernelArgs {
  // general parameters
  int ThisId;
  int NumGPUs;
  int N;

  // some pre-computed sizes
  int SliceSize;
  int ChunkSize;
  int NumChunks;

  T **ThisPtrToNextOutput;
  T **PrevPtrToThisOutput;

  // local and remote input, output, and buffer
  const T *__restrict__ ThisInput;
  volatile T *__restrict__ ThisOutput;
  volatile T *__restrict__ ThisBuffer;
  volatile T *__restrict__ NextBuffer;

  // local and remote flags
  volatile int *__restrict__ ThisNewDataAvailableFlag;
  volatile int *__restrict__ NextNewDataAvailableFlag;
  volatile int *__restrict__ ThisChunkDoneFlag;
  volatile int *__restrict__ PrevChunkDoneFlag;
};

__shared__ volatile void *nextOutput;

// AllreduceKernel函数，用于在多个GPU之间执行AllReduce操作
template <int THREADS, int UNROLL, class FUNC, bool PUSHRECV, typename T>
// __launch_bounds__宏可以设置内核函数的线程块大小和限制内存使用等级
__launch_bounds__(THREADS + WARP_SIZE, 1) __global__
    void AllReduceKernel(const AllReduceKernelArgs<T> args) {
  // 如果数据量N为0，则直接返回
  if (args.N == 0) return;
  const int tid = threadIdx.x;

  // First wait for args.PrevPtrToThisOutput to become nullptr to ensure that
  // the previous GPU is done with a previous collective operation.
  // 等待前一个GPU完成所有collective operations
  if (tid == 0) {
    Wait([=] { return *((T *volatile *)args.PrevPtrToThisOutput) == nullptr; });

    *((T *volatile *)args.PrevPtrToThisOutput) = (T *)args.ThisOutput;

    // 等待下一个GPU的指针不为空
    Wait([=] { return *((T *volatile *)args.ThisPtrToNextOutput) != nullptr; });

    if (PUSHRECV)
      nextOutput = *((volatile void *volatile *)args.ThisPtrToNextOutput);
  }
  __syncthreads();

  // 对于每个chunk，执行AllReduce操作
  for (int chunk = 0; chunk < args.NumChunks; ++chunk) {
    // calculate slice size.  for all chunks except (possibly) the last one,
    // this will just be args.SliceSize. For the last one, it may be smaller
    // 计算切片大小。对于除了最后一个之外的所有块，它都将是args.SliceSize。对于最后一个，它可能会更小
    int bigSliceN = args.SliceSize;
    int smallSliceN = 0;
    int lastSliceN = 0;
    int numSlices = args.NumGPUs * NUM_SUBCHUNKS;
    int numBigSlices = numSlices;
    int numSmallSlices = 0;

    // last chunk
    // 如果是最后一个chunk，则计算最后一个slice的大小
    if ((chunk + 1 == args.NumChunks) && (args.N % args.ChunkSize > 0))
      CalcLastChunk<THREADS, UNROLL, T>(&bigSliceN,
                                        &smallSliceN,
                                        &lastSliceN,
                                        &numSlices,
                                        &numBigSlices,
                                        &numSmallSlices,
                                        args.N,
                                        args.NumChunks,
                                        args.ChunkSize);

    // this offset is only applied to Data pointers, not to Buffer pointers,
    // since we only have one buffer per chunk
    // 计算当前chunk的偏移量
    int chunkOffset = chunk * args.ChunkSize;

    /////////////// begin AllReduce steps ///////////////

    // step 0: push data to next GPU
    // stpe 0: 将数据推送到下一个GPU
    int step = 0;
    int slice = args.ThisId;
    int offset;
    int sliceSize;

    if (tid < THREADS) {
      for (int s = 0; s < NUM_SUBCHUNKS; ++s) {
        if (s > 0) {
          slice += args.NumGPUs;
        }
        getSliceSizeAndOffset(&sliceSize,
                              &offset,
                              slice,
                              numSlices,
                              numBigSlices,
                              numSmallSlices,
                              bigSliceN,
                              smallSliceN,
                              lastSliceN);

        if (!PUSHRECV && chunk > 0) {
          WAIT_FOR_CHUNK(chunk, s);
        }

        Copy<UNROLL, THREADS>(args.NextBuffer + offset,
                              args.ThisInput + chunkOffset + offset,
                              sliceSize);

        __syncthreads();
      }
    } else {  // is consumer thread
      for (int s = 0; s < NUM_SUBCHUNKS; ++s) {
        __syncthreads();
        SIGNAL_NEW_DATA_AVAILABLE(chunk, s, step);
      }
    }

    // steps j with 1 <= j < k - 1, where k = number of GPUs:
    // reduce and copy to next GPU
    for (step = 1; step < args.NumGPUs - 1; ++step) {
      if (tid < THREADS) {
        slice = (args.NumGPUs + slice - 1) % args.NumGPUs;
        for (int s = 0; s < NUM_SUBCHUNKS; ++s) {
          if (s > 0) {
            slice += args.NumGPUs;
          }
          getSliceSizeAndOffset(&sliceSize,
                                &offset,
                                slice,
                                numSlices,
                                numBigSlices,
                                numSmallSlices,
                                bigSliceN,
                                smallSliceN,
                                lastSliceN);

          WAIT_FOR_NEW_DATA(chunk, s, step);

          Reduce<UNROLL, THREADS, FUNC>(args.NextBuffer + offset,
                                        args.ThisBuffer + offset,
                                        args.ThisInput + chunkOffset + offset,
                                        sliceSize);

          __syncthreads();
        }
      } else {
        for (int s = 0; s < NUM_SUBCHUNKS; ++s) {
          __syncthreads();
          SIGNAL_NEW_DATA_AVAILABLE(chunk, s, step);
        }
      }
    }

    // step k - 1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    step = args.NumGPUs - 1;

    if (tid < THREADS) {
      slice = (args.NumGPUs + slice - 1) % args.NumGPUs;
      for (int s = 0; s < NUM_SUBCHUNKS; ++s) {
        if (s > 0) {
          slice += args.NumGPUs;
        }
        getSliceSizeAndOffset(&sliceSize,
                              &offset,
                              slice,
                              numSlices,
                              numBigSlices,
                              numSmallSlices,
                              bigSliceN,
                              smallSliceN,
                              lastSliceN);

        WAIT_FOR_NEW_DATA(chunk, s, step);

        if (PUSHRECV) {
          ReduceAndCopy<UNROLL, THREADS, FUNC>(
              (volatile T *)nextOutput + chunkOffset + offset,
              args.ThisOutput + chunkOffset + offset,
              args.ThisBuffer + offset,
              args.ThisInput + chunkOffset + offset,
              sliceSize);
        } else {
          ReduceAndCopy<UNROLL, THREADS, FUNC>(
              args.NextBuffer + offset,
              args.ThisOutput + chunkOffset + offset,
              args.ThisBuffer + offset,
              args.ThisInput + chunkOffset + offset,
              sliceSize);
        }

        __syncthreads();
      }
    } else {
      for (int s = 0; s < NUM_SUBCHUNKS; ++s) {
        __syncthreads();
        SIGNAL_NEW_DATA_AVAILABLE(chunk, s, step);
      }
    }

    // steps j with k <= j < 2*k-2: copy result to next GPU
    for (step = args.NumGPUs; step < 2 * args.NumGPUs - 2; ++step) {
      if (tid < THREADS) {
        slice = (args.NumGPUs + slice - 1) % args.NumGPUs;
        for (int s = 0; s < NUM_SUBCHUNKS; ++s) {
          if (s > 0) {
            slice += args.NumGPUs;
          }
          getSliceSizeAndOffset(&sliceSize,
                                &offset,
                                slice,
                                numSlices,
                                numBigSlices,
                                numSmallSlices,
                                bigSliceN,
                                smallSliceN,
                                lastSliceN);

          WAIT_FOR_NEW_DATA(chunk, s, step);

          if (PUSHRECV) {
            Copy<UNROLL, THREADS>(
                (volatile T *)nextOutput + chunkOffset + offset,
                args.ThisOutput + chunkOffset + offset,
                sliceSize);
          } else {
            DoubleCopy<UNROLL, THREADS>(args.NextBuffer + offset,
                                        args.ThisOutput + chunkOffset + offset,
                                        args.ThisBuffer + offset,
                                        sliceSize);
          }

          __syncthreads();
        }
      } else {
        for (int s = 0; s < NUM_SUBCHUNKS; ++s) {
          __syncthreads();
          SIGNAL_NEW_DATA_AVAILABLE(chunk, s, step);
        }
      }
    }

    if (!PUSHRECV) {
      // Make final copy from buffer to dest.
      if (tid < THREADS) {
        slice = (args.NumGPUs + slice - 1) % args.NumGPUs;
        for (int s = 0; s < NUM_SUBCHUNKS; ++s) {
          if (s > 0) {
            slice += args.NumGPUs;
          }
          getSliceSizeAndOffset(&sliceSize,
                                &offset,
                                slice,
                                numSlices,
                                numBigSlices,
                                numSmallSlices,
                                bigSliceN,
                                smallSliceN,
                                lastSliceN);

          WAIT_FOR_NEW_DATA(chunk, s, step);

          // Here we need to copy from buffer to this output.
          Copy<UNROLL, THREADS>(args.ThisOutput + chunkOffset + offset,
                                args.ThisBuffer + offset,
                                sliceSize);

          __syncthreads();
        }
      } else {
        for (int s = 0; s < NUM_SUBCHUNKS; ++s) {
          __syncthreads();
          if (chunk + 1 < args.NumChunks) {
            SIGNAL_CHUNK_DONE(chunk, s);
          }
        }
      }
    }
  }

  // wait for the last data to be pushed to us
  if (tid < THREADS) {
    if (PUSHRECV) {
      WAIT_FOR_NEW_DATA(args.NumChunks, NUM_SUBCHUNKS - 1, 0);
    }

    if (tid == 0) {
      args.ThisNewDataAvailableFlag[0] = 0;
      if (!PUSHRECV) {
        args.ThisChunkDoneFlag[0] = 0;
      }
      *args.ThisPtrToNextOutput = nullptr;
    }
  }
}

/*
chunk、subchunk、slice的关系？
这个函数是将所有`GPU`上的数据进行`AllReduce`操作。`AllReduce`操作可以分为多轮进行，每轮中每个`GPU`处理自己的一部分
数据，并将处理结果`reduce`到最终结果中。在这个函数中，为了提高性能，将所有数据进行分块，分为`chunk`、`subchunk`和
`slice`三种块。

1、`chunk`：将数据分块后，每块的大小为`chunk`，这个`chunk`的大小是根据当前数据总量计算出来的，目的是保证每个`GPU`上
的计算能够均衡，同时又能够避免出现一个特别大的块和一个特别小的块的情况，从而保证计算的效率。

2、`subchunk`：每个`GPU`将自己的`chunk`再分成多个`subchunk`，每个`subchunk`的大小为`subchunk`，每个`GPU`的`subchunk`
数量为`NUM_SUBCHUNKS`。这里的目的是，将`chunk`分成多个小块，使得每个`GPU`可以同时处理多个小块，从而提高计算效率。

3、`slice`：每个`GPU`的`subchunk`再按照`UNROLL_SIZE`进行对齐，得到一个`slice`，即每个`GPU`在一轮通信中处理的数据块大小。
这里的目的是，将`subchunk`进一步切割成大小相等的小块，使得每个线程处理的数据块大小相等，从而提高计算效率。

综上所述，每个`GPU`处理的数据块大小为`slice`，每个`GPU`的`subchunk`数量为`NUM_SUBCHUNKS`，故每轮通信的数据块大小为
`NUM_SUBCHUNKS * comm->nDev * slice`。
*/
template <class FUNC, typename T>
ncclResult_t ncclAllReduceWithTypeAndFunc(const void *sendbuff,
                                          void *recvbuff,
                                          const int count,
                                          ncclComm *comm,
                                          cudaStream_t stream) {
  if (count == 0) return ncclSuccess;
  // 获取当前GPU的ID
  int index = comm->ncclId;

  // There is one slice per GPU, so a slice can be at most bufferN / numGPUs,
  // where bufferN is the number of elements of type T that fit into the buffer.
  // For efficiency, we want the slice size to be a multiple of UNROLL_SIZE
  // 计算每个GPU所需要分配的buffer的大小，其中bufferN是buffer中可以装下的T类型数据的个数
  int bufferN = comm->buffSize / sizeof(T);  // buffer的个数
  // 将buffer上的数据分块，得到未对齐的slicesize
  int bufferNPerSlice = bufferN / (NUM_SUBCHUNKS * comm->nDev);
  // 对齐slicesize，希望slice的大小可以被UNROLL_SIZE整除，以提高效率
  int sliceSize = (bufferNPerSlice / UNROLL_SIZE) * UNROLL_SIZE;

  // 计算前驱和后继的ID
  int nextId = (index + 1) % comm->nDev;
  int prevId = (index + comm->nDev - 1) % comm->nDev;

  // 创建AllReduceKernelArgs结构体，保存了一些参数，包括每个GPU的ID、需要处理的数据量、slice的大小、
  // chunk的大小、输入输出指针等等
  AllReduceKernelArgs<T> args;

  args.ThisId = index;        // 当前GPU的ID
  args.NumGPUs = comm->nDev;  // 总GPU个数
  args.N = count;             // 需要处理的数据量

  args.SliceSize = sliceSize;  // slice大小
  int subchunkSize = comm->nDev * args.SliceSize;
  args.ChunkSize = NUM_SUBCHUNKS * subchunkSize;  // chunk大小

  // avoid a case where we have one or more big chunks and one tiny one
  // 避免过大或过小的情况出现
  int remainder = args.N % args.ChunkSize;
  if ((args.N > args.ChunkSize) && (remainder > 0) &&
      (args.N < 5 * args.ChunkSize) && (2 * remainder < args.ChunkSize)) {
    // 缩小slice的大小、chunk的大小，并重新计算NumChunks
    args.SliceSize /= 2;
    int subchunkSize = comm->nDev * args.SliceSize;
    args.ChunkSize = NUM_SUBCHUNKS * subchunkSize;

    // round down so we end up with a big last chunk
    args.NumChunks = args.N / args.ChunkSize;
  } else {
    // round up
    args.NumChunks = (args.N + args.ChunkSize - 1) / args.ChunkSize;
  }

  // 设置当前GPU的前驱和后继的输出指针
  args.ThisPtrToNextOutput = (T **)&(comm->ptrs[nextId].local->recvPtrs[0]);
  args.PrevPtrToThisOutput = (T **)&(comm->ptrs[prevId].remote->recvPtrs[0]);

  args.ThisInput = (const T *)sendbuff;      // 输入指针
  args.ThisOutput = (volatile T *)recvbuff;  // 输出指针
  args.ThisBuffer =
      (volatile T *)comm->ptrs[prevId].local->buff;  // 当前GPU的前驱的buffer
  args.NextBuffer =
      (volatile T *)comm->ptrs[nextId].remote->buff;  // 当前GPU的后继的buffer

  // 用于判断前驱GPU是否已经将数据写入到共享内存中,当前GPU是否可以进行下一步计算
  args.ThisNewDataAvailableFlag =
      comm->ptrs[prevId].local->flags;  // 当前GPU的前驱的flag
  args.NextNewDataAvailableFlag =
      comm->ptrs[nextId].remote->flags;  // 当前GPU的后继的flag

  args.ThisChunkDoneFlag =
      comm->ptrs[nextId].local->flags + 1;  // 当前GPU的后继的flag+1
  args.PrevChunkDoneFlag =
      comm->ptrs[prevId].remote->flags + 1;  // 当前GPU的前驱的flag+1

  // 是否可以访问从远程GPU传递的recvbuff指针,区别在于单or多进程
  if (comm->useRemoteRecv) {
    // 调用CUDA kernel函数，执行AllReduce操作
    AllReduceKernel<NUM_THREADS, UNROLL_COUNT, FUNC, true, T>
        <<<1, NUM_THREADS + 1, 0, stream>>>(args);
  } else {
    AllReduceKernel<NUM_THREADS, UNROLL_COUNT, FUNC, false, T>
        <<<1, NUM_THREADS + 1, 0, stream>>>(args);
  }
  return ncclSuccess;
}

template <typename T>
ncclResult_t ncclAllReduceWithType(const void *sendbuff,
                                   void *recvbuff,
                                   int count,
                                   ncclRedOp_t op,
                                   ncclComm *comm,
                                   cudaStream_t stream) {
  switch (op) {
    case ncclSum:
      return ncclAllReduceWithTypeAndFunc<FuncSum<T>, T>(
          sendbuff, recvbuff, count, comm, stream);
    case ncclProd:
      return ncclAllReduceWithTypeAndFunc<FuncProd<T>, T>(
          sendbuff, recvbuff, count, comm, stream);
    case ncclMax:
      return ncclAllReduceWithTypeAndFunc<FuncMax<T>, T>(
          sendbuff, recvbuff, count, comm, stream);
    case ncclMin:
      return ncclAllReduceWithTypeAndFunc<FuncMin<T>, T>(
          sendbuff, recvbuff, count, comm, stream);
  }
  return ncclInvalidOperation;
}

class AllReduceFunctor {
 public:
  ncclResult_t operator()(const void *sendbuff,
                          void *recvbuff,
                          int count,
                          ncclDataType_t datatype,
                          ncclRedOp_t op,
                          int /*root*/,
                          ncclComm *comm,
                          cudaStream_t stream) {
    switch (datatype) {
      case ncclChar:
        return ncclAllReduceWithType<char>(
            sendbuff, recvbuff, count, op, comm, stream);
      case ncclInt:
        return ncclAllReduceWithType<int>(
            sendbuff, recvbuff, count, op, comm, stream);
#if CUDART_VERSION >= 7050
      case ncclHalf:
        return ncclAllReduceWithType<half>(
            sendbuff, recvbuff, count, op, comm, stream);
#endif
      case ncclFloat:
        return ncclAllReduceWithType<float>(
            sendbuff, recvbuff, count, op, comm, stream);
      case ncclDouble:
        return ncclAllReduceWithType<double>(
            sendbuff, recvbuff, count, op, comm, stream);
      case ncclInt64:
        return ncclAllReduceWithType<long long>(
            sendbuff, recvbuff, count, op, comm, stream);
      case ncclUint64:
        return ncclAllReduceWithType<unsigned long long int>(
            sendbuff, recvbuff, count, op, comm, stream);
    }

    return ncclInvalidType;
  }
};

extern "C" DSOGLOBAL ncclResult_t ncclAllReduce(const void *sendbuff,
                                                void *recvbuff,
                                                int count,
                                                ncclDataType_t datatype,
                                                ncclRedOp_t op,
                                                ncclComm_t comm,
                                                cudaStream_t stream) {
  return enqueue(AllReduceFunctor(),
                 sendbuff,
                 recvbuff,
                 count,
                 datatype,
                 op,
                 0,
                 comm,
                 stream);
}
