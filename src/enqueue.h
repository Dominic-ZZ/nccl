/*************************************************************************
 * Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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

#ifndef enqueue_h_
#define enqueue_h_

#include "core.h"

int getRingIndex(const ncclComm_t comm, int device);
void lockEventQueue(EventQueue *eq);
void releaseEventQueue(EventQueue *eq);
void CUDART_CB freeEvent(cudaStream_t stream, cudaError_t status, void *eq_void);

/* Syncronize with user stream and launch the collective.
 * All work is performed asynchronously with the host thread.
 * The actual collective should be a functor with the
 * folloaing signature.
 * ncclResult_t collective(void* sendbuff, void* recvbuff,
 *                         int count, ncclDataType_t type, ncclRedOp_t op,
 *                         int root, ncclComm_t comm);
 * Unneeded arguments should be ignored. The collective may
 * assume that the appropriate cuda device has been set.
 与user stream同步并加载collective函数。所有的工作都是与host thread异步执行的。
 实际的collective应该为如下的函数格式。

 */

/*
这段代码是一个 enqueue 函数，用于将一个 collective 操作（如 all-reduce、all-gather
等）加入到队列中，以便异步执行。这个函数会在指定的流上启动 collective 操作，并将一个事件记录到
events 队列中。这个事件表示 collective 操作已经完成。如果队列中已经有完成的 collective
操作，则等待该事件完成后才启动新的 collective 操作。

这个函数的参数包括一个 ColFunc，它是一个函数指针，指向实际的 collective
实现函数。此外，还有输入和输出缓冲区、元素数量、数据类型、归约操作类型、根节点 ID、通信句柄以及 CUDA
流。这些参数将被传递给 ColFunc 函数，以执行实际的 collective 操作。函数返回一个 ncclResult_t
类型的值，表示操作是否成功。
*/
template <typename ColFunc>
ncclResult_t enqueue(ColFunc colfunc,
                     const void *sendbuff,
                     void *recvbuff,
                     int count,
                     ncclDataType_t type,
                     ncclRedOp_t op,
                     int root,
                     ncclComm_t comm,
                     cudaStream_t stream)
{
  int curDevice;
  CUDACHECK(cudaGetDevice(&curDevice));  // 获取当前设备的ID

  // No need for a mutex here because we assume that all enqueue operations happen in a fixed
  // order on all devices. Thus, thread race conditions SHOULD be impossible.
  // 在这里不需要互斥锁，因为我们假定所有的入队操作在所有设备上都以固定的顺序进行。因此，线程之间的竞争条件应该是不可能的。
  EventQueue *eq = &comm->events;  // 获取通信对象的事件队列

  // Ensure that previous collective is complete
  // 确保上一个集体操作已经完成
  cudaError_t flag = cudaEventQuery(eq->isDone[eq->back]);  // 查询上一个事件是否完成
  if (flag == cudaErrorNotReady)                            // 如果上一个事件还没有完成
    CUDACHECK(cudaStreamWaitEvent(stream, eq->isDone[eq->back], 0));  // 在CUDA流中等待事件完成

  // Launch the collective here
  // 启动收集操作

  ncclResult_t ret = colfunc(sendbuff, recvbuff, count, type, op, root, comm, stream);
  // 调用收集函数

  eq->back = (eq->back + 1) % MAXQUEUE;                      // 更新队列尾指针
  CUDACHECK(cudaEventRecord(eq->isDone[eq->back], stream));  // 在CUDA流中记录完成事件
  return ret;
}

#endif // End include guard
