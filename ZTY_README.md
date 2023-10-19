# NCCL阅读笔记 （2023.10.19）

> 基于NCCL-v2.10.3-1

## Example: One Device per Process or Thread

案例：每个进程或线程调用一张卡

```
#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>


#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


// 生成哈希值，用于标识不同主机的唯一标识符，以便在集体通信中进行通信协调
static uint64_t getHostHash(const char* string) {
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}


// 获取主机名
static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }
}


int main(int argc, char* argv[])
{
  int size = 32*1024*1024;


  int myRank, nRanks, localRank = 0;


  //initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank)); // 获取当前进程的编号
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks)); // 获取当前通信域中进程的总数


  //calculating localRank based on hostname which is used in selecting a GPU
  uint64_t hostHashs[nRanks];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[myRank] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  // 计算每个进程（rank）在本地（host）上的排名（localRank）
  for (int p=0; p<nRanks; p++) {
     if (p == myRank) break;
     if (hostHashs[p] == hostHashs[myRank]) localRank++;
  }


  ncclUniqueId id;
  ncclComm_t comm;
  float *sendbuff, *recvbuff;
  cudaStream_t s;


  //get NCCL unique ID at rank 0 and broadcast it to all others
  if (myRank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));


  //picking a GPU based on localRank, allocate device buffers
  CUDACHECK(cudaSetDevice(localRank));
  CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
  CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));
  CUDACHECK(cudaStreamCreate(&s));


  //initializing NCCL
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));


  //communicating using NCCL
  NCCLCHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, size, ncclFloat, ncclSum,
        comm, s));


  //completing NCCL operation by synchronizing on the CUDA stream
  CUDACHECK(cudaStreamSynchronize(s));


  //free device buffers
  CUDACHECK(cudaFree(sendbuff));
  CUDACHECK(cudaFree(recvbuff));


  //finalizing NCCL
  ncclCommDestroy(comm);


  //finalizing MPI
  MPICHECK(MPI_Finalize());


  printf("[MPI Rank %d] Success \n", myRank);
  return 0;
}
```



## Step1：ncclGetUniqueId

```cpp
// get NCCL unique ID at rank 0 and broadcast it to all others
if (myRank == 0)
    ncclGetUniqueId(&id);
MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
```

根据NV的例子，第一步rank0首先执行`ncclGetUniqueId`获取id，然后通过mpi广播给其他rank。接下来看一下UniqueId是怎么产生的，以及它有什么作用。

ncclGetUniqueId函数位于src\init.cc中

```
NCCL_API(ncclResult_t, ncclGetUniqueId, ncclUniqueId* out);
ncclResult_t ncclGetUniqueId(ncclUniqueId* out) {
  NCCLCHECK(ncclInit());
  NCCLCHECK(PtrCheck(out, "GetUniqueId", "out"));
  return bootstrapGetUniqueId(out);
}
```

主要由这三个函数构成：`ncclInit()`，`PtrCheck()`，`bootstrapGetUniqueId()`

##### 1、`ncclInit()`

nccl初始化函数















# 一些疑问

##### 1、为什么用malloc申请的在CPU上的指针数组的指针（float**），能够再用来在GPU上申请（cudaMalloc）空间？

（下述代码取自于单进程-单线程-多卡demo）

```C++
  float** sendbuff = (float**)malloc(nDev * sizeof(float*));
  float** recvbuff = (float**)malloc(nDev * sizeof(float*));
  cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);


  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
    CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(s+i));
  }
```

答：

1. sendbuff和recvbuff只是用来存储指针数组的指针，在sendbuff这个长度为4的指针数组中，会存储4个指针的地址。

2. cudaMalloc这个函数，它的首个形参所输入的地址，的确是需要主存（或者说是CPU）上的地址，因为cudaMalloc 会将在显存上获得的数组首地址赋值给该主存的地址；例如：

   ```cpp
   float *device_data=NULL;
   size_t size = 1024*sizeof(float);
   cudaMalloc((void**)&device_data, size);
   ```

   上面这个例子中我在显存中申请了一个包含1024个单精度浮点数的一维数组。而device_data这个指针是存储在主存上的。之所以取device_data的地址，是为了将cudaMalloc在显存上获得的数组首地址赋值给device_data。

3. 问题的重点在于cudaMalloc这个函数的用法

4. Reference：https://blog.csdn.net/bendanban/article/details/8151335












# Reference

1. 官方文档：https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html
1. OneFlow：https://zhuanlan.zhihu.com/p/614746112