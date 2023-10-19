#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>

#define MPICHECK(cmd)                                \
    do                                               \
    {                                                \
        int e = cmd;                                 \
        if (e != MPI_SUCCESS)                        \
        {                                            \
            printf("Failed: MPI error %s:%d '%d'\n", \
                   __FILE__, __LINE__, e);           \
            exit(EXIT_FAILURE);                      \
        }                                            \
    } while (0)

#define CUDACHECK(cmd)                                         \
    do                                                         \
    {                                                          \
        cudaError_t e = cmd;                                   \
        if (e != cudaSuccess)                                  \
        {                                                      \
            printf("Failed: Cuda error %s:%d '%s'\n",          \
                   __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while (0)

#define NCCLCHECK(cmd)                                         \
    do                                                         \
    {                                                          \
        ncclResult_t r = cmd;                                  \
        if (r != ncclSuccess)                                  \
        {                                                      \
            printf("Failed, NCCL error %s:%d '%s'\n",          \
                   __FILE__, __LINE__, ncclGetErrorString(r)); \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while (0)

// 生成哈希值，用于标识不同主机的唯一标识符，以便在集体通信中进行通信协调
static uint64_t getHostHash(const char *string)
{
    // Based on DJB2a, result = result * 33 ^ char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++)
    {
        result = ((result << 5) + result) ^ string[c];
    }
    return result;
}

// 获取主机名
static void getHostName(char *hostname, int maxlen)
{
    gethostname(hostname, maxlen);
    for (int i = 0; i < maxlen; i++)
    {
        if (hostname[i] == '.')
        {
            hostname[i] = '\0';
            return;
        }
    }
}

int main(int argc, char *argv[])
{
    int size = 32 * 1024 * 1024;

    int myRank, nRanks, localRank = 0;

    // initializing MPI
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank)); // 获取当前进程的编号
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks)); // 获取当前通信域中进程的总数

    // calculating localRank based on hostname which is used in selecting a GPU
    uint64_t hostHashs[nRanks];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[myRank] = getHostHash(hostname);
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
    // 计算每个进程（rank）在本地（host）上的排名（localRank）
    for (int p = 0; p < nRanks; p++)
    {
        if (p == myRank)
            break;
        if (hostHashs[p] == hostHashs[myRank])
            localRank++;
    }
    /* 假设我们有两台服务器（host0和host1），每个机器上都有4个GPU（GPU0~3和GPU4~7），此时一共有8个进程，nRanks = 8。
    对于host0上的GPU2(myRank=2)：
        p=0，hostHashs[0] == hostHashs[2]，localRank=1；
        p=1，hostHashs[1] == hostHashs[2]，localRank=2；
        p=2，p == myRank，localRank=2，退出。
        那么，GPU2在host0上的localRank=2。
    对于host1上的GPU6(myRank=6)：
        p=0，hostHashs[0] != hostHashs[6]，localRank=0；
        p=1，hostHashs[1] != hostHashs[6]，localRank=0；
        p=2，hostHashs[2] != hostHashs[6]，localRank=0；
        ......
        p=4，hostHashs[4] == hostHashs[6]，localRank=1；
        p=5，hostHashs[5] == hostHashs[6]，localRank=2；
        p=6，p == myRank，localRank=2，退出。
        那么，GPU6在host1上的localRank=2.
    */

    ncclUniqueId id;
    ncclComm_t comm;
    float *sendbuff, *recvbuff;
    cudaStream_t s;

    // get NCCL unique ID at rank 0 and broadcast it to all others
    if (myRank == 0)
        ncclGetUniqueId(&id);
    MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    // picking a GPU based on localRank, allocate device buffers
    CUDACHECK(cudaSetDevice(localRank));
    CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
    CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(&s));

    // initializing NCCL
    NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

    // communicating using NCCL
    NCCLCHECK(ncclAllReduce((const void *)sendbuff, (void *)recvbuff, size, ncclFloat, ncclSum,
                            comm, s));

    // completing NCCL operation by synchronizing on the CUDA stream
    CUDACHECK(cudaStreamSynchronize(s));

    // free device buffers
    CUDACHECK(cudaFree(sendbuff));
    CUDACHECK(cudaFree(recvbuff));

    // finalizing NCCL
    ncclCommDestroy(comm);

    // finalizing MPI
    MPICHECK(MPI_Finalize());

    printf("[MPI Rank %d] Success \n", myRank);
    return 0;
}