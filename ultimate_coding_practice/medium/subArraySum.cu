#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;

#define n 999999
#define threadsPerBlock 256

#define CHECK_CUDA(call) do {                    \
    cudaError_t e = (call);                      \
    if(e != cudaSuccess) {                        \
        cout<<"CUDA Error: "<<cudaGetErrorString(e) \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl; \
        exit(1);                                     \
    }                                               \
} while(0)


__global__ void perblockSum(int* a, int* b, int N, int S, int E) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ int cache[];
    // load in shared memory
    if(tid<(E-S + 1)) {
        cache[threadIdx.x] = a[S + tid];
    } else {
        cache[threadIdx.x] = 0;
    }
    __syncthreads();

    // parallel reduction for per block sum
    int i = blockDim.x /2;
    while(i>0) {
        if(threadIdx.x < i) {
            cache[threadIdx.x] += cache[threadIdx.x + i];
        }
        __syncthreads();
        i = i/2;
    }

    if(threadIdx.x == 0) {
        b[blockIdx.x] = cache[0];
    }
}

// Accumulate per block sum to final output
__global__ void accumulateTotalSum(int* a, int* c, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ int cache[];

    if(tid<N) {
        cache[threadIdx.x] = a[tid];
    } else {
        cache[threadIdx.x] = 0;
    }
    __syncthreads();

    int i = blockDim.x /2;
    while(i>0) {
        if(threadIdx.x < i) {
            cache[threadIdx.x] += cache[threadIdx.x + i];
        }
        __syncthreads();
        i = i/2;
    }

    if(threadIdx.x == 0) {
        atomicAdd(c, cache[0]);
    }
}

// The sum of a subarray given an input array input of length N,
// and two indices S and E. S and E are inclusive

int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));


    int N=n;
    int h_a[N], h_c, S, E;
    int *d_a, *d_b, *d_c;

    CHECK_CUDA(cudaMalloc(&d_a, N*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_c, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_c, 0, sizeof(int)));

    // fill data in host device
    for(int i=0; i<N ;i++) {
        h_a[i] = (i*i + 3*i + 6)%13;
    }
    S = 3;
    E = N-3;

    CHECK_CUDA(cudaEventRecord(start, 0));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice));

    int blocksPerGrid = (E - S + 1 + threadsPerBlock - 1)/ threadsPerBlock;
    CHECK_CUDA(cudaMalloc(&d_b, blocksPerGrid*sizeof(int)));
    size_t shmem = threadsPerBlock*sizeof(int);
    // calculate per block sum for given interval
    perblockSum<<<blocksPerGrid, threadsPerBlock, shmem>>>(d_a, d_b, N, S, E);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Accumulate the total sum from all the blocks
    int blocksPerGrid1 = (blocksPerGrid + threadsPerBlock - 1)/ threadsPerBlock;
    size_t shmem2 = threadsPerBlock*sizeof(int);
    accumulateTotalSum<<< blocksPerGrid1, threadsPerBlock, shmem2>>>(d_b, d_c, blocksPerGrid);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"Elapsed time(in ms) : "<< elapsed_time<<endl;  // 0.416

    cout<<"Sub-array sum : "<<h_c<<endl;  // 249602

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}