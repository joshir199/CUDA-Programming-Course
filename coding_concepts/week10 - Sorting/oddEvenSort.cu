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

// Sort the array into sorted blocks in ascending order
__global__ void oddEvenSorting(int* a, int N) {
    int offset = blockIdx.x * blockDim.x;
    extern __shared__ int cache[];
    int tid = threadIdx.x + offset;

    if(tid<N) {
        cache[threadIdx.x] = a[tid];
    } else {
        cache[threadIdx.x] = INT_MAX;
    }
    __syncthreads();

    // threadIdx range is [0, blockDim.x]
    for(int phase = 0; phase<blockDim.x; phase++) {

        if(threadIdx.x + 1 < blockDim.x) {  // per block
            // Even phase ([0,1],[2,3],[4,5] ...
            if((phase%2==0) && (threadIdx.x%2==0)) {
                if(cache[threadIdx.x + 1]<cache[threadIdx.x]) {
                    int temp = cache[threadIdx.x];
                    cache[threadIdx.x] = cache[threadIdx.x + 1];
                    cache[threadIdx.x + 1] = temp;
                }
            }

            // Odd phase ([1,2],[3,4],[5,6] ...
            if((phase%2==1) && (threadIdx.x%2==1)) {
                if(cache[threadIdx.x + 1]<cache[threadIdx.x]) {
                    int temp = cache[threadIdx.x];
                    cache[threadIdx.x] = cache[threadIdx.x + 1];
                    cache[threadIdx.x + 1] = temp;
                }
            }
        }
        __syncthreads();
    }

    if(tid<N){
        a[tid] = cache[threadIdx.x];
    }

}

// Merge the sorted blocks globally
__global__ void mergeSortedBlock(int* a, int N, int blockSize, int blockPhase) {
    int stA, stB;
    if(blockPhase %2 ==0) { // Even phase block [0,1], [2,3] ...
        stA = 2 * blockIdx.x * blockSize;
        stB = stA + blockSize;  // next block starting
    } else {  // Odd phase block [1,2], [3,4] ...
        stA = (2 * blockIdx.x + 1) * blockSize;
        stB = stA + blockSize;
    }

    if(stB>=N) {return;}

    // define shared memory to store and do sorting of combined block
    extern __shared__ int cache[];
    int totalSize = 2*blockSize;

    for(int i = threadIdx.x; i< totalSize; i+= blockDim.x) {
        if(stA + i < N) {
            cache[i] = a[stA + i];
        } else {
            cache[i] = INT_MAX;
        }
    }
    __syncthreads();


    // threadIdx range is [0, totalSize]
    for(int phase = 0; phase<totalSize; phase++) {

        // Even phase ([0,1],[2,3],[4,5] ...
        if(phase%2==0) {
            for (int idx = threadIdx.x; idx + 1 < totalSize; idx += blockDim.x) {

                if((idx % 2 == 0) && cache[idx + 1] < cache[idx]) {
                    int temp = cache[idx];
                    cache[idx] = cache[idx + 1];
                    cache[idx + 1] = temp;
                }
            }
        }

        // Odd phase ([1,2],[3,4],[5,6] ...
        if(phase%2==1) {
            for (int idx = threadIdx.x; idx + 1 < totalSize; idx += blockDim.x) {

                if((idx % 2 == 1) && cache[idx + 1] < cache[idx]) {
                    int temp = cache[idx];
                    cache[idx] = cache[idx + 1];
                    cache[idx + 1] = temp;
                }
            }
        }
        __syncthreads();
    }

    for(int i = threadIdx.x; i< totalSize; i+= blockDim.x) {
        if(stA + i < N) {
            a[stA + i] = cache[i];
        }
    }

}

// Odd-Even sort algorithm follows: Hierarchical Odd–Even Sort = Local Sort + Inter-Block Merge.
// It can be easily used for Multi-GPU extensions with distributed systems.
// Time complexity : O(N*(logN)^2) and worst case : O(N^2) 

int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int N = n;
    int h_a[N], h_c[N];
    int *d_a;

    CHECK_CUDA(cudaMalloc(&d_a, N*sizeof(int)));

    // fill data in host device
    for(int i=0; i<N ;i++) {
        h_a[i] = rand() % 999;
    }

    CHECK_CUDA(cudaEventRecord(start, 0));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice));

    // local sort per block
    int blocksPerGrid = (N + threadsPerBlock - 1)/threadsPerBlock;
    size_t shmem = threadsPerBlock * sizeof(int);
    oddEvenSorting<<<blocksPerGrid, threadsPerBlock, shmem>>>(d_a, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());


    // Global Merge Part
    // Only half pairs of block will go per phase (either even or odd)
    int mergeBlocksPerGrid = (blocksPerGrid + 1)/2;
    size_t shmem2 = 2 * threadsPerBlock * sizeof(int);
    for(int p = 0; p<blocksPerGrid; p++) {
        mergeSortedBlock<<<mergeBlocksPerGrid, threadsPerBlock, shmem2>>>(d_a, N, threadsPerBlock, p);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    CHECK_CUDA(cudaMemcpy(h_c, d_a, N*sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"Elapsed time(in ms) : "<< elapsed_time<<endl;  // 6617.26

    for(int i = 0; i< 50 && i<N; i++) {
        cout<<"Odd-Even sorting result at i:"<<i<<", is: "<<h_c[i]<<", original array: "<<h_a[i]<<endl;
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}