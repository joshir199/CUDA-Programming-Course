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


// Do bitonic sort. Only applicable when the blocksize are power of 2
__global__ void bitonicSort(int* a, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ int cache[];

    if(tid<N) {
        cache[threadIdx.x] = a[tid];
    } else {
        cache[threadIdx.x] = INT_MAX;
    }
    __syncthreads();

    // Bitonic sort proceeds in log2(N) stages, where each stage doubles
    // the size of the sorted subsequence.
    // Outer loop : Represents the size of the subsequence currently being merged.
    // It doubles at each iterations
    for(int i = 2; i<= 2*blockDim.x ; i*=2) {

        // Determines whether the current group should be sorted in ascending or descending order.
        // For bitonic sequence, it should be alternating
        bool ascending = ((threadIdx.x & i/2) == 0);

        // Inner loop : Controls how far apart elements are compared and possibly swapped.
        for(int stride = i/2; stride>0; stride /= 2) {
            // Partner index : XOR determines which elements should be compared.
            // Ensures unique and symmetric pairing between threads.
            int ixj = threadIdx.x ^ stride;

            // compare the elements at these indexes
            if(ixj<blockDim.x && ixj > threadIdx.x) {
                // compare and swap for wrong order
                if((cache[threadIdx.x] > cache[ixj]) == ascending) {
                    int temp = cache[ixj];
                    cache[ixj] = cache[threadIdx.x];
                    cache[threadIdx.x] = temp;
                }
            }
            __syncthreads();
        }
    }

    if(tid<N) {
        a[tid] = cache[threadIdx.x];
    }

}

// Bitonic sort algorithm follows: Hierarchical Odd–Even Sort = Local Sort + Inter-Block Merge.
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
    bitonicSort<<<blocksPerGrid, threadsPerBlock, shmem>>>(d_a, N);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c, d_a, N*sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"Elapsed time(in ms) : "<< elapsed_time<<endl;  // 2.06  (faster than Odd-Even)

    for(int i = 0; i< 50 && i<N; i++) {
        cout<<"Odd-Even sorting result at i:"<<i<<", is: "<<h_c[i]<<", original array: "<<h_a[i]<<endl;
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}