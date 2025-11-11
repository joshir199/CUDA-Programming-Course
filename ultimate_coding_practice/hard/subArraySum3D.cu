#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;

#define n 64
#define m 64
#define k 128
#define threadsPerBlock 512 // assuming max value of each dimension is <512

#define CHECK_CUDA(call) do {                    \
    cudaError_t e = (call);                      \
    if(e != cudaSuccess) {                        \
        cout<<"CUDA Error: "<<cudaGetErrorString(e) \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl; \
        exit(1);                                     \
    }                                               \
} while(0)


// Get per 2D Grid subarray sum
__global__ void per2DGridSubArraySum(int* a, int* c, int dep_id, int rs, int re, int cs, int ce, int M, int K) {

    // Each block stores individual rows
    extern __shared__ int cache[];  //per block cache shared memory of size (ce - cs + 1) (column size)
    int perGridOffset = dep_id * M * K ;  // M*K = size of 2D Grid

    int rowStart = perGridOffset  +  (blockIdx.x + rs) * K;  // K = column size (M<500 & K<500)
    // enough threads to load the data into shared memory at once
    if(threadIdx.x < (ce - cs + 1)) {
        int gIdx = rowStart + cs + threadIdx.x;
        cache[threadIdx.x] = a[gIdx];
    } else {
        cache[threadIdx.x] = 0;
    }
    __syncthreads();

    // Parallel Reduction for per block sum
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            cache[threadIdx.x] += cache[threadIdx.x + i];
        }
        __syncthreads();
    }

    //Copy all block's sum into output variable using atomic operation
    if(threadIdx.x == 0) {
        atomicAdd(c, cache[0]);
    }
}


int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    //  input 3D array
    /*
    input [[[1, 2, 3, 2],
            [4, 5, 1, 1],
            [2, 2, 3, 4]],
            [[1, 1, 1, 4],
             [2, 6, 2, 1],
             [3, 2, 8, 2]]]
       N = 2, M = 3, K = 4

       // Here, K is fastest changing dimension -> column
    */

    int N=n;
    int M=m;
    int K=k;
    int row_s, row_e, col_s, col_e, dep_s, dep_e;
    int h_a[N*M*K], h_c;
    int *d_a, *d_c;

    CHECK_CUDA(cudaMalloc(&d_a, N*M*K*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_c, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_c, 0, sizeof(int)));

    // fill data in host device
    for(int i=0; i<N*M*K ;i++) {
        h_a[i] = (rand())%9;
    }
    row_s = 1;
    row_e = M - 1;
    col_s = 0;
    col_e = K - 1;
    dep_s = 5;
    dep_e = N-9;

    CHECK_CUDA(cudaEventRecord(start, 0));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N*M*K*sizeof(int), cudaMemcpyHostToDevice));


    // Do per depth sum first for the 3D array
    int numBlocks = dep_e - dep_s + 1; // Number of 2D grids
    size_t shmem = threadsPerBlock * sizeof(int);
    int numRowBlocks = row_e - row_s + 1; // Number of rows = number of blocks
    int h_b[numBlocks];

    for(int i = 0; i<numBlocks; i++) {
        per2DGridSubArraySum<<<numRowBlocks, threadsPerBlock, shmem>>>(d_a, d_c, i + dep_s, row_s, row_e, col_s, col_e, M, K);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemset(d_c, 0, sizeof(int)));
        h_b[i] = h_c;
    }

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"Elapsed time(in ms) : "<< elapsed_time<<endl;  // 0.84

    for(int i=0 ; i< numBlocks && i< 20; i++) {
        cout<<"per 2D Grid sum : "<< h_b[i]<<endl;
    }
    h_c = 0;
    for(int i=0 ; i< numBlocks; i++) {
        h_c += h_b[i];
    }

    cout<<"Total 3D subarray sum is: "<<h_c<<endl;  // Subarray sum = 1649443

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}