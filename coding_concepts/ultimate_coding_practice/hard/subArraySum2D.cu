#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;

#define n 768
#define m 768

#define CHECK_CUDA(call) do {                    \
    cudaError_t e = (call);                      \
    if(e != cudaSuccess) {                        \
        cout<<"CUDA Error: "<<cudaGetErrorString(e) \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl; \
        exit(1);                                     \
    }                                               \
} while(0)


__global__ void subarraySum2D(int* a, int* b, int rs, int re, int cs, int ce, int N, int M) {

    // Each block stores individual rows
    extern __shared__ int cache[];  //per block cache shared memory of size (ce - cs + 1) (column size)

    // threads cooperating to load the data into shared memory
    for(int i = threadIdx.x; i<(ce - cs + 1); i+=blockDim.x) {
        int gIdx = blockIdx.x * M + cs + i;
        cache[i] = a[gIdx];
    }
    __syncthreads();


    // since, Each thread is responsible for multiple data items, We can't apply parallel reduction directly
    // We will use Intra-Threads Sum using local threads
    int localSum = 0;
    for(int i = threadIdx.x; i<(ce - cs + 1); i+=blockDim.x) {
        localSum += cache[i];
    }
    __syncthreads();


    // Now, we will do per block sum using atomics
    // Initialize the per blockSum variable
    __shared__ int blockSum;
    if(threadIdx.x == 0) { blockSum = 0;}
    __syncthreads();

    // Get per block sum
    atomicAdd(&blockSum, localSum);
    __syncthreads();

    //Copy the sum into output array
    if(threadIdx.x == 0) {
        b[blockIdx.x] = blockSum;
    }
}



__global__ void accumulateTotalSum(int* a, int* c, int totalSize) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ int cache[];

    if(tid < totalSize) {
        cache[threadIdx.x] = a[tid];
    } else {
        cache[threadIdx.x] = 0;
    }
    __syncthreads();

    int i = blockDim.x / 2;
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



int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int N = n;
    int M = m;
    int row_s, row_e, col_s, col_e;

    // Matrix A[NxM]
    int h_a[N*M], h_c;
    int *d_a, *d_b, *d_c;

    CHECK_CUDA(cudaMalloc(&d_a, N*M*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_c, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_c, 0, sizeof(int)));

    // fill data in host device
    for(int i=0; i<N*M ;i++) {
        h_a[i] = (i*i + 3*i - 46)%13 ;
    }
    row_s = 1;
    row_e = N - 1;
    col_s = 0;
    col_e = M - 1;

    CHECK_CUDA(cudaEventRecord(start, 0));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N*M*sizeof(int), cudaMemcpyHostToDevice));

    // Do per Row sum first for the 2D array
    int threadsPerBlock = 256;
    int blocksPerGrid = row_e - row_s + 1; // Number of rows = number of blocks
    CHECK_CUDA(cudaMalloc(&d_b, blocksPerGrid*sizeof(int)));
    size_t shmem = (col_e - col_s + 1) * sizeof(int);

    subarraySum2D<<<blocksPerGrid, threadsPerBlock, shmem>>>(d_a, d_b, row_s, row_e, col_s, col_e, N, M);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    int h_b[blocksPerGrid];
    CHECK_CUDA(cudaMemcpy(h_b, d_b, blocksPerGrid*sizeof(int), cudaMemcpyDeviceToHost));

    // Accumulate the per Row sum into output
    int blocksPerGrid1 = (blocksPerGrid + threadsPerBlock - 1)/ threadsPerBlock;
    size_t shmem2 = threadsPerBlock * sizeof(int);

    accumulateTotalSum<<<blocksPerGrid1, threadsPerBlock, shmem2>>>(d_b, d_c, blocksPerGrid);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"Elapsed time(in ms) : "<< elapsed_time<<endl;  // 0.26

    for(int i=0 ; i< blocksPerGrid && i< 20; i++) {
        cout<<"per row sum : "<< h_b[i]<<endl;
    }

    cout<<"Total 2D subarray sum is: "<<h_c<<endl;  // Subarray sum = 249498

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}