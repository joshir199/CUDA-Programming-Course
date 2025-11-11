#include <iostream>
#include <cuda_runtime.h>
using namespace std;

#define N 1100
#define threadsPerBlock 128

__global__ void kernelVectorNorm(int* a, int* c) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ int cache[threadsPerBlock];
    int cacheId = threadIdx.x;
    int temp = 0; // temporary variable to store partialSums per threads
    if(tid<N) {
        temp = a[tid] * a[tid];
        tid += blockDim.x * gridDim.x;
    }
    cache[cacheId] = temp;
    __syncthreads();

    // parallel reduction within each block
    int i = blockDim.x/2;
    while(i>0) {
        if(i > cacheId) {
            cache[cacheId] += cache[cacheId + i];
        }
        __syncthreads();
        i = i/2;
    }

    if(cacheId ==0) {
        // copy at the blockIdx for accumulated results from each block
        c[blockIdx.x] = cache[cacheId];
    }
}

int main() {
    int h_a[N]; // define variable to host

    int *d_a, *d_c;   // define variable for device

    // allocate memory to variable for device
    cudaMalloc(&d_a, N*sizeof(int));
    cudaMalloc(&d_c, N*sizeof(int));

    // initialize he matrix A and B
    for(int i =0; i<N; i++) {
        h_a[i] = 2*i - 1;
    }

    // transfer data from host to Device
    cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice);

    int blockPerGrid = min(32, (N + threadsPerBlock-1)/threadsPerBlock);
    // call the kernel to execute on GPU
    kernelVectorNorm<<< blockPerGrid, threadsPerBlock>>>(d_a, d_c);

    int partialSum[blockPerGrid];
    // collect the result from Device
    cudaMemcpy(partialSum, d_c, blockPerGrid*sizeof(int), cudaMemcpyDeviceToHost);

    float squareSum = 0;
    for(int i = 0 ; i<blockPerGrid;i++) {
        squareSum += partialSum[i];
    }
    // print the result
    cout<<"blockPerGrid: "<<blockPerGrid<<", squareSum : " << squareSum <<", vector norm  : "<<sqrtf(squareSum) <<endl;


    // free the global memory
    cudaFree(d_a);
    cudaFree(d_c);

    return 0;
}