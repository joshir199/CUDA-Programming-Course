#include <iostream>
#include <cuda_runtime.h>
using namespace std;

#define N 1101
#define threadsPerBlock 128

/*
Key Concepts
Hierarchical Scan (Blelloch method)
1. Divide & Conquer
  Break large scan into per-block scans. (Do offset Doubling Pattern)
  Aggregate block totals. (Just call the same kernel)
  Distribute back.

2. Two-Level Hierarchy
  Level 1: Threads within a block (fast, shared memory).
  Level 2: Blocks across the grid (slower, needs second pass).

3. Work Complexity
  Still O(N), but parallel steps = O(log N).
*/

//Only when the N is less than totals threads per block
__global__ void kernelPrefixSumPerBlock(int* a, int* c, int* blocks) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x; // thread Ids assuming multiple blocks
    __shared__ int cache[threadsPerBlock]; // shared memory
    int cacheId = threadIdx.x; // shared memory Id per block

    // copy the array data into shared memory
    if(tid<N) {
        cache[cacheId] = a[tid];
    } else {
        cache[cacheId] = 0;
    }
    __syncthreads();

    // Up-sweep phase
    //Do the offset doubling pattern sum for each element
    int i = 1;
    while(i<N) {
        int temp = 0;
        if(cacheId>=i){
            temp = cache[cacheId - i];
        }
        __syncthreads();
        cache[cacheId] += temp;
        __syncthreads();
        i = i*2;
    }

    // The resultant sum will have inclusive prefix sum for each element
    // down-sweep phase
    // Shift each item to right to get exclusive sum
    int perblockSum = cache[cacheId];
    __syncthreads();
    if(cacheId == 0) {
        cache[cacheId] = 0;
    } else {
        cache[cacheId] = perblockSum - a[tid];
    }
    __syncthreads();

    // write the output to the array
    if(tid<N){
        c[tid] = cache[cacheId];
    }
    // Last element of block = block total sum
    // store the perblockSum for further processing
    if(cacheId == blockDim.x - 1){
        blocks[blockIdx.x] = perblockSum;
    }
}

// To add the accumulated prefix sum per block to each elements
__global__ void blockSumAddition(int* a, int* blocks) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid<N) {
        a[tid] += blocks[blockIdx.x];
    }
}



int main() {
    int h_a[N], h_c[N]; // define variable to host

    int *d_a, *d_c, *blockSum, *blockPrefix, *totalSum;   // define variable for device

    int blockPerGrid = min(32, (N + threadsPerBlock-1)/threadsPerBlock);

    // allocate memory to variable for device
    cudaMalloc(&d_a, N*sizeof(int));
    cudaMalloc(&d_c, N*sizeof(int));
    cudaMalloc(&blockSum, blockPerGrid*sizeof(int));
    cudaMalloc(&blockPrefix, blockPerGrid*sizeof(int));
    cudaMalloc(&totalSum, sizeof(int));

    // initialize he vector A
    for(int i =0; i<N; i++) {
        h_a[i] = 2*i - 101;
    }

    // transfer data from host to Device
    cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice);
    // call the kernel to execute on GPU
    // Step 1: Get the prefix sum per block
    kernelPrefixSumPerBlock<<< blockPerGrid, threadsPerBlock>>>(d_a, d_c, blockSum);

    // collect the result from Device
    cudaMemcpy(h_c, d_c, N*sizeof(int), cudaMemcpyDeviceToHost);
    int blocks[blockPerGrid];
    cudaMemcpy(blocks, blockSum, blockPerGrid*sizeof(int), cudaMemcpyDeviceToHost);


    // Step 2: Get the resultant sum to be added for each block elements
    // Its similar to do exclusive prefix sum for each block sum.
    kernelPrefixSumPerBlock<<< 1, blockPerGrid>>>(blockSum, blockPrefix, totalSum);
    int blockPrefixSum[blockPerGrid];
    int totalsum[1];
    cudaMemcpy(blockPrefixSum, blockPrefix, blockPerGrid*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(totalsum, totalSum, sizeof(int), cudaMemcpyDeviceToHost);

    // step 3: Get the final prefix sum for each element by combining its prior block sum
    blockSumAddition<<<blockPerGrid, threadsPerBlock>>>(d_c, blockPrefix);

    int result[N];
    cudaMemcpy(result, d_c, N*sizeof(int), cudaMemcpyDeviceToHost);

    cout<<"Total Sum: "<< totalsum <<endl;
    for(int i = 0 ; i<blockPerGrid;i++) {
        // print the result
        cout<<"Per Block Sum at i= " << i << " : " << blocks[i] << ", blockPrefix : " << blockPrefixSum[i]<<endl;
    }
    for(int i = 0 ; i<N;i++) {
        // print the result
        cout<<"Exclusive Prefix Sum at i= " << i << " : " << result[i] << ", Previous: " << h_c[i]<<endl;
    }

    // free the global memory
    cudaFree(d_a);
    cudaFree(d_c);
    cudaFree(blockSum);
    cudaFree(blockPrefix);
    cudaFree(totalSum);

    return 0;
}