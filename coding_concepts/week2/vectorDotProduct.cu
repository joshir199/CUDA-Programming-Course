#include <iostream>
#include <cuda_runtime.h>
using namespace std;

#define N 50000
// My GPU Nvidia RTX A4000 has this maxThreadsPerBlock value as 1024
#define threadsPerBlock 1024
#define min(a,b) (a<b?a:b)

// set the number of blocks to be smaller so that the CPU does not take enough time for the
// summation og results from each blocks to keep the optimal speed and memory usage. (32 by default)
const int blockPerGrid = min(32, (N + threadsPerBlock - 1)/threadsPerBlock);


// we will be using shared memory and synchronisation mechanism to do computation
__global__ void dotProduct(float* a, float* b, float* c) {
    // Each block will have private copy of shared memory.
    // Therefor, only all threads per block will be declared here
    __shared__ float cache[threadsPerBlock];
    float temp = 0;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIdx = threadIdx.x; // Since, each block has its own shared memory

    while(tid < N) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIdx] = temp; // store the temporary sum of corresponding elements per thread
    __syncthreads(); // ensure all the value read/write is completed on all threads (to avoid any race condition)

    // Let's add each corresponding element product as reduction operation to calculate dot product.
    // Now, each threads contains the values that needs to be added for final result.
    // We will use parallel computation by collecting the values from threads in binary tree format

    int i = threadsPerBlock/2;
    while(i!=0) {
        // combine the corresponding values(at > cacheIdx) when the cacheIdx is less than the half of the threads
        if(cacheIdx < i) {
            cache[cacheIdx] = cache[cacheIdx] + cache[cacheIdx + i];
            // do not use __syncthreads() here (or within conditional statements), as it may cause thread divergence issue
        }
        __syncthreads(); // ensure all the value read/write is completed on all threads and avoid "thread divergence"
        // half the number of threads
        i = i/2 ;
    }

    // Now, each block is having the resultant sum of the section of dot product results
    // We just need to sum it, but we can do that on CPU too as number of blocks will be very less
    // copy the those partition answer values to c vector

    if(cacheIdx==0) {
        // copy at the blockIdx for accumulated results from each block
        c[blockIdx.x] = cache[cacheIdx];
    }
}


int main() {

    //define vector arrays on host
    float h_a[N], h_b[N], h_c[blockPerGrid];
    //define pointers on device
    float *d_a, *d_b, *d_c;

    //allocate memory on device
    cudaMalloc(&d_a, N*sizeof(float));
    cudaMalloc(&d_b, N*sizeof(float));
    cudaMalloc(&d_c, N*sizeof(float));

    // assign values for input vectors a and b on host
    for(int i=0; i<N;i++) {
        h_a[i] = 1.56*i;
        h_b[i] = i*i - 1.99;
    }

    // copy the vector a and b from host to device
    cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N*sizeof(float), cudaMemcpyHostToDevice);

    // call the kernel to do Dot Product
    dotProduct<<<blockPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);

    // copy the resultant output from Device to host
    cudaMemcpy(h_c, d_c, blockPerGrid*sizeof(float), cudaMemcpyDeviceToHost);

    // sum the resultant values to get final answer
    float dot_product = 0;
    for(int i = 0 ; i<blockPerGrid;i++){
        dot_product += h_c[i];
    }

    // print the final result
    cout<<"N: "<<N<<endl;
    cout<<"threadsPerBlock: "<<threadsPerBlock<<endl;
    cout<<"blockPerGrid: "<<blockPerGrid<<endl;
    cout<<"dot product value: "<<dot_product<<endl;
    cout<<"Accumulated results at different blocks:"<<endl;
    for(int i=0;i<blockPerGrid;i++) {
        cout<<"at block "<<i<<": "<<h_c[i]<<endl;
    }

    // free the memory from the device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}