#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;

#define N 32768
#define threadsPerBlock 256

#define CHECK_CUDA(call) do {                  \
    cudaError_t e = (call);                      \
    if(e != cudaSuccess) {                              \
        cerr<<"CUDA Error: "<<cudaGetErrorString(e)     \
        <<" on "<<__FILE__<<" at "<<__LINE__<<endl;     \
        exit(1);                                    \
    }                                               \
} while(0)


// Dot product function using shared memory and parallel reduction
__global__ void kernelDotProduct(float* a, float* b, float* c) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ int cache[threadsPerBlock];
    int cacheId = threadIdx.x;
    int temp = 0;

    if(tid<N) {
        temp = a[tid]*b[tid];
        tid += blockDim.x * gridDim.x;
    }
    cache[cacheId] = temp;
    __syncthreads();

    // parallel reduction to sum all the per-thread partial dot product.
    int i = threadsPerBlock/2;
    while(i>0){
        if(cacheId<i) {
            cache[cacheId] += cache[cacheId + i];
        }
        __syncthreads();
        i = i/2;
    }

    // get the final per-block dot product sum at 0-th index of each block
    if(cacheId == 0) {
        c[blockIdx.x] = cache[cacheId];
    }
}


// Function which uses zero-copy memory for read and write between host & Device.
float cudaZeroCopyFunction(int blocksPerGrid) {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));


    float *h_a = (float*)malloc(N*sizeof(float));
    float *h_b = (float*)malloc(N*sizeof(float));
    float *h_c = (float*)malloc(blocksPerGrid*sizeof(float));

    float *d_a, *d_b, *d_c;

    // define the host memory with proper flags to make it accessible by GPU directly for read/write
    CHECK_CUDA(cudaHostAlloc(&h_a, N*sizeof(float), cudaHostAllocMapped | cudaHostAllocWriteCombined));
    CHECK_CUDA(cudaHostAlloc(&h_b, N*sizeof(float), cudaHostAllocMapped | cudaHostAllocWriteCombined));
    CHECK_CUDA(cudaHostAlloc(&h_c, blocksPerGrid*sizeof(float), cudaHostAllocMapped)); //Only read is required

    for(int i=0; i<N; i++) {
        h_a[i] = (2*i - 99) * 0.00987 + 1.025;
        h_b[i] = i/512 - 0.54*i;
    }

    CHECK_CUDA(cudaEventRecord(start, 0));

    // While Host memory still have CPU pointers, Get th GPU device pointer for these memory
    CHECK_CUDA(cudaHostGetDevicePointer(&d_a, h_a, 0));
    CHECK_CUDA(cudaHostGetDevicePointer(&d_b, h_b, 0));
    CHECK_CUDA(cudaHostGetDevicePointer(&d_c, h_c, 0));

    // Run the kernel with GPU pointer for accessible Host memory
    kernelDotProduct<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize()); // Synchronize the memory b/w CPU & GPU.

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    float dt = 0;
    for(int i = 0; i< blocksPerGrid; i++) {
        dt += h_c[i];
    }
    cout<<" Dot product for zero-copy memory : "<< dt<<endl;

    CHECK_CUDA(cudaFreeHost(h_a));
    CHECK_CUDA(cudaFreeHost(h_b));
    CHECK_CUDA(cudaFreeHost(h_c));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return elapsed_time;
}


float normalMemoryCopyFunction(int blocksPerGrid) {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));


    float *h_a = (float*)malloc(N*sizeof(float));
    float *h_b = (float*)malloc(N*sizeof(float));
    float *h_c = (float*)malloc(blocksPerGrid*sizeof(float));

    float *d_a, *d_b, *d_c;
    // Allocate memory for the device
    CHECK_CUDA(cudaMalloc(&d_a, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, blocksPerGrid*sizeof(float)));

    for(int i=0; i<N; i++) {
        h_a[i] = (2*i - 99) * 0.00987 + 1.025;
        h_b[i] = i/512 - 0.54*i;
    }

    CHECK_CUDA(cudaEventRecord(start, 0));

    // transfer the data to device allocated memory
    CHECK_CUDA(cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, N*sizeof(float), cudaMemcpyHostToDevice));

    kernelDotProduct<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemcpy(h_c, d_c, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    float dt = 0;
    for(int i = 0; i< blocksPerGrid; i++) {
        dt += h_c[i];
    }
    cout<<" Dot product for normal Host to Device copy memory : "<< dt<<endl;

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return elapsed_time;
}


int main() {

    cudaDeviceProp prop;
    int deviceNumber;
    CHECK_CUDA(cudaGetDevice(&deviceNumber));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, deviceNumber));
    if(!prop.canMapHostMemory) {
        cout<<"Existing Device does not support host memory mapping";
    }

    int blocksPerGrid = (N + threadsPerBlock - 1)/threadsPerBlock;

    // If device supports mapping of host memory, set the flags for it.
    CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceMapHost));

    float t1 = normalMemoryCopyFunction(blocksPerGrid);
    cout<<" Elapsed time(in ms) for Normal Copy function dot product: "<<t1<<endl;  // 0.098464

    float t2 = cudaZeroCopyFunction(blocksPerGrid);
    cout<<" Elapsed time(in ms) for Zero Copy function dot product: "<<t2<<endl;    // 0.025888

    cout<<"Speed Up ratio: "<< t1/t2<< endl;   // 3.80

    return 0;
}