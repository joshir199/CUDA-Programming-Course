#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;


#define N 999999
#define F_len 2047  // filter size
#define threadsPerBlock 256

#define CHECK_CUDA(call) do {                    \
    cudaError_t e = (call);                      \
    if(e != cudaSuccess) {                        \
        cout<<"CUDA Error: "<<cudaGetErrorString(e) \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl; \
        exit(1);                                     \
    }                                               \
} while(0)

__constant__ float kernel[F_len];

// 1D convolution of array using the filter of length len.
// Here, convolution is  in 1D direction.
// e.g: C[i] = Sum(x[i+j]*f[j] For j in [-F_len/2, F_len/2]).
// thus, it will only have both left and right halo.
__global__ void convolutionFull(float* a, float* c, int SharedCacheSize) {

    extern __shared__ float cache[];  // Ensure total memory < 64KB

    int halo = F_len/2;
    // globalId = threadId in block + block_start_Idx - left_halo
    int tileStart = blockIdx.x * blockDim.x - halo;


    // When the cache size is very large, each thread can cooperate to load the elements
    // [Coalesced], where each threads reads multiple inputs into the shared memory
    for(int i = threadIdx.x ; i< SharedCacheSize; i += blockDim.x) {
        int globalIdx = i + tileStart;
        if(globalIdx >=0 && globalIdx < N) {
            cache[i] = a[globalIdx];
        } else {
            cache[i] = 0.0f;  // to pad the block with halo number of zeros on both side
        }
    }
    __syncthreads();  // ensure all shared memory operations are completed


    // output Idx = threadId + block output length
    int outIdx = threadIdx.x + blockIdx.x * blockDim.x;

    // this will stop the threadIdx to overflow
    if(outIdx<N) {
        float sum = 0.0f;
        for(int i = 0;i<F_len; i++) { // already padded with zero
            sum += kernel[i] * cache[threadIdx.x + i];
        }

        c[outIdx] = sum;
    }
}


int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));


    float h_a[N], h_c[N], h_kernel[F_len];
    float *d_a, *d_c;

    CHECK_CUDA(cudaMalloc(&d_a, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, N*sizeof(float)));

    // fill data in host device
    for(int i=0; i<N ;i++) {
        h_a[i] = (rand() % 90) * 0.018f;
    }

    // fill the constant memory array on host
    for(int i=0;i<F_len;i++) {
        if(i<F_len/2) {
            h_kernel[i] = 1.0f;
        } else if(i==F_len/2) {
            h_kernel[i] = 0;
        } else {
            h_kernel[i] = -1.0f;
        }
    }

    CHECK_CUDA(cudaEventRecord(start, 0));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice));
    // Transfer constant memory data from host to constant memory
    CHECK_CUDA(cudaMemcpyToSymbol(kernel, h_kernel, F_len*sizeof(float)));

    int blocksPerGrid = (N + threadsPerBlock - 1)/threadsPerBlock;
    int SharedTileSize = threadsPerBlock + F_len - 1;
    size_t shmem = (SharedTileSize) * sizeof(float);

    convolutionFull<<<blocksPerGrid, threadsPerBlock, shmem>>>(d_a, d_c, SharedTileSize);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c, d_c, N*sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"Elapsed time(in ms) : "<< elapsed_time<<endl;   // 3.41

    for(int i = 0; i< 50 && i<N; i++) {
        cout<<"Convolution result at i:"<<i<<", is: "<<h_c[i]<<endl;
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}