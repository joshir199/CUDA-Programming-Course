#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;


#define N 999999
#define F_len 255
#define B 64
#define threadsPerBlock (B + F_len - 1)

#define CHECK_CUDA(call) do {                    \
    cudaError_t e = (call);                      \
    if(e != cudaSuccess) {                        \
        cout<<"CUDA Error: "<<cudaGetErrorString(e) \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl; \
        exit(1);                                     \
    }                                               \
} while(0)

__constant__ float filter[F_len];

// 1D convolution of array using the filter of length len.
// Here, convolution is  in 1D direction.
// e.g: C[i] = Sum(x[i+j]*f[j] For j in [-F_len/2, F_len/2]).
// thus, it will only have both left and right halo.
__global__ void convolution(float* a, float* c) {


    __shared__ float cache[threadsPerBlock];
    int cacheIdx = threadIdx.x;
    int halo = F_len/2;
    // globalId = threadId in block + block_start_Idx - left_halo
    int gIdx = threadIdx.x + blockIdx.x * B - halo;
    // check when globalId is valid index, copy from the input array
    if(gIdx >=0 && gIdx < N) {
        cache[cacheIdx] = a[gIdx];
    } else{
        // for invalid index Id, add 0 as padding.
        cache[cacheIdx] = 0.0f;
    }
    __syncthreads();

    // calculate convolution for B inputs per block
    // It will produce only B outputs per block
    if(threadIdx.x < B) {
        // output Idx = threadId + block output length
        int outIdx = threadIdx.x + blockIdx.x * B;
        // this will stop the threadIdx to overflow
        if (outIdx<N) {

            float sum = 0.0f;
            for(int i = 0;i<F_len; i++) { // already padded with zero
                sum += filter[i] * cache[threadIdx.x + i];
            }
            c[outIdx] = sum;
        }

    }
}


int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));

    float h_a[N], h_c[N], h_kernel[F_len];
    float *d_a, *d_c;

    CHECK_CUDA(cudaMalloc(&d_a, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, N*sizeof(float)));

    // fill data in host device
    for(int i=0; i<N ;i++) {
        h_a[i] = 1.0f; //(rand() % 90) * 0.018f;
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

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice));
    // Transfer constant memory data from host to constant memory
    CHECK_CUDA(cudaMemcpyToSymbol(filter, h_kernel, F_len*sizeof(float)));

    int blocksPerGrid = (N + threadsPerBlock - 1)/threadsPerBlock;
    //size_t shmem = (threadsPerBlock + F_len - 1) * sizeof(float);

    convolution<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_c);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c, d_c, N*sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"Elapsed time(in ms) : "<< elapsed_time<<endl;

    for(int i = 0; i< 50 && i<N; i++) {
        cout<<"Convolution result at i:"<<i<<", is: "<<h_c[i]<<endl;
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}