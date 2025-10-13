#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;


#define N 999999
#define F_len 2047  // filter size
#define threadsPerBlock 256
#define M (N-F_len+1) //output size

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
// Here, convolution is only in forward direction without any padding.
// e.g: C[i] = Sum(x[i+j]*f[j] For j in [0, F_len-1].
// thus, it will only have right halo.
__global__ void convolution1DNoPadding(float* input, float* output, int SharedTileSize)
{

  extern __shared__ float cache[];
  int tileStart = blockIdx.x * blockDim.x; // starting index for each block

  for(int i = threadIdx.x ; i<SharedTileSize; i += blockDim.x) { // check for shared memory length limit

    int globalIdx = i + tileStart;
    if(globalIdx < N) {  // check for input length limit
      cache[i] = input[globalIdx];
    } else {
      cache[i] = 0.0f;
    }
  }
  __syncthreads();


  int outputId = threadIdx.x + tileStart; // get globalId for output array by combining per block result
  if(outputId < M) {  // check for output length limit
    float sum = 0.0f;
    for(int i =0; i<F_len; i++) {
      sum += filter[i] * cache[i + threadIdx.x];
    }
    output[outputId] = sum;
  }
}


int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));



    float h_a[N], h_c[M], h_kernel[F_len];
    float *d_a, *d_c;

    CHECK_CUDA(cudaMalloc(&d_a, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, M*sizeof(float)));

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
    CHECK_CUDA(cudaMemcpyToSymbol(filter, h_kernel, F_len*sizeof(float)));

    int blocksPerGrid = (M + threadsPerBlock - 1)/threadsPerBlock;
    int SharedTileSize = threadsPerBlock + F_len - 1;
    size_t shmem = SharedTileSize * sizeof(float);

    convolution1DNoPadding<<<blocksPerGrid, threadsPerBlock, shmem>>>(d_a, d_c, SharedTileSize);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c, d_c, M*sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"Elapsed time(in ms) : "<< elapsed_time<<endl;   // 3.54

    for(int i = 0; i< 50 && i<M; i++) {
        cout<<"Convolution result at i:"<<i<<", is: "<<h_c[i]<<endl;
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}