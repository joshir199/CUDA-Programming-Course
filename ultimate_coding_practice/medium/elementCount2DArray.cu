#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;

#define N 1024
#define M 1024
#define threadsDim 16

#define CHECK_CUDA(call) do {                    \
    cudaError_t e = (call);                      \
    if(e != cudaSuccess) {                        \
        cout<<"CUDA Error: "<<cudaGetErrorString(e) \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl; \
        exit(1);                                     \
    }                                               \
} while(0)


// Counts the number of elements with the integer value k in an array of 32-bit integers
__global__ void countElementIn2DArray(int* input, int* output, int K) {

    int x_c = threadIdx.x + blockIdx.x * blockDim.x;
    int y_r = threadIdx.y + blockIdx.y * blockDim.y;

    __shared__ int cache;  //per block cache shared memory
    // per block initialization using only single thread to avoid collisions
    if(threadIdx.x == 0 && threadIdx.y ==0 ) { cache = 0;}
    __syncthreads();

    int count = 0; //per thread count variable to store the frequency

    if(x_c < M && y_r < N) { // global index condition check
        if(input[y_r * M + x_c] == K) {
            count = 1;
        }
    }

    // update the count values of each thread into the cache of each block
    atomicAdd(&cache, count);
    __syncthreads();

    // now, add the counts from each block to output
    if(threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(output, cache);
    }

}


int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));


    int h_a[N*M], h_c, k;
    int *d_a, *d_c;

    CHECK_CUDA(cudaMalloc(&d_a, N*M*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_c, sizeof(int)));

    // fill data in host device
    for(int i=0; i<N*M ;i++) {
        h_a[i] = (i*i + 3*i - 46)%13;  //(rand() % 90) * 0.018f ;
    }
    k = 8;

    CHECK_CUDA(cudaEventRecord(start, 0));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N*M*sizeof(int), cudaMemcpyHostToDevice));

    dim3 block(threadsDim, threadsDim);
    dim3 grid((M + threadsDim - 1)/threadsDim, (N + threadsDim - 1)/threadsDim);

    countElementIn2DArray<<<grid, block>>>(d_a, d_c, k);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"Elapsed time(in ms) : "<< elapsed_time<<endl;  // 0.46

    cout<<"Count of element k="<<k<<", is: "<<h_c<<endl;  // Count of element k=8, is: 44559

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}