#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;

#define n 64
#define m 256
#define k 256

#define CHECK_CUDA(call) do {                    \
    cudaError_t e = (call);                      \
    if(e != cudaSuccess) {                        \
        cout<<"CUDA Error: "<<cudaGetErrorString(e) \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl; \
        exit(1);                                     \
    }                                               \
} while(0)


// Counts the number of elements with the integer value p in 3D array of 32-bit integers
__global__ void elementCount3DArray(int* a, int* c, int N, int M, int K, int p){

    int x_c = threadIdx.x + blockDim.x * blockIdx.x;  // fastest changing dimension = K
    int y_r = threadIdx.y + blockDim.y * blockIdx.y;  // second fastest changing dimension = M
    int z_d = threadIdx.z + blockDim.z * blockIdx.z;  // depth = N

    //3D block of size NxMxK
    __shared__ int cache; //to store the frequency of element P
    // per block initialization using only single thread to avoid collisions
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {cache = 0;}
    __syncthreads();

    int count = 0; //per thread count flag

    if(x_c < K && y_r < M && z_d < N) { // global index condition check

        if(a[x_c + y_r*K + z_d*M*K] == p) {
            count = 1; // change the flag to 1 if element is found
        }
    }

    // accumulate all the threads values to shared memory per block
    atomicAdd(&cache, count);
    __syncthreads();


    // Now accumulate the output from per block to output
    if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z ==0) {
        atomicAdd(c, cache);
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
    int h_a[N*M*K], h_c, P;
    int *d_a, *d_c;

    CHECK_CUDA(cudaMalloc(&d_a, N*M*K*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_c, sizeof(int)));

    // fill data in host device
    for(int i=0; i<N*M*K ;i++) {
        h_a[i] = (i*i + 3*i - 46)%13;
    }
    P = 8;

    CHECK_CUDA(cudaEventRecord(start, 0));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N*M*K*sizeof(int), cudaMemcpyHostToDevice));

    dim3 block(16, 8, 8);  // 16*8*8 = 1024 (maximum)
    dim3 grid((K + 15)/16, (M + 7)/8, (N+7)/8);

    countElementIn3DArray<<<grid, block>>>(d_a, d_c, P);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"Elapsed time(in ms) : "<< elapsed_time<<endl;  // 0.46

    cout<<"Count of element k="<<P<<", is: "<<h_c<<endl;  // Count of element k=8, is: 44559

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}