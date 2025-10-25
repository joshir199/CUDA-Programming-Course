#include <iostream>
#include <cstdlib>     // for rand()
#include <ctime>       // for seeding rand()
#include <cuda_runtime.h>
using namespace std;

// width and height are same
#define N 64

#define CHECK_CUDA(call) do {                            \
    cudaError_t e = (call);                              \
    if(e!=cudaSuccess) {                                 \
        cerr<<"CUDA Error: "<<cudaGetErrorString(e)      \
        <<" at "<<__FILE__<<":"<<__LINE__<<endl;         \
        exit(1);                                         \
    }                                                    \
} while(0)

__global__ void kernelIncreaseBrightness(int* a, int*c, int clamp) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    if(x<N && y<N) {
        int temp = a[offset] + clamp;
        if(temp > 255) {
            c[offset] = 255;
        } else {
            c[offset] = temp;
        }
    }
}


int main() {

    srand((unsigned)time(NULL));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));

    int h_a[N*N], h_c[N*N], clamp;
    int *d_a, *d_c;

    CHECK_CUDA(cudaMalloc(&d_a, N*N*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_c, N*N*sizeof(int)));

    clamp = 149;
    for(int i =0;i<N*N;i++){
        h_a[i] = rand() % 150;
    }

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N*N*sizeof(int), cudaMemcpyHostToDevice));

    dim3 block(16, 16); // 256 threads in a block
    dim3 grid((N+15)/16, (N+15)/16); // number of blocks in a grid

    kernelIncreaseBrightness<<<grid, block>>>(d_a, d_c, clamp);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c, d_c, N*N*sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"Elapsed Time(in ms) : "<< elapsed_time<<endl;

    for(int i =0; i<N*N;i++){
        cout<<"Pixel values at i: "<<h_c[i]<<", before: "<<h_a[i]<<endl;
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_c));

    return 0;
}