#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;

#define N 8 //row
#define M 16  //column
#define threadsPerBlock 64

#define CHECK_CUDA(call) do {                    \
    cudaError_t e = (call);                      \
    if(e != cudaSuccess) {                        \
        cout<<"CUDA Error: "<<cudaGetErrorString(e) \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl; \
        exit(1);                                     \
    }                                               \
} while(0)


__global__ void kernelMatrixTranspose(int* a, int* c) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // int offset  = x + y * blockDim.x * gridDim.x; [global width of your grid in the X dimension.]
    // int offset_t = y + x * blockDim.y * gridDim.y;[global height of your grid in the Y dimension.]
    //
    if(x<M && y<N) {
        int offset = x + y*M;  // row indexing = index in x-dir + column_size * index in y_dir
        int offset_t = y + x*N;  // row indexing = index in y-dir + row_size * index in x_dir
        c[offset_t] = a[offset];
    }
}


int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));

    int h_a[N*M], h_c[M*N];
    int *d_a, *d_c;

    CHECK_CUDA(cudaMalloc(&d_c, M*N*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_a, N*M*sizeof(int)));

    for(int i =0;i<M*N;i++){
        h_a[i] = rand() % 200;
    }

    CHECK_CUDA(cudaMemcpy(d_a, h_a, M*N*sizeof(int), cudaMemcpyHostToDevice));

    // threadsPerBlock threads in x, threadsPerBlock threads in y → threadsPerBlock^2 threads per block
    dim3 block(threadsPerBlock, threadsPerBlock);
    dim3 grid((M+threadsPerBlock-1)/threadsPerBlock,(N+threadsPerBlock-1)/threadsPerBlock);
    kernelMatrixTranspose<<<grid, block>>>(d_a, d_c);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c, d_c, N*M*sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"Elapsed time(in ms) : "<<elapsed_time<<endl;

    for(int i =0;i<M*N;i++) {
        cout<<"Matrix A : "<<h_a[i]<<" its transpose: "<<h_c[i]<<endl;
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_c));

    return 0;
}