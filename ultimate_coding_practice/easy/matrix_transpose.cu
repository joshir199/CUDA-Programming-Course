#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;

#define rows 1024
#define cols 256
#define threadsPerBlock 16

#define CHECK_CUDA(call) do {                    \
    cudaError_t e = (call);                      \
    if(e != cudaSuccess) {                        \
        cout<<"CUDA Error: "<<cudaGetErrorString(e) \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl; \
        exit(1);                                     \
    }                                               \
} while(0)

__global__ void matrix_transpose_kernel(const int* input, int* output){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // output Matrix = cols x rows
    // the matrix are spread and allocated in form of matrix within the 2D matrix shaped blocks
    if(x < cols && y<rows) {
        int offset = x + y *cols;
        int offset_t = y + x*rows;
        output[offset_t] = input[offset];
    }
}


int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));

    int size = rows * cols;
    // Input matrix A = rows x cols;
    int h_a[size], h_c[size];
    int *d_a, *d_c;

    CHECK_CUDA(cudaMalloc(&d_c, size*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_a, size*sizeof(int)));

    for(int i =0;i<size;i++){
        h_a[i] = rand() % 200;
    }

    CHECK_CUDA(cudaMemcpy(d_a, h_a, size*sizeof(int), cudaMemcpyHostToDevice));

    // threadsPerBlock threads in x, threadsPerBlock threads in y → threadsPerBlock^2 threads per block
    dim3 block(threadsPerBlock, threadsPerBlock);
    dim3 grid((cols+threadsPerBlock-1)/threadsPerBlock,(rows+threadsPerBlock-1)/threadsPerBlock);
    matrix_transpose_kernel<<<grid, block>>>(d_a, d_c);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c, d_c, size*sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"Elapsed time(in ms) : "<<elapsed_time<<endl;

    for(int i =0;i<100;i++) {
        cout<<"Matrix A : "<<h_a[i]<<" its transpose: "<<h_c[i]<<endl;
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_c));

    return 0;
}