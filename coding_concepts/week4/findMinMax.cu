#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;

#define N 9999

#define threadsPerBlock 256

#define CHECK_CUDA(call) do {                       \
    cudaError_t e = (call);                         \
    if(e!=cudaSuccess) {                                \
        cerr<<"CUDA Error : "<<cudaGetErrorString(e)    \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl;   \
        exit(1);                                    \
    }                                                   \
} while(0)

// this uses atomic operation function to calculate min & max values
__global__ void kernelGetMinMax(int* a, int* c){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x; // Since we are using fewer number of grids and threads
                                         // we require each thread process multiple datapoints.

    // While loop to traverse through the multiple grids
    while(tid<N) {
        atomicMin(&c[0], a[tid]); // if tempmin > a[tid] => tempmin = a[tid]
        atomicMax(&c[1], a[tid]); // if tempmax < a[tid] => tempmax = a[tid]
        tid += offset;
    }
}

int main() {

    // initialize and start recording time using CUDA Events
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));

    // define the vector for histogram computation for host and device
    int h_a[N], h_c[2];
    int *d_a, *d_c;

    // allocate memory for device variable
    CHECK_CUDA(cudaMalloc(&d_a, N*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_c, 2*sizeof(int)));

    // Initialize the vector with data between 0 and 255
    for(int i =0;i<N;i++) {
        h_a[i] = rand() % 9999;
    }
    h_c[0] = 99999;
    h_c[1] = -1;

    // transfer data to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_c, h_c, 2*sizeof(int), cudaMemcpyHostToDevice));

    int blocksPerGrid = 48 * 2;  // 48 is number of Multiprocessor in this GPU
                                // And, this number of grids are for best performance

    kernelGetMinMax<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_c);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy histogram result from device to Host
    CHECK_CUDA(cudaMemcpy(h_c, d_c, 2*sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"Elapsed time(in ms) : "<<elapsed_time<<endl;  // Elapsed time(in ms) : 11.8983


    cout<<"Min values = "<<h_c[0]<<", and Max value is "<<h_c[1]<<endl;

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}