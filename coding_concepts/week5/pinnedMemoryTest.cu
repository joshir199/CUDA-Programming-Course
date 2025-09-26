#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;

#define N 999999

#define CHECK_CUDA(call) do {                       \
    cudaError_t e = (call);                         \
    if(e != cudaSuccess) {                           \
        cerr<<"CUDA Error: "<<cudaGetErrorString(e)     \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl;       \
        exit(1);                                    \
    }                                               \
} while(0)


// Analysis of time taken for transferring the pageable memory from host to Device
float cudaPageableMemoryTest(bool hostToDevice) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int *h_a;
    int *d_a;

    // allocate pageable host memory using malloc
    h_a =  (int*)malloc(N*sizeof(int));
    CHECK_CUDA(cudaMalloc(&d_a, N*sizeof(int)));


    for(int i = 0; i<N; i++){
        h_a[i] = i;
    }

    CHECK_CUDA(cudaEventRecord(start, 0));
    // pageable memory transfer to GPU using cudaMemcpy
    for(int i = 0 ;i<100;i++) {
        if(hostToDevice) {
            CHECK_CUDA(cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice));
        } else {
            CHECK_CUDA(cudaMemcpy(h_a, d_a, N*sizeof(int), cudaMemcpyDeviceToHost));
        }
    }

    // record the time taken for such operation
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    free(h_a);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return elapsed_time;
}


// Analysis of time taken for transferring the pinned(non-pageable) memory from host to Device
float cudaPinnedMemoryTest(bool hostToDevice) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int *h_a;
    int *d_a;

    // define and allocate the pinned host memory buffer using cudaHostAlloc
    CHECK_CUDA(cudaHostAlloc(&h_a, N*sizeof(int), cudaHostAllocDefault));
    CHECK_CUDA(cudaMalloc(&d_a, N*sizeof(int)));


    CHECK_CUDA(cudaEventRecord(start, 0));
    // pageable memory transfer to GPU
    for(int i = 0 ;i<100;i++) {
        if(hostToDevice) {
            CHECK_CUDA(cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice));
        } else {
            CHECK_CUDA(cudaMemcpy(h_a, d_a, N*sizeof(int), cudaMemcpyDeviceToHost));
        }
    }

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cudaFreeHost(h_a);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return elapsed_time;
}


int main() {

    float pageableUp = cudaPageableMemoryTest(true);
    float pageableDown = cudaPageableMemoryTest(false);
    
    cout<<"Pageable Memory Transfer Elapsed time(in ms) for HostToDevice: " \
    <<pageableUp<<", and for DeviceToHost: "<<pageableDown<<endl;

    float pinnedUp = cudaPinnedMemoryTest(true);
    float pinnedDown = cudaPinnedMemoryTest(false);
    
    cout<<"Pinned Memory Transfer Elapsed time(in ms) for HostToDevice: " \
    <<pinnedUp<<", and for DeviceToHost: "<<pinnedDown<<endl;
    
    
    cout<<"SpeedUp for pinned memory: "<<(pageableUp + pageableDown)/((pinnedUp + pinnedDown))<<endl;
    
    return 0;

}