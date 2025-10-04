#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;


#define N 999999
#define threadsPerBlock 256

#define CHECK_CUDA(call) do {                    \
    cudaError_t e = (call);                      \
    if(e != cudaSuccess) {                        \
        cout<<"CUDA Error: "<<cudaGetErrorString(e) \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl; \
        exit(1);                                     \
    }                                               \
} while(0)


// Counts the number of elements with the integer value k in an array of 32-bit integers
__global__ void elementCounter(int* a, int* c, int k) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // So instead of all threads atomically writing to the same global counter (which would
    // cause huge contention for large N), we first combine results within each block.
    __shared__ int cache; // visible to threads per block only
    // If every thread did this, we’d have a race condition.
    // Only thread is enough to initialize it.
    if(threadIdx.x ==0) { cache = 0;}
    __syncthreads();

    //Each thread keeps a local counter in registers.
    int count = 0; // define per thread count value

    if(tid<N && a[tid] == k) {
        count = 1; // per thread count of occurrence of element k.
    }

    // Now each thread adds its local result (0 or 1) into the block’s shared counter.
    // But cache will be visible to threads of its own block, so only 256 threads will update value
    atomicAdd(&cache, count); // let all thread update the count values to shared memory
    __syncthreads();  // synch to update total matches inside this block

    // Now, sum the accumulated counts per block
    if(threadIdx.x == 0) {
        atomicAdd(c, cache);
    }
}



int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));

    int h_a[N], h_c, k;
    int *d_a, *d_c;

    CHECK_CUDA(cudaMalloc(&d_a, N*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_c, sizeof(int)));

    // fill data in host device
    for(int i=0; i<N ;i++) {
        h_a[i] = (i*i + 3*i - 46)%13;  //(rand() % 90) * 0.018f ;
    }
    k = 8;

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice));

    int blocksPerGrid = (N + threadsPerBlock - 1)/threadsPerBlock;

    elementCounter<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_c, k);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"Elapsed time(in ms) : "<< elapsed_time<<endl;


    cout<<"Count of element k="<<k<<", is: "<<h_c<<endl;


    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}