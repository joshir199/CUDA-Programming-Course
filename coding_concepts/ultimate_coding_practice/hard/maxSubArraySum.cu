#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;

#define n 999999
#define threadsPerBlock 256

#define max(a,b) (a>b)?a:b

#define CHECK_CUDA(call) do {                    \
    cudaError_t e = (call);                      \
    if(e != cudaSuccess) {                        \
        cout<<"CUDA Error: "<<cudaGetErrorString(e) \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl; \
        exit(1);                                     \
    }                                               \
} while(0)


// Get the per window (=block) sum
__global__ void windowBlockSum(int* a, int* b, int N, int w) {

    extern __shared__ int cache[];
    // load in shared memory in cooperative manner if w>threadsPerBlock
    for(int i = threadIdx.x ; i<w; i+= blockDim.x) {
        int gIdx = blockIdx.x + i; // each block starts with new window elements
        if(gIdx < N) {
            cache[i] = a[gIdx];
        } else {
            cache[i] = 0;
        }
    }
    __syncthreads();

    // calculate Intra-thread local sum to have per thread value
    int localSum = 0;
    for(int i = threadIdx.x ; i<w; i+= blockDim.x) {
        localSum += cache[i];
    }
    __syncthreads();

    __shared__ int blockCache[256]; // threads in a block
    blockCache[threadIdx.x] = localSum;
    __syncthreads();

    //Parallel reduction for each window sum
    for(int i = blockDim.x/2; i>0; i/=2) {
        if(threadIdx.x < i) {
            blockCache[threadIdx.x] += blockCache[threadIdx.x + i];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0) {
        b[blockIdx.x] = blockCache[0];
    }
}

// Get Maximum values per block sum to final output
__global__ void getFinalMaxValue(int* a, int* c, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ int cache[];

    if(tid<N) {
        cache[threadIdx.x] = a[tid];
    } else {
        cache[threadIdx.x] = INT_MIN;
    }
    __syncthreads();

    // get per block max value
    int i = blockDim.x /2;
    while(i>0) {
        if(threadIdx.x < i) {
            cache[threadIdx.x] = max(cache[threadIdx.x], cache[threadIdx.x + i]);
        }
        __syncthreads();
        i = i/2;
    }

    // get final max value using atomic operation
    if(threadIdx.x == 0) {
        atomicMax(c, cache[0]);
    }
}

// The maximum sum of any contiguous subarray of length exactly window_size w.

int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));


    int N=n;
    int h_a[N], h_c, w;
    int *d_a, *d_b, *d_c;
    int minValue = INT_MIN;

    CHECK_CUDA(cudaMalloc(&d_a, N*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_c, sizeof(int)));
    // Prefer to use cudaMemcpy for copying MAX or MIN values
    CHECK_CUDA(cudaMemcpy(d_c, &minValue, sizeof(int), cudaMemcpyHostToDevice));

    // fill data in host device
    for(int i=0; i<N ;i++) {
        h_a[i] = (i*i + 3*i + 6)%13;
    }
    w = 289;

    CHECK_CUDA(cudaEventRecord(start, 0));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice));

    int blocksPerGrid = N - w + 1; // each block will have single window of elements
    CHECK_CUDA(cudaMalloc(&d_b, blocksPerGrid*sizeof(int)));
    size_t shmem = w*sizeof(int); // size of window
    // calculate per block sum for given window (slided across the array length)
    windowBlockSum<<<blocksPerGrid, threadsPerBlock, shmem>>>(d_a, d_b, N, w);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());


    // Accumulate the Maximum value from all the blocks
    int blocksPerGrid1 = (blocksPerGrid + threadsPerBlock - 1)/ threadsPerBlock;
    size_t shmem2 = threadsPerBlock*sizeof(int);
    getFinalMaxValue<<<blocksPerGrid1, threadsPerBlock, shmem2>>>(d_b, d_c, blocksPerGrid);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"Elapsed time(in ms) : "<< elapsed_time<<endl;  // 9.29


    for(int i= 0; i<N && i<20; i++) {
        cout<<"Input array element: "<<h_a[i]<<endl;
    }

    cout<<"Max Sub-array sum in window w: "<<w<<", is "<<h_c<<endl;  // 2317

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}