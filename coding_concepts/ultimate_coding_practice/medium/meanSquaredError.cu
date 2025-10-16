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


__device__ float atomicAddFloat(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i;
    int assumed;
    do {
        assumed = old;
        // convert assumed bits to float, compare with val, choose greater
        float assumed_f = __int_as_float(assumed);
        float new_f = val + assumed_f;
        int new_i = __float_as_int(new_f);
        // try to CAS swap
        old = atomicCAS(address_as_i, assumed, new_i);
    } while (assumed != old);
    return __int_as_float(old);
}


// Calculate Mean Squared Error using predicted output and true output.
__global__ void mseKernel(float* a, float* b, float* c) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ float cache[];

    if(tid<N) {
        cache[threadIdx.x] = (a[tid] - b[tid]) * (a[tid] - b[tid]) / N;
    } else {
        cache[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // calculate per block sum
    int i = blockDim.x/2;
    while(i>0) {
        if(threadIdx.x < i) {
            cache[threadIdx.x] += cache[threadIdx.x + i];
        }
        __syncthreads();
        i=i/2;
    }
    // save per block sum
    if(threadIdx.x == 0) {
        c[blockIdx.x] = cache[0];
    }
}

// calculate the final sum using parallel reduction of per block sum
// and accumulate it using atomic operations
__global__ void reductionKernel(float* a, float* c, int totalElement) {

    extern __shared__ float cache[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid < totalElement) {
        cache[threadIdx.x] = a[tid];
    } else {
        cache[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    int i = blockDim.x/2;
    while(i>0) {
        if(threadIdx.x < i) {
            cache[threadIdx.x] += cache[threadIdx.x + i];
        }
        __syncthreads();
        i=i/2;
    }

    if(threadIdx.x == 0) {
        atomicAddFloat(c, cache[0]);
    }
}


int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    int blockPerGrid1 = (N + threadsPerBlock - 1)/ threadsPerBlock;

    float h_a[N], h_b[N], h_c;
    float *d_a, *d_b, *d_c, *d_d;

    CHECK_CUDA(cudaMalloc(&d_a, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_d, blockPerGrid1*sizeof(float)));

    // fill data in host device
    for(int i=0; i<N ;i++) {
        h_a[i] = (rand() % 90) * 0.018f - (rand() % 90) * 0.05f;
        h_b[i] = (rand() % 99) * 0.08f;
    }

    CHECK_CUDA(cudaEventRecord(start, 0));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, N*sizeof(float), cudaMemcpyHostToDevice));

    size_t shmem = (threadsPerBlock) * sizeof(float);
    mseKernel<<<blockPerGrid1, threadsPerBlock, shmem>>>(d_a, d_b, d_d);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());


    int blockPerGrid2 = (blockPerGrid1 + threadsPerBlock -1)/threadsPerBlock;
    size_t shmem2 = threadsPerBlock * sizeof(float);
    reductionKernel<<<blockPerGrid2, threadsPerBlock, shmem2>>>(d_d, d_c, blockPerGrid1);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(&h_c, d_c, sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"Elapsed time(in ms) : "<< elapsed_time<<endl;  // 0.71

    cout<<"Mean Squared Error Result :"<<h_c<<", and blocksPerGrid1 is: "<<blockPerGrid1<<endl;


    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaFree(d_d));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}