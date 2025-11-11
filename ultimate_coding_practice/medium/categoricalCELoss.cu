#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
using namespace std;

#define n 9999  // number of data points <= 10,000
#define cl 99  // number of class labels <= 1000

#define threadsPerBlock 1024

#define CHECK_CUDA(call) do {                       \
    cudaError_t e = (call);                             \
    if(e != cudaSuccess) {                            \
        cerr<<"CUDA Error"<<cudaGetErrorString(e)       \
        <<"in "<<__FILE__<<", at "<<__LINE__<<endl;     \
        exit(1);                                    \
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

// get cross entropy loss for each data point per block
__global__ void perRowEntropyLoss(float* a, int* b, float* c, int N, int C) {

    extern __shared__ float cache[];

    int gIdx = blockIdx.x * C + threadIdx.x;
    float temp = 0.0f;
    // load exponential of each elements per row into per block
    if(threadIdx.x < C) {
        temp = expf(a[gIdx]);
    }
    cache[threadIdx.x] = temp;
    __syncthreads();

    // parallel sum of data points across class
    for(int i = blockDim.x/2; i>0; i/=2) {
        if(threadIdx.x < i) {
            cache[threadIdx.x] += cache[threadIdx.x + i];
        }
        __syncthreads();
    }

    // get log of total Cross entropy loss per datapoint - true label
    if(threadIdx.x == 0) {
        unsigned int p = blockIdx.x * C + b[blockIdx.x];  // true label for that block
        c[blockIdx.x] = logf(cache[0]) - a[p];
    }

}

// Accumulate the loss over all data points
__global__ void accumulateFinalLoss(float* a, float* c, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    extern __shared__ float cache[];

    if(tid < N) {
        cache[threadIdx.x] = a[tid];
    } else {
        cache[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // parallel reduction for block sum
    for(int i = blockDim.x/2; i>0; i/=2) {
        if(threadIdx.x < i) {
            cache[threadIdx.x] += cache[threadIdx.x + i];
        }
        __syncthreads();
    }

    // get final sum using atomics
    if(threadIdx.x == 0) {
        atomicAddFloat(c, cache[0]);
    }
}


int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int N = n;
    int C = cl; // number of classes [0, C-1]

    float h_a[N*C], h_c;
    int h_l[N];  // true label
    float *d_a, *d_b, *d_c;
    int *d_l;  // true label

    CHECK_CUDA(cudaMalloc(&d_a, N*C*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_l, N*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_c, sizeof(float)));
    CHECK_CUDA(cudaMemset(d_c, 0, sizeof(float)));

    for(int i = 0; i<N*C; i++) {
        h_a[i] = ((i + 2) % 99)*0.008f;
        //cout<<"h_a:"<<h_a[i]<<endl;
    }
    for(int i = 0; i<N; i++) {
        h_l[i] = ((i +2) % C);
        //cout<<"h_l:"<<h_l[i]<<endl;
    }

    CHECK_CUDA(cudaEventRecord(start, 0));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N*C*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_l, h_l, N*sizeof(int), cudaMemcpyHostToDevice));

    int blocksPerGrid = N; // each datapoint or row as separate block (N<= 10K)
    size_t shmem = threadsPerBlock * sizeof(float);
    perRowEntropyLoss<<<blocksPerGrid, threadsPerBlock, shmem>>>(d_a, d_l, d_b, N, C);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    //float h_b[N];
    //CHECK_CUDA(cudaMemcpy(h_b, d_b, N*sizeof(float), cudaMemcpyDeviceToHost));

    int blocksPerGrid1 = (N + threadsPerBlock - 1) / threadsPerBlock;
    accumulateFinalLoss<<<blocksPerGrid1, threadsPerBlock, shmem>>>(d_b, d_c, N);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(&h_c, d_c, sizeof(float), cudaMemcpyDeviceToHost));
    h_c = h_c/N;  // final average value

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"Elapsed time(in ms) is "<<elapsed_time<<endl;  // 0.95

    cout<<"Total Categorical Cross Entropy Loss : "<<h_c<<endl;  // 4.31

    //for(int i = 0; i<N && i<20; i++) {
    //    cout<<"h_b value: "<<h_b[i]<<endl;
    //}

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaFree(d_l));

    return 0;

}