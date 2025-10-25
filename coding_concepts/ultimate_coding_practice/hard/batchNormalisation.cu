#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
using namespace std;

#define n 9999  // Batch size <= 10K
#define cf 99   // feature size <= 1024

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

// get matrix transpose A[NxC]  -> Output[CxN]
__global__ void matrix_transpose_kernel(float* input, float* output, int rows, int cols) {

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x<cols && y<rows) {
        output[y + x*rows] = input[x + y *cols];
    }
}

// get Mean for each feature row (C) per block for matrix A[CxN]
__global__ void perFeatureMean(float* a, float* b, int N, int C) {

    extern __shared__ float cache[];

    // load each elements per C per row into per block
    // Per thread loads multiple elements cooperatively
    for(int i = threadIdx.x; i< N; i+=blockDim.x) {
        int gIdx = blockIdx.x * N + i;
        cache[i] = a[gIdx];
    }
    __syncthreads();

    // Per thread sum for each block
    int localSum = 0.0f;
    for(int i = threadIdx.x; i< N; i+=blockDim.x) {
        localSum += cache[i];
    }
    __syncthreads();

    __shared__ float blockCache[256];  // threads Per block = 256
    blockCache[threadIdx.x] = localSum;
    __syncthreads();

    // parallel sum of data points across feature
    for(int i = blockDim.x/2; i>0; i/=2) {
        if(threadIdx.x < i) {
            blockCache[threadIdx.x] += blockCache[threadIdx.x + i];
        }
        __syncthreads();
    }

    // get the Mean of each features using all elements
    if(threadIdx.x == 0) {
        b[blockIdx.x] = blockCache[0]/N;
    }
}

// get Variance for each feature row (C) per block for matrix A[CxN]
__global__ void perFeatureVariance(float* a, float* b, float* c, int N, int C) {

    extern __shared__ float cache[];

    // load each elements per C per row into per block
    // Per thread loads multiple elements cooperatively
    for(int i = threadIdx.x; i< N; i+=blockDim.x) {
        int gIdx = blockIdx.x * N + i;
        cache[i] = (a[gIdx] - b[blockIdx.x]) *  (a[gIdx] - b[blockIdx.x]);
    }
    __syncthreads();

    // Per thread sum for each block
    int localSum = 0.0f;
    for(int i = threadIdx.x; i< N; i+=blockDim.x) {
        localSum += cache[i];
    }
    __syncthreads();

    __shared__ float blockCache[256];  // threads Per block = 256
    blockCache[threadIdx.x] = localSum;
    __syncthreads();

    // parallel sum of data points across class
    for(int i = blockDim.x/2; i>0; i/=2) {
        if(threadIdx.x < i) {
            blockCache[threadIdx.x] += blockCache[threadIdx.x + i];
        }
        __syncthreads();
    }

    // get Variance of each feature using all elements
    if(threadIdx.x == 0) {
        c[blockIdx.x] = blockCache[0]/N;
    }
}

// get normalized output using feature mean and variance for matrix A[NxC]
// We will calculate per batch of elements per block using per feature mean & variance.
__global__ void normalizedFinalOutput(float* a, float* b, float* c, float* d, int N, int C, float* gamma, float* beta, float eps) {
    int offset = blockIdx.x * C;  // each block have C features
    // per thread represents features and per block represents elements
    if(threadIdx.x < C) {
        float standard_norm = (a[threadIdx.x + offset] - b[threadIdx.x]) * rsqrtf(c[threadIdx.x] + eps);
        d[threadIdx.x + offset] = gamma[threadIdx.x] * standard_norm + beta[threadIdx.x];
    }

}


int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int N = n;
    int C = cf; // number of features [0, C-1]

    float h_a[N*C], h_d[N*C], h_gm[C], h_beta[C];
    float *d_a, *d_at, *d_b, *d_c, *d_d, *d_gm, *d_beta;


    CHECK_CUDA(cudaMalloc(&d_a, N*C*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_at, C*N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, C*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, C*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_d, N*C*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gm, C*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_beta, C*sizeof(float)));

    for(int i = 0; i<N*C; i++) {
        h_a[i] = ((i + 2) % 99)*0.8f;
        //cout<<"h_a:"<<h_a[i]<<endl;
    }
    for(int i=0;i<C; i++) {
        h_gm[i] = ((2*i + 1)%9) * 0.8f;
        h_beta[i] = ((i+1)%5)*1.0f;
    }
    float eps = 0.00001f;

    CHECK_CUDA(cudaEventRecord(start, 0));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N*C*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_gm, h_gm, C*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_beta, h_beta, C*sizeof(float), cudaMemcpyHostToDevice));

    // Transpose the matrix to calculate per feature Mean & Variance
    dim3 block(16, 16);
    dim3 grid((C + 15)/16,(N + 15)/16);
    matrix_transpose_kernel<<<grid, block>>>(d_a, d_at, N, C);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // After transpose, we will use A_T for Mean & Variance calculation
    int blocksPerGrid = C; // each feature row as separate block (C<= 1024)
    int threadsPerBlock = 256;
    size_t shmem = N * sizeof(float);  // N elements per block
    perFeatureMean<<<blocksPerGrid, threadsPerBlock, shmem>>>(d_at, d_b, N, C);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Similar setup for Variance calculation
    perFeatureVariance<<<blocksPerGrid, threadsPerBlock, shmem>>>(d_at, d_b, d_c, N, C);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Now, We do the final normalised calculation for each element in original batch
    int threadsPerBlock1 = 128; // each threads for features >= C
    int blocksPerGrid1 = N; // each block for each element
    normalizedFinalOutput<<<blocksPerGrid1, threadsPerBlock1>>>(d_a, d_b, d_c, d_d, N, C, d_gm, d_beta, eps);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_d, d_d, N*C*sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"Elapsed time(in ms) is "<<elapsed_time<<endl;  // 1.87

    for(int i = 0; i<N*C && i<20; i++) {
        cout<<"Batch Normalised Value is : "<<h_d[i]<<", original element: "<<h_a[i]<<endl;
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaFree(d_d));
    CHECK_CUDA(cudaFree(d_gm));
    CHECK_CUDA(cudaFree(d_beta));

    return 0;

}
