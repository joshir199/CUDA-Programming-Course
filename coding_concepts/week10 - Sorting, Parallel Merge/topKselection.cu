#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cfloat>
#include <cuda_runtime.h>
using namespace std;


#define n 999999
#define k 29
#define threadsPerBlock 256

#define CHECK_CUDA(call) do {                    \
    cudaError_t e = (call);                      \
    if(e != cudaSuccess) {                        \
        cout<<"CUDA Error: "<<cudaGetErrorString(e) \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl; \
        exit(1);                                     \
    }                                               \
} while(0)


__global__ void bitonicTopKSortBlock(const float* a, float* c, int N, int K) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    extern __shared__ float cache[];

    if(tid < N) {
        cache[threadIdx.x] = a[tid];
    } else {
        cache[threadIdx.x] = -FLT_MAX;
    }
    __syncthreads();

    // "i<=blockDim.x", here equality sign is very important
    for(int i = 2; i<=blockDim.x; i *= 2) {
        bool descending = ((threadIdx.x / i)%2 == 0);

        for(int stride = i/2; stride>0; stride /= 2) {
            int ixj = threadIdx.x ^ stride;

            if(ixj < blockDim.x && ixj > threadIdx.x) {
                // check for decreasing order mismatch
                if((cache[ixj] > cache[threadIdx.x]) == descending) {
                    float temp = cache[ixj];
                    cache[ixj] = cache[threadIdx.x];
                    cache[threadIdx.x] = temp;
                }
            }
            __syncthreads();
        }
    }

    // copy top K elements from each block which are sorted in decreasing order
    if(threadIdx.x < K && tid<N) {
        c[threadIdx.x + K * blockIdx.x] = cache[threadIdx.x];
    }
}



int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int N = n;
    int K = k;
    float h_a[N], h_c[K];
    float *d_a, *d_c;

    CHECK_CUDA(cudaMalloc(&d_a, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, N*sizeof(float)));


    // fill data in host device
    for(int i=0; i<N ;i++) {
        h_a[i] = i * 1.0f; //((i + 9 + rand()) % 999)*0.08f;
        //cout<<"h_a[i]: "<<h_a[i]<<endl;
    }

    CHECK_CUDA(cudaEventRecord(start, 0));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice));

    // local sort per block
    int M = N;
    int curBlocks = (M + threadsPerBlock - 1)/threadsPerBlock;

    // kep sorting until only K or less elements are left
    while(M > K) {
        size_t shmem = threadsPerBlock * sizeof(float);
        bitonicTopKSortBlock<<<curBlocks, threadsPerBlock, shmem>>>(d_a, d_c, M, K);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        M = K * curBlocks;
        curBlocks = (M + threadsPerBlock - 1)/threadsPerBlock;

        // swap buffers
        swap(d_a, d_c);
    }

    // now, sort those K elements and get the output.
    size_t shmem = threadsPerBlock * sizeof(float);
    bitonicTopKSortBlock<<<1, threadsPerBlock, shmem>>>(d_a, d_c, M, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c, d_c, K*sizeof(float), cudaMemcpyDeviceToHost));


    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"Elapsed time(in ms) : "<< elapsed_time<<endl;  // 0.56

    for(int i = 0; i< 50 && i<K; i++) {
        cout<<"Top K result at i:"<<i<<", is: "<<h_c[i]<<", original array: "<<h_a[i]<<endl;
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}