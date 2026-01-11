#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cfloat>
#include <cuda_runtime.h>
#include <curand_kernel.h>
using namespace std;


#define n 50000
#define threadsPerBlock 256
#define threadCoarse 1024

#define CHECK_CUDA(call) do {                    \
    cudaError_t e = (call);                      \
    if(e != cudaSuccess) {                        \
        cout<<"CUDA Error: "<<cudaGetErrorString(e) \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl; \
        exit(1);                                     \
    }                                               \
} while(0)


__global__ void getSamplingProbability(int seed, float* p) {
    if(threadIdx.x == 0) {  // single thread operation
        curandState state;
        curand_init(seed, 0, 0, &state);

        float r = curand_uniform(&state);     // r in (0, 1]
        if(r == 1.0f) {
            r = 0.999999f;
        }
        *p = r;
    }
}

__global__ void softmaxFunction(float* a, float maxVal, float* totalSum, int N) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float cache[256];

    float temp = 0.0f; // per thread value storage

    // load the exponential of each term with handled overflow
    // condition using maximum of each element
    if(tid<N) {
        temp = expf(a[tid] - maxVal);
    }
    cache[threadIdx.x] = temp;
    __syncthreads();

    if(tid<N) {
        a[tid] = temp;
    }

    int i = threadsPerBlock/2;
    while(i>0) {
        if(threadIdx.x < i) {
            cache[threadIdx.x] += cache[threadIdx.x + i];
        }
        __syncthreads();
        i = i/2;
    }

    // get the total sum of each exponential term from each block
    // into single variable (variable should be global rather than local
    // thread because it will reset to zero every iterations)
    if(threadIdx.x == 0) {
        atomicAdd(totalSum, cache[0]); //use Max function for float values
    }
}


// descending order
// Num of threads per block = 256 (power of 2 only)
// Keep track of indexes while sorting.
// Sorting must preserve the mapping from sorted values ? original indices.
// This is called key–value sorting.
__global__ void bitonicSort(float* data, int* indexArr, int N) {

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    __shared__ float cache[256];
    __shared__ int cacheInd[256];
    if(tid<N) {
        cache[threadIdx.x] = data[tid];
        cacheInd[threadIdx.x] = indexArr[tid];
    } else {
        cache[threadIdx.x] = -FLT_MAX;
        cacheInd[threadIdx.x] = -1;
    }
    __syncthreads();

    // for bitonic sort, we start with size of 2 to compare the elements within with 1st half to 2nd half
    for(int i = 2; i<=blockDim.x; i*= 2){
        // this gives the order for block of size i. It should be bitonic
        // [0, i-1] -> dec, [i, 2i -1] -> inc
        bool descending = ((threadIdx.x / i)%2 == 0);   // (threadIdx.x / i) => starting point of each block

        for(int stride = i/2; stride>0; stride/=2) {
            int ixj = threadIdx.x ^ stride; // this will give the complement ids per threadId based on stride length
            // if threadIdx.x = 1 and stride = 2  (within the block of 2^stride, pairs  of index with gap of stride)
            // ixj = 3, similarly for threadIdx.x = 3, ixj = 1


            // check for boundary limit and only one way comparison
            if(ixj<blockDim.x && ixj > threadIdx.x) {
                if((cache[ixj] > cache[threadIdx.x]) == descending) {
                    float temp = cache[ixj];
                    cache[ixj] = cache[threadIdx.x];
                    cache[threadIdx.x] = temp;

                    int tmp = cacheInd[ixj];
                    cacheInd[ixj] = cacheInd[threadIdx.x];
                    cacheInd[threadIdx.x] = tmp;
                }
            }
            __syncthreads();
        }
    }

    if(tid<N) {
        data[tid] = cache[threadIdx.x];
        indexArr[tid] = cacheInd[threadIdx.x];
    }
}

// Sequential merge kernel for decreasing order for both array and its indexes (preserving key-value)
__global__ void mergeKernel_singleThread(const float* A, const int* A_ind, int M, const float* B, const int* B_ind, int N, float* C, int* C_ind) {
    // only one thread does work
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int i = 0, j = 0, k = 0;
        while (i < M && j < N) {
            if (A[i] >= B[j]) {
                C[k] = A[i];
                C_ind[k] = A_ind[i];
                k++;
                i++;
            } else {
                C[k] = B[j];
                C_ind[k] = B_ind[j];
                k++;
                j++;
            }
        }
        while (i < M) {
            C[k] = A[i];
            C_ind[k] = A_ind[i];
            k++;
            i++;
        }
        while (j < N) {
            C[k] = B[j];
            C_ind[k] = B_ind[j];
            k++;
            j++;
        }
    }
}


// Bitonic sort algorithm follows: Hierarchical Odd–Even Sort = Local Sort + Inter-Block Merge.
// It can be easily used for Multi-GPU extensions with distributed systems.
// Time complexity : O(N*(logN)^2) and worst case : O(N^2)

int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int N = n;
    float p = 0.95f;
    float h_a[N], h_c[N], totalSum;
    float *d_a, *d_c, *ts;
    int h_aInd[N], h_cInd[N], seed;
    int *d_aInd, *d_cInd;

    CHECK_CUDA(cudaMalloc(&d_a, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_aInd, N*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_c, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_cInd, N*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&ts, sizeof(float)));

    // fill data in host device
    for(int i=0; i<N ;i++) {
        h_a[i] = (rand() % 939) * 0.01f;
        h_aInd[i] = i;
    }
    totalSum = 0.0f;
    seed = 33;

    cout<<totalSum<<endl;

    CHECK_CUDA(cudaEventRecord(start, 0));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_aInd, h_aInd, N*sizeof(int), cudaMemcpyHostToDevice));

    // local sort per block
    int blocksPerGrid = (N + threadsPerBlock - 1)/threadsPerBlock;
    bitonicSort<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_aInd, N);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(d_c, d_a, N*sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(d_cInd, d_aInd, N*sizeof(int), cudaMemcpyDeviceToDevice));


    // Step 3: progressive merging
    // Start with segment width = threads (size that was sorted per block)
    int width = threadsPerBlock;
    while (width < N) {
        // For each pair of segments: [start, start+width) and [start+width, start+2*width)
        for (int start = 0; start < N; start += 2 * width) {
            int sizeA = min(width, max(0, N - start));
            int sizeB = min(width, max(0, N - (start + sizeA)));
            if (sizeB <= 0) continue; // nothing to merge

            float* A_ptr = d_c + start;
            float* B_ptr = d_c + start + sizeA;
            float* C_ptr = d_a + start; // write merged segment into tmp buffer at same offset
            // for Index
            int* A_ind_ptr = d_cInd + start;
            int* B_ind_ptr = d_cInd + start + sizeA;
            int* C_ind_ptr = d_aInd + start;

            // Launch simple merge kernel for this pair (single-thread)
            mergeKernel_singleThread<<<1, 1>>>(A_ptr, A_ind_ptr, sizeA, B_ptr, B_ind_ptr, sizeB, C_ptr, C_ind_ptr);
            CHECK_CUDA(cudaDeviceSynchronize());

            // Copy merged output back to 'data' so next merges read latest values
            CHECK_CUDA(cudaMemcpy(d_c + start, d_a + start, (sizeA + sizeB) * sizeof(float), cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpy(d_cInd + start, d_aInd + start, (sizeA + sizeB) * sizeof(int), cudaMemcpyDeviceToDevice));
        }
        width *= 2;
    }



    CHECK_CUDA(cudaMemcpy(h_c, d_c, N*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_cInd, d_cInd, N*sizeof(int), cudaMemcpyDeviceToHost));

    for(int i = 0; i< 30 && i<N; i++) {
        cout<<" First Bitonic sorting result at i:"<<i<<", is: "<<h_c[i]<<endl;
    }

    float maxLogitValue = h_c[0];
    CHECK_CUDA(cudaMemset(ts, 0, 1*sizeof(float)));
    softmaxFunction<<<blocksPerGrid, threadsPerBlock>>>(d_c, maxLogitValue, ts, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c, d_c, N*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&totalSum, ts, sizeof(float), cudaMemcpyDeviceToHost));

    cout<<" sum value: "<<totalSum<<endl;

    // get the largest index whose sum is greater than p
    int index = 0;
    float nucleusSum = 0.0f;
    while(index<N) {
        h_c[index] /= totalSum;    // convert it into probabilities
        float prob = h_c[index];
        if(p - prob <0) {
            break;
        } else {
            p -= prob;
            nucleusSum += prob;
            index++;
        }
    }

    cout<<" nucleus set length: "<<index<<endl;
    // after getting sorted (descending order)
    // Convert filtered set of size index to probabilities (no need for softmax now)
    // Get the new random nucleus probability (p_new) using the seed value
    getSamplingProbability<<<1, 1>>>(seed, ts);
    CHECK_CUDA(cudaDeviceSynchronize());

    float p_new;
    CHECK_CUDA(cudaMemcpy(&p_new, ts, sizeof(float), cudaMemcpyDeviceToHost));

    cout<<"p_new: "<<p_new<<endl;
    // get the largest index whose sum is greater than p_new
    int id = 0;
    while(id<index) {
        float prob = h_c[id]/nucleusSum;
        if(p_new - prob <0) {
            break;
        } else {
            p_new -= prob;
            id++;
        }
    }
    cout<<"id: "<<id - 1<<endl;

    int selectedId = id > 0 ? h_cInd[id - 1] : 0;

    cout<<"Selected sample Id: "<<selectedId<<endl;

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"Elapsed time(in ms) : "<< elapsed_time<<endl;  // 65.03

    for(int i = 0; i< 30 && i<N; i++) {
        cout<<"Bitonic sorting result at i:"<<i<<", is: "<<h_c[i]<<", original index: "<<h_cInd[i]<<", original array: "<<h_a[i]<<endl;
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaFree(ts));
    CHECK_CUDA(cudaFree(d_aInd));
    CHECK_CUDA(cudaFree(d_cInd));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}