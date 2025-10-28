#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cfloat>
#include <cuda_runtime.h>
using namespace std;


#define n 999999
#define threadsPerBlock 256
#define Tile 256

#define CHECK_CUDA(call) do {                    \
    cudaError_t e = (call);                      \
    if(e != cudaSuccess) {                        \
        cout<<"CUDA Error: "<<cudaGetErrorString(e) \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl; \
        exit(1);                                     \
    }                                               \
} while(0)



__global__ void bitonicSortBlock(float* a, int N) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ float cache[];

    if(tid<N) {
        cache[threadIdx.x] = a[tid];
    } else {
        cache[threadIdx.x] = FLT_MAX;
    }
    __syncthreads();

    // define number of steps as power of 2,  2^n where n starts at 1 and till 2^n == 2 * blockDim.x
    // because, for 2^n == blockDim.x, it will create bitonic sequence rather than sorted block
    for(int i = 2; i<= 2*blockDim.x; i*=2) {
        // for each size i, half will be ascending and other half will be descending
        // ascending for thread which starts at even multiple of i like 0, 1*i, 2*i, ...
        bool ascending = ((threadIdx.x / (i/2)) % 2 == 0); // thread which starts at even multiple of 2

        // for each step, we have k number of stages to reorder the elements where 2^k == i.
        for(int stride = i/2; stride>0; stride /= 2) {

            // Get partner index for each threadIdx using XOR operation
            int ixj = threadIdx.x ^ stride;

            // check only for valid ordered pairs where the partner Index is ahead of threadIdx
            if(ixj<blockDim.x && ixj > threadIdx.x) {
                // if index is increasing part, but values are not, swap [true == true]
                // or if index is decreasing part, but values are not, swap [false == false]
                if(ascending == (cache[ixj] < cache[threadIdx.x])) {
                    float temp = cache[ixj];
                    cache[ixj] = cache[threadIdx.x];
                    cache[threadIdx.x] = temp;
                }
            }
            __syncthreads();
        }
    }

    if(tid < N) {
        a[tid] = cache[threadIdx.x];
    }
}


// Binary search to get the lowest index for A & B which combines to get index in C
__device__ int get_Co_rank(float* a, float* b, int M, int N, int k) {
    int s = max(0, k-N); // low index for A
    int e = min(M, k);   // high index for A

    int mid=0;
    int j;  // mid for index in A and j for index in B
    while(s<e) {
        mid = (s+e)/2;
        j = k-mid;

        if(j>0 && a[mid] >= b[j-1]) {
            e = mid; // get the lowest index where this condition agrees
        } else {
            s = mid + 1; // go for more forward index
        }
    }

    return s; // lowest index in A which satisfy co-rank
}


// Binary search to get the lowest index for A & B which combines to get index in C
__device__ int get_Co_rank_general(float* a, float* b, int M, int N, int k) {
    int s = max(0, k-N); // low index for A
    int e = min(M, k);   // high index for A

    int mid=0;
    int j;  // mid for index in A and j for index in B
    while(s<e) {
        mid = (s+e)/2;
        j = k-mid;

        if(j>0 && mid<M && a[mid] < b[j-1]) {
            s = mid + 1; // got very elements in A, get more higher values
        } else if(mid>0 && j<N && a[mid-1] > b[j]) {
            e = mid - 1; // got too many higher A values, go back
        } else {
            return mid;
        }
    }

    return s; // lowest index in A which satisfy co-rank
}

__device__ void mergeSequential(float* a, int M, float* b, int N, float* c) {
    int i =0;
    int j =0;
    int k =0;

    while(i<M && j<N) {
        if(a[i] <= b[j]) {  // maintain stable order in case of equal condition
            c[k] = a[i];
            i++;
            k++;
        } else {
            c[k] = b[j];
            j++;
            k++;
        }
    }

    // if elements in A remains, fill it into c
    while(i<M){
        c[k] = a[i];
        i++;
        k++;
    }
    // if elements in B remains, fill it into c
    while(j<N){
        c[k] = b[j];
        j++;
        k++;
    }
}

__global__ void parallelMerge(float* a, float* c, int M, int N) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float cacheA[Tile];  // each tile will process Tile elements at a time to produce Tile output
    __shared__ float cacheB[Tile];
    int elementsPerBLock = Tile;

    float* b = &a[M];

    //Get current and next output index per block
    int k_cur = elementsPerBLock * blockIdx.x;
    int k_next = min((blockIdx.x + 1)*elementsPerBLock , M+N);

    if(k_cur >= M+N) { return;} // check on index limit

    // Now, we will calculate the co-rank index from A
    int i_cur = get_Co_rank_general(a, b, M, N, k_cur);
    int i_next = get_Co_rank_general(a, b, M, N, k_next);

    // get similar index for B
    int j_cur = k_cur - i_cur;
    int j_next = k_next - i_next;

    // length of segment in A and B per block
    int sizeA = i_next - i_cur;
    int sizeB = j_next - j_cur;

    // Load elements cooperatively into the shared memory for processing per block
    for(int i = threadIdx.x; i<sizeA; i+= blockDim.x) {
        cacheA[i] = a[i_cur + i];
    }
    for(int i = threadIdx.x; i<sizeB; i+= blockDim.x) {
        cacheB[i] = b[j_cur + i];
    }
    __syncthreads();

    // Now, we can merge the elements using single thread for elements in shared memory.
    if(threadIdx.x == 0) {
        mergeSequential(cacheA, sizeA, cacheB, sizeB, &c[k_cur]);
    }
    __syncthreads();

    if(tid<M+N) {
        a[tid] = c[tid];
    }
}


int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int N = n;
    float h_a[N], h_c[N];
    float *d_a, *d_c;

    CHECK_CUDA(cudaMalloc(&d_a, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, N*sizeof(float)));

    // fill data in host device
    for(int i=0; i<N ;i++) {
        h_a[i] = ((i + 9 + rand()) % 999)*0.08f;
    }

    CHECK_CUDA(cudaEventRecord(start, 0));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice));

    // local sort per block
    int blocksPerGrid = (N + threadsPerBlock - 1)/threadsPerBlock;
    size_t shmem = threadsPerBlock * sizeof(float);
    bitonicSortBlock<<<blocksPerGrid, threadsPerBlock, shmem>>>(d_a, N);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(d_c, d_a, N*sizeof(float), cudaMemcpyDeviceToDevice));
    //CHECK_CUDA(cudaMemcpy(h_c, d_a, N*sizeof(float), cudaMemcpyDeviceToHost));

    for(int i = 1; i< blocksPerGrid; i++) {
        if(i == blocksPerGrid -1){
            parallelMerge<<<i+1, threadsPerBlock>>>(d_a, d_c, i*threadsPerBlock, N - i*threadsPerBlock);
        } else {
            parallelMerge<<<i+1, threadsPerBlock>>>(d_a, d_c, i*threadsPerBlock, threadsPerBlock);
        }
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    CHECK_CUDA(cudaMemcpy(h_c, d_c, N*sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"Elapsed time(in ms) : "<< elapsed_time<<endl;  // 301.2

    for(int i = 0; i< 50 && i<N; i++) {
        cout<<"Odd-Even sorting result at i:"<<i<<", is: "<<h_c[i]<<", original array: "<<h_a[i]<<endl;
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}