#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;


#define m 899999
#define n 100000
#define threadsPerBlock 256
#define Tile 256  // keep it same as threadsPerblock for minimal storage of elements per block

#define min(a,b) ((a) < (b) ? a : b)
#define max(a,b) ((a) > (b) ? a : b)

#define CHECK_CUDA(call) do {                    \
    cudaError_t e = (call);                      \
    if(e != cudaSuccess) {                        \
        cout<<"CUDA Error: "<<cudaGetErrorString(e) \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl; \
        exit(1);                                     \
    }                                               \
} while(0)


// Binary search to get the lowest index for A & B which combines to get index in C
__device__ int get_Co_rank(int* a, int* b, int M, int N, int k) {
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

__device__ void mergeSequential(int* a, int M, int* b, int N, int* c) {
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


// Merge the index segments from A and B into C in parallel
// Use shared memory tiles to coalesce global reads (each tile loads a chunk from A and B).
// Global writes are coalesced :
//   - each block reads continuous subarrays of A and B
//   - each thread writes a contiguous portion of C).
// While effective, the drawbacks include:
// Each tile load → ~2× global reads per output tile. (2*(M + N))
// Shared memory mostly half-used.
__global__ void tiledParallelMerge(int* a, int* b, int* c, int M, int N) {

    // each block will calculate elementsPerBLock of the output
    // by storing into the shared memory
    // If Number of Block is given manually and Tile is different than threadsPerBlock
    int elementsPerBLock = (M + N + gridDim.x - 1)/gridDim.x;

    // Make sure, elementsPerBLock <= Tile, otherwise it won't fit in cache
    __shared__ int cacheA[Tile]; // shared memory for array A
    __shared__ int cacheB[Tile]; // shared memory for array B

    //Get current and next output index per block
    int k_cur = elementsPerBLock * blockIdx.x;
    int k_next = min((blockIdx.x + 1)*elementsPerBLock , M+N);

    if(k_cur >= M+N) { return;} // check on index limit

    // Now, we will calculate the co-rank index from A
    int i_cur = get_Co_rank(a, b, M, N, k_cur);
    int i_next = get_Co_rank(a, b, M, N, k_next);

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
}


int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int M = m;
    int N = n;

    int h_a[M], h_b[N], h_c[M+N];
    int *d_a, *d_b, *d_c;

    CHECK_CUDA(cudaMalloc(&d_a, M*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_b, N*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_c, (M+N)*sizeof(int)));

    // fill data in host device
    for(int i=0; i<M ;i++) {
        h_a[i] = (i + 9 + rand())%7 + 7*i;
        //cout<<"h_a: "<<h_a[i]<<endl;
    }
    for(int i=0; i<N ;i++) {
        h_b[i] = (i + 9 + rand())%6 + 7*i;
        //cout<<"h_b: "<<h_b[i]<<endl;
    }

    CHECK_CUDA(cudaEventRecord(start, 0));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, M*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, N*sizeof(int), cudaMemcpyHostToDevice));

    int blocksPerGrid = (M+N + threadsPerBlock - 1)/threadsPerBlock; // can use different block size

    tiledParallelMerge<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, M, N);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c, d_c, (M+N)*sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"Elapsed time(in ms) : "<< elapsed_time<<endl;  // 1.92


    for(int i = 0; i< 50 && i<(M+N); i++) {
        cout<<"Final Merged array result at i:"<<i<<", is: "<<h_c[i]<<endl;
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}