#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;


#define n 99999
#define threadsPerBlock 1024

#define CHECK_CUDA(call) do {                    \
    cudaError_t e = (call);                      \
    if(e != cudaSuccess) {                        \
        cout<<"CUDA Error: "<<cudaGetErrorString(e) \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl; \
        exit(1);                                     \
    }                                               \
} while(0)



// General 1-bit radix sort upto 32-bit unsigned integer element
__global__ void oneBitRadixSort(unsigned int* a, unsigned int* b, unsigned int* c, unsigned int* d, unsigned int iter, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int key, bit;
    extern __shared__ unsigned int cache[];  // create shared memory of length (blockDim + 1) for inclusive scan per block
    unsigned int temp = 0;

    // create a bit map at each iter LSD level to each index
    if(tid<N) {
        key = a[tid]; // get the element
        bit = (key >> iter) & 1;  // get the "iter"th LSD bit in terms of 0 or 1
        temp = bit;  // store the bit value corresponding to index
        b[tid] = bit;
    }
    cache[threadIdx.x] = temp;
    __syncthreads();

    // Now, we will calculate the equivalent output index for each input element
    // based on their bits at "iter" LSD level
    // For any index which have "1" as bit,
    // its output index = Num of zero bit elements + Num of 1 bit elements before it
    //               = (Input size - # total One bit elements) + # One bit elements before it
    // Similarly, for index which have "0" as bit
    // its output index = Num of zero bit elements before it
    //                 = Input index  -  (# One bit element before it)


    // Using Exclusive Scan (Prefix Sum), we can get number of one bits before each index
    // Let's perform the inclusive scan first and later modify to make it exclusive
    // It will be per block inclusive scan
    for(int i = 1; i<blockDim.x; i*=2) {
        unsigned int t = 0;
        if(threadIdx.x >= i) {  // sum with previous valid indexes
            t = cache[threadIdx.x - i];
        }
        __syncthreads();

        cache[threadIdx.x] += t;
        __syncthreads();
    }


    if(tid<N) {
        c[tid] = cache[threadIdx.x];
    }
    int last = min(blockDim.x, N - blockIdx.x * blockDim.x) - 1;
    if(threadIdx.x == last) {
        d[blockIdx.x] = cache[last];
    }

}


__global__ void finalScatterPass(unsigned int* a, unsigned int* b, unsigned int* c, unsigned int* d, unsigned int* e, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int blockSum, bit;

    if(tid<N) {
        unsigned int totalOneBits = d[gridDim.x - 1]; // total One bit count from per block cumulative sum
        bit = b[tid];  // per element bit value for particular iteration
        blockSum = (blockIdx.x == 0) ? 0 : d[blockIdx.x - 1];  // get sum of previous blocks
        unsigned int onesBeforeIndex = c[tid] + blockSum - bit; // per block + cumulative sum before - current bit
        unsigned int outputIndex = (bit == 0) ? (tid - onesBeforeIndex) : (N - totalOneBits + onesBeforeIndex);
        e[outputIndex] = a[tid];
    }

}




int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int N = n;
    unsigned int h_a[N], h_c[N], h_b[N];
    unsigned int *d_a, *d_b, *d_c, *d_d, *d_e;

    CHECK_CUDA(cudaMalloc(&d_a, N*sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_b, N*sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_c, N*sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_e, N*sizeof(unsigned int)));

    // fill data in host device
    // Each data will be <= 10 bit  = h_a[i] < 1024;
    for(unsigned int i=0; i<N ;i++) {
        h_a[i] = (rand() + i + 1)%999 + 1;
    }

    CHECK_CUDA(cudaEventRecord(start, 0));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N*sizeof(unsigned int), cudaMemcpyHostToDevice));

    // local sort per block
    int blocksPerGrid = (N + threadsPerBlock - 1)/threadsPerBlock;
    size_t shmem = (threadsPerBlock + 1) * sizeof(unsigned int);
    CHECK_CUDA(cudaMalloc(&d_d, (blocksPerGrid)*sizeof(unsigned int)));

    unsigned int h_d[blocksPerGrid];

    unsigned int numBits = 10;   // Can be set to 32-bit unsigned integers
    for(unsigned int i = 0; i< numBits; i++) {
        //cudaMemset(d_d, 0, blocksPerGrid * sizeof(unsigned int));
        oneBitRadixSort<<<blocksPerGrid, threadsPerBlock, shmem>>>(d_a, d_b, d_c, d_d, i, N);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_d, d_d, blocksPerGrid*sizeof(unsigned int), cudaMemcpyDeviceToHost));
        // do the per block cumulative sum on CPU
        for(int j = 1; j<blocksPerGrid; j++) {
            h_d[j] += h_d[j-1];
            //cout<<"h_d: i:"<<i<<", "<<h_d[j]<<endl;
        }

        CHECK_CUDA(cudaMemcpy(d_d, h_d, blocksPerGrid*sizeof(unsigned int), cudaMemcpyHostToDevice));

        finalScatterPass<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, d_d, d_e, N);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        unsigned int* temp = d_e;
        d_e = d_a;
        d_a = temp;
    }

    CHECK_CUDA(cudaMemcpy(h_c, d_a, N*sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_b, d_b, N*sizeof(unsigned int), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"Elapsed time(in ms) : "<< elapsed_time<<endl;  // 0.70

    for(int i = 0; i< 50 && i<N; i++) {
        cout<<"General Radix sorting result at i:"<<i<<", is: "<<h_c[i]<<", original array: "<<h_a[i]<<endl;
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}