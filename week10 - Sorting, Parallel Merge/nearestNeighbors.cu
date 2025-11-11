#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cfloat>
#include <cuda_runtime.h>
using namespace std;


#define n 333333
#define threadsPerBlock 256

#define CHECK_CUDA(call) do {                    \
    cudaError_t e = (call);                      \
    if(e != cudaSuccess) {                        \
        cout<<"CUDA Error: "<<cudaGetErrorString(e) \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl; \
        exit(1);                                     \
    }                                               \
} while(0)


__global__ void warpBasedNearestNeighbor(float* a, int* c, int N) {

    int localIdx = blockIdx.x;
    int WarpSize = 32; // by default, warp consist of 32 threads
    int laneId = threadIdx.x % WarpSize;  // lane index within warp (32 threads work together)

    int localMinIdx = -1;  // define per thread local registers
    float localMin = FLT_MAX;

    for(int i = laneId; i<N; i+= WarpSize) {

        if(i == localIdx) {continue;}  // skip itself
        float diffx = a[3*localIdx] - a[3*i];
        float diffy = a[3*localIdx + 1] - a[3*i + 1];
        float diffz = a[3*localIdx + 2] - a[3*i + 2];
        float acc = diffx * diffx  + diffy * diffy + diffz * diffz;

        if(acc < localMin || (acc == localMin && i<localMinIdx)) {
            localMin = acc;
            localMinIdx = i;
        }
    }

    // Instead of using shared memory, we can do warp level reduction
    // (minimum among all elements within the warp)
    for(int i = WarpSize/2; i>0; i/=2) {
        float warpMin = __shfl_down_sync(0xFFFFFFFF, localMin, i);
        int warpMinIndex = __shfl_down_sync(0xFFFFFFFF, localMinIdx, i);

        if(warpMin < localMin || (warpMin == localMin && warpMinIndex < localMinIdx)) {
            localMin = warpMin;
            localMinIdx = warpMinIndex;
        }
    }

    if(laneId == 0) {
        c[localIdx] = localMinIdx;
    }

}


__global__ void perBlockNearestNeighbor(float* a, int* c, int N) {

    int localIdx = blockIdx.x;  // one element per block

    // Each thread independently tracks its best neighbor in registers, no shared memory needed.
    float localMin = FLT_MAX;
    int localMinIdx = -1;

    for(int i = threadIdx.x; i<N; i+=blockDim.x) {
        if(i == localIdx) {continue;}  // skip itself
        float diffx = a[3*localIdx] - a[3*i];
        float diffy = a[3*localIdx + 1] - a[3*i + 1];
        float diffz = a[3*localIdx + 2] - a[3*i + 2];
        float acc = diffx * diffx  + diffy * diffy + diffz * diffz;
        // track minimum distance index
        // Each thread independently tracks its best neighbor in registers.
        if(acc < localMin || (acc == localMin && i<localMinIdx)) {
            localMin = acc;
            localMinIdx = i;
        }
    }

    // copy Intra-Thread minimum into each thread per block
    __shared__ float blockMin[threadsPerBlock];  // threads Per block = 256
    __shared__ float blockMinIndex[threadsPerBlock];  // threads Per block = 256
    blockMin[threadIdx.x] = localMin;
    blockMinIndex[threadIdx.x] = localMinIdx;
    __syncthreads();

    // Use parallel reduction to get per block minimum index
    // If two indexes are at same distance, choose smaller index
    for(int i = blockDim.x/2; i>0; i/=2) {
        if(threadIdx.x < i) {
            float min1 = blockMin[threadIdx.x];
            float min2 = blockMin[threadIdx.x + i];
            int minId1 = blockMinIndex[threadIdx.x];
            int minId2 = blockMinIndex[threadIdx.x + i];

            if(min2 < min1 || (min2 == min1 && minId2 < minId1)) {
                blockMin[threadIdx.x] = min2;
                blockMinIndex[threadIdx.x] = minId2;
            }

        }
        __syncthreads();
    }

    // Get per element (per block) nearest neighbors index
    if(threadIdx.x == 0) {
        c[blockIdx.x] = blockMinIndex[0];
    }

}



int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int N = n;
    float h_a[N*3];
    int h_c[N];
    float *d_a;
    int *d_c;

    CHECK_CUDA(cudaMalloc(&d_a, N*3*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, N*sizeof(int)));


    // fill data in host device
    for(int i=0; i<N*3 ;i++) {
        h_a[i] = ((i + 9 + rand()) % 99)*0.08f;
        //cout<<"h_a[i]: "<<h_a[i]<<endl;
    }

    CHECK_CUDA(cudaEventRecord(start, 0));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N*3*sizeof(float), cudaMemcpyHostToDevice));


    perBlockNearestNeighbor<<<N, threadsPerBlock>>>(d_a, d_c, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c, d_c, N*sizeof(int), cudaMemcpyDeviceToHost));


    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"Elapsed time(in ms) : "<< elapsed_time<<endl;  // 641.6  // for warp-base = 4465.3

    for(int i = 0; i< 50 && i<N; i++) {
        cout<<"Nearest Neighbors result at i:"<<i<<", is: "<<h_c[i]<<", original array: "<<h_a[i]<<endl;
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}