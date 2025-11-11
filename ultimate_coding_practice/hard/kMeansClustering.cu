#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cfloat>
#include <cuda_runtime.h>
using namespace std;


#define k 10 // number of clusters (<16)
#define n 50000 // number of samples  (<50K)


#define CHECK_CUDA(call) do {                    \
    cudaError_t e = (call);                      \
    if(e != cudaSuccess) {                        \
        cout<<"CUDA Error: "<<cudaGetErrorString(e) \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl; \
        exit(1);                                     \
    }                                               \
} while(0)


__global__ void kmeanClustering(float* sx, float* sy, float* cx, float* cy, int* label, int* pointsCount, float* xsum, float* ysum, int N, int K) {

    // Per block handles single points
    int gIdx = blockIdx.x;
    int tid = threadIdx.x;
    if(gIdx >= N || tid>=16) { return;}

    __shared__ float nearestCenter[16]; // shared memory for nearest cluster per point
    __shared__ int nearestCenterId[16]; // shared memory for nearest cluster Id per point
                                        // threadsPerBlock = 16

    // Per block needs K threads to get distance from each of K centroids
    if(tid < K) {
        // get distance from all the class centers
        float dx = sx[gIdx] - cx[tid];
        float dy = sy[gIdx] - cy[tid];
        nearestCenter[tid] = dx * dx + dy * dy;
        nearestCenterId[tid] = tid;
    } else {
        nearestCenter[tid] = FLT_MAX;
        nearestCenterId[tid] = tid;
    }
    __syncthreads();

    // Parallel reduction to get per point, closest class and its index
    for(int i = blockDim.x/2; i>0; i/=2) {
        if(threadIdx.x < i) {
            if(nearestCenter[threadIdx.x + i] < nearestCenter[threadIdx.x]) {
                nearestCenter[threadIdx.x] = nearestCenter[threadIdx.x + i];
                nearestCenterId[threadIdx.x] = nearestCenterId[threadIdx.x + i];
            }
        }
        __syncthreads();
    }

    // update new class label per point from each block and compute per class points count
    if(tid == 0) {
        int newClass = nearestCenterId[0];
        label[gIdx] = newClass;
        atomicAdd(&pointsCount[newClass], 1);       // accumulate points count of each block to its class bin
        atomicAdd(&xsum[newClass], sx[gIdx]);       // accumulate x-coord of each block to its class bin
        atomicAdd(&ysum[newClass], sy[gIdx]);       // accumulate y-coord of each block to its class bin
    }

}

// Calculate new Centroid for each clusters
__global__ void calculateNewCenter(float* cx, float* cy, int* pointsCount, float* xsum, float* ysum, int K) {
    // Compute new centroid or keep the older one if no points assigned to it
    int tid = threadIdx.x;
    if(tid < K) {
        int cnt = pointsCount[tid];
        cx[tid] = (cnt > 0) ? xsum[tid]/cnt : cx[tid];
        cy[tid] = (cnt > 0) ? ysum[tid]/cnt : cy[tid];
    }

}

// To implement K-Means Clustering for 2D points. Given array of N(<50K) points in 2D plane
// initial centroids, number of classes K(<16) and maximum_iterations.
// 1. For each point i, assign K threads per block and compute distance
// of i with all the other K class centroids and find the index of minimum distance (parallel reduction).
// 2. Create label array( size N) to keep track of new class label for each points
// 3. Create histogram bin of size K for x and y coordinate sum and counts, store these values for points for each class
// 4. Compute mean coordinate of new label points to get new centroid in new Kernel
// 5. Repeat for maximum_iterations.


int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int N = n;
    int K = k;
    int max_iter = 15;


    float h_sx[N], h_sy[N], h_cx[K], h_cy[K], h_xsum[K], h_ysum[K];
    int h_l[N], h_pc[K];
    float *d_sx, *d_sy, *d_cx, *d_cy, *d_xsum, *d_ysum;
    int *d_l, *d_pc;

    CHECK_CUDA(cudaMalloc(&d_sx, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_sy, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_cx, K*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_cy, K*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_xsum, K*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ysum, K*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_l, N*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_pc, K*sizeof(int)));


    for(int i=0;i<N;i++) {
        h_sx[i] = ((rand() + i)%19)*1.0f;
        h_sy[i] = ((rand() + i*i)%19)*1.0f;
        h_l[i] = 0;
        //cout<<"h_sx: "<<h_sx[i]<<", h_sy: "<<h_sy[i]<<endl;
    }
    for(int i=0;i<K;i++) {
        h_cx[i] = ((rand() + i+1)%19)*1.0f;
        h_cy[i] = ((rand() + i*i +2)%19)*1.0f;
        h_pc[i] = 0;
        h_xsum[i] = 0.0f;
        h_ysum[i] = 0.0f;
        //cout<<"h_cx: "<<h_cx[i]<<", h_cy: "<<h_cy[i]<<endl;
    }

    CHECK_CUDA(cudaEventRecord(start, 0));

    CHECK_CUDA(cudaMemcpy(d_sx, h_sx, N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_sy, h_sy, N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_cy, h_cy, K*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_cx, h_cx, K*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_l, h_l, N*sizeof(int), cudaMemcpyHostToDevice));


    int threadsPerBlock = 16;

    for(int i= 0; i< max_iter; i++) {

        // Initialize the counters and sums at each iteration
        CHECK_CUDA(cudaMemcpy(d_pc, h_pc, K*sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_xsum, h_xsum, K*sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_ysum, h_ysum, K*sizeof(float), cudaMemcpyHostToDevice));

        // compute new centroid using K-means clustering
        kmeanClustering<<<N, threadsPerBlock>>>(d_sx, d_sy, d_cx, d_cy, d_l, d_pc, d_xsum, d_ysum, N, K);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        calculateNewCenter<<<1, threadsPerBlock>>>(d_cx, d_cy, d_pc, d_xsum, d_ysum, K);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

    }


    // copy the final coordinates of centroid
    CHECK_CUDA(cudaMemcpy(h_cx, d_cx, K*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_cy, d_cy, K*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_l, d_l, N*sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaMemcpy(h_pc, d_pc, K*sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_xsum, d_xsum, K*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_ysum, d_ysum, K*sizeof(float), cudaMemcpyDeviceToHost));
    for(int i = 0; i< K && i<20; i++) {
        cout<<"Sums & counters for class i:"<<i<<", is h_pc: "<<h_pc[i]<<", x-sum: "<<h_xsum[i]<<", y-sum: "<<h_ysum[i]<<endl;
    }

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"ELapsed time(in ms) is "<<elapsed_time<<endl; //3.34

    for(int i = 0; i< K && i<20; i++) {
        cout<<"New centroid coordinates for class i:"<<i<<", is x: "<<h_cx[i]<<", y: "<<h_cy[i]<<endl;
    }
    for(int i = 0; i< N && i<20; i++) {
        cout<<"Final label for each points at i:"<<i<<", is "<<h_l[i]<<endl;
    }


    CHECK_CUDA(cudaFree(d_sx));
    CHECK_CUDA(cudaFree(d_sy));
    CHECK_CUDA(cudaFree(d_cx));
    CHECK_CUDA(cudaFree(d_cy));
    CHECK_CUDA(cudaFree(d_l));
    CHECK_CUDA(cudaFree(d_xsum));
    CHECK_CUDA(cudaFree(d_ysum));
    CHECK_CUDA(cudaFree(d_pc));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}