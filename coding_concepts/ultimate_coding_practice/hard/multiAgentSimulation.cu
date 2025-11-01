#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;

#define n 222222
#define threadsPerBlock 256

#define CHECK_CUDA(call) do {                \
    cudaError_t e = (call);                     \
    if(e != cudaSuccess) {                          \
        cerr<<"CUDA Error:"<<cudaGetErrorString(e)      \
        <<" at "<<__FILE__<<", line:"<<__LINE__<<endl;  \
        exit(1);                                        \
    }                                           \
} while(0)


// Compute multi-agent simulation using the neighbors for each agent
// to calculate the updated velocity and position
__global__ void agentSimulation(float* a, float* c, int N, float alpha) {

    int localIdx = blockIdx.x;

    int neighborCount = 0;
    float velocity_x = 0;
    float velocity_y = 0;

    // load and get the partial velocities sum per thread
    for(int i = threadIdx.x; i<N; i+=blockDim.x) {
        float dx = a[4*localIdx] - a[4*i];  // x-coordinate
        float dy = a[4*localIdx + 1] - a[4*i + 1];  // y-coordinate
        float dist = (i == localIdx) ? 26.0f : (dx * dx + dy * dy); // when i == j, make it above threshold=25.0
        if(dist < 25.0f) {
            neighborCount += 1;         // get accumulated count per thread
            velocity_x += a[4*i + 2];   // get accumulated velocity x-component per thread
            velocity_y += a[4*i + 3];   // get accumulated velocity y-component per thread
        }
    }
    __syncthreads();

    // copy the per thread count, velocity_x and velocity_y into each block array
    __shared__ int blockNeighborCount[threadsPerBlock];
    __shared__ float blockVelXSum[threadsPerBlock];
    __shared__ float blockVelYSum[threadsPerBlock];

    blockNeighborCount[threadIdx.x] = neighborCount;
    blockVelXSum[threadIdx.x] = velocity_x;
    blockVelYSum[threadIdx.x] = velocity_y;
    __syncthreads();

    // get the per block total count and velocity sum
    for(int i = blockDim.x/2; i>0; i/=2) {
        if(threadIdx.x < i) {
            blockNeighborCount[threadIdx.x] += blockNeighborCount[threadIdx.x + i];
            blockVelXSum[threadIdx.x] += blockVelXSum[threadIdx.x + i];
            blockVelYSum[threadIdx.x] += blockVelYSum[threadIdx.x + i];
        }
        __syncthreads();
    }


    // using single thread per block, get the output result for one elements per block
    if(threadIdx.x == 0) {

        float vx = a[4*localIdx + 2];
        float vy = a[4*localIdx + 3];
        float vx_avg = (blockNeighborCount[0] > 0) ? blockVelXSum[0]/blockNeighborCount[0] : vx;
        float vy_avg = (blockNeighborCount[0] > 0) ? blockVelYSum[0]/blockNeighborCount[0] : vy;

        float vx_new = vx + alpha * (vx_avg - vx);  // update x-velocity
        float vy_new = vy + alpha * (vy_avg - vy);  // update y-velocity

        // update the velocity components
        c[4*localIdx + 2] = vx_new;
        c[4*localIdx + 3] = vy_new;

        // update the position components
        c[4*localIdx] = a[4*localIdx] + vx_new;
        c[4*localIdx + 1] = a[4*localIdx + 1] + vy_new;

    }

}


// Multi-Agent simulation
// Given an array of agents with each agent having its 2D position and associated velocity vector
// We need to simulate to compute updated velocity and its position
// Simulation Rules involves:
// 1. Get the neighbors(excluding itself) of each agents which is strictly within circular area of
//    radius = 5.0  =>  (x_i - x_j)^2 + (y_i - y_j)^2 < 25.0  => for all j (!= i) to be neighbor
// 2. Calculate the average of neighbors velocities, with number of neighbors of i as N_i
//    v_avg_i = N_i > 0 ? avg(velocities of neighbors of i) : v_i
// 3. Update the velocity of the agent with average of neighbors velocity
//    v_new_i = v_i + alpha (v_avg - v_i)  , alpha = 0.05 (fixed)
// 4. Update the position of the agent using updated velocity
//    p_new_i = P_i + v_new_i


int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int N = n;
    int alpha = 0.05;

    float h_a[N*4], h_c[N*4];
    float *d_a, *d_c;

    CHECK_CUDA(cudaMalloc(&d_a, N*4*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, N*4*sizeof(float)));

    for(int i =0; i<N;i++){
        h_a[4*i] = ((rand() + i + 1)%9) * 1.0f;  // x
        h_a[4*i + 1] = ((rand() + i + 1)%9) * 1.0f;   // y
        h_a[4*i + 2] = ((rand() + i + 1)%9) * 0.07f;   // vx
        h_a[4*i + 3] = ((rand() + i + 1)%9) * 0.07f;   // vy
    }

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N*4*sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaEventRecord(start, 0));

    int blocksPerGrid = N;
    agentSimulation<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_c, N, alpha);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c, d_c, N*4*sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"Elapsed time(in ms): "<<elapsed_time<<endl;   // 306.2

    for(int i = 0; i<30 && i<4*N; i++) {
        cout<<"Multi-agent simulated result, at i"<<i<<", is "<<h_c[i]<<", before: "<<h_a[i]<<endl;
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_c));


    return 0;
}