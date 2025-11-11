#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;


#define m 768 // row
#define n 768 // col
#define threads 16
#define UINT_VERTEX (m*n + 5)

#define CHECK_CUDA(call) do {                    \
    cudaError_t e = (call);                      \
    if(e != cudaSuccess) {                        \
        cout<<"CUDA Error: "<<cudaGetErrorString(e) \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl; \
        exit(1);                                     \
    }                                               \
} while(0)


__constant__ int horx_c[4];
__constant__ int very_r[4];

// Grid Search using BFS in a incremental order by checking all its neighbors at each step
// Once, destinationIndex is found, finish.
__global__ void bfsGraphSearch(int* a, int* level, int* newVertexVisited, int curLevel, int dst, int M, int N) {

    int x_c = threadIdx.x + blockIdx.x * blockDim.x;
    int y_r = threadIdx.y + blockIdx.y * blockDim.y;

    if(x_c>=N || y_r>=M) { return;}
    
    int gIdx = y_r * N + x_c;   // threadId for matrix elements

    if(level[gIdx] == curLevel - 1) {
        
        // neighbor coordinate : down, up, right, left
        for(int i = 0; i<4; i++) {
            int nIdx = x_c + horx_c[i];
            int nIdy = y_r + very_r[i];
            
            // Add out of limit neighbor check
            if(nIdx >= 0 && nIdx < N && nIdy >= 0 && nIdy < M) {
                int nId = nIdy * N + nIdx;
                // check for free & unvisited block
                if(level[nId] == UINT_VERTEX && (a[nId] == 0)) {
                    level[nId] = curLevel;
                    if(nId == dst) {
                        *newVertexVisited = 0;   // change the value to 0 if destination found 
                        break;
                    }
                }
            }
        }
        
    }

}

int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int M=m;
    int N=n;
    int r_st = 0;   // starting grid index
    int c_st = 1;
    int r_en = M-7; // ending grid index
    int c_en = N-1;

    int h_a[M*N], h_l[M*N], h_nf, h_c[M*N], h_len;
    int *d_a, *d_l, *d_nf;

    CHECK_CUDA(cudaMalloc(&d_l, M*N*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_a, M*N*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_nf, sizeof(int))); // flag to track if needed more iteration or not

    for(int i=0;i<M;i++) {
        for(int j =0; j<N; j++) {
            h_a[i*N + j] = (j==i)?1:0; //(rand() + j) % 2;    // 0 - free, 1 - blockers
            //cout<<"h_a[i]: "<<h_a[i*N + j]<<endl;
        }
    }
    int srcId = r_st * N + c_st;    // get the row-major Id for start & end points on grid
    int dstId = r_en * N + c_en;
    h_a[srcId] = 0;   // make sure the start block is free
    h_a[dstId] = 0;   // make sure the end block is free
    

    for(int i=0;i<M*N;i++) {
        h_l[i] = UINT_VERTEX;
    }
    int curLevel = 0;
    h_l[srcId] = curLevel; // set the source vertex to level 0

    // neighbor coordinate : down, up, right, left
    int h_cxc[4] = {0, 0, 1, -1}; // first vertical and then horizontal
    int h_cyr[4] = {1, -1, 0, 0};

    h_nf = (srcId==dstId) ? 0 : 1; // if src == dst, not need to search

    // copy data in constant memory
    CHECK_CUDA(cudaMemcpyToSymbol(very_r, h_cyr, 4*sizeof(int)));
    CHECK_CUDA(cudaMemcpyToSymbol(horx_c, h_cxc, 4*sizeof(int)));

    CHECK_CUDA(cudaEventRecord(start, 0));

    CHECK_CUDA(cudaMemcpy(d_l, h_l, M*N*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_a, h_a, M*N*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_nf, &h_nf, sizeof(int), cudaMemcpyHostToDevice));

    dim3 blocks(threads, threads);
    dim3 grid((N + threads - 1)/threads, (M + threads - 1)/threads);

    while(h_nf && curLevel<(M+N+1)) {

        curLevel += 1;

        // kernel to do bfs iteration level-wise
        bfsGraphSearch<<<grid, blocks>>>(d_a, d_l, d_nf, curLevel, dstId, M, N);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        // update the flag for next iteration
        CHECK_CUDA(cudaMemcpy(&h_nf, d_nf, sizeof(int), cudaMemcpyDeviceToHost));
        //cout<<"h_nf: "<<h_nf<<endl;
    }
    
    cout<<"curLevel: "<<curLevel<<endl;

    CHECK_CUDA(cudaMemcpy(h_c, d_l, M*N*sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"ELapsed time(in ms) is "<<elapsed_time<<endl; //20.64
    
    // Get the shortest path based on difference conditions
    if(dstId == srcId) {
        h_len = 0;
    } else if(h_c[dstId] == UINT_VERTEX) {
        h_len = -1;
    } else {
        h_len = h_c[dstId];
    }

    //Shortest path from source index at src:1 and destination Id 585215, is: 1527
    cout<<"Shortest path from source index at src:"<<srcId<<" and destination Id "<<dstId<<", is: "<<h_len<<endl; 
    

    CHECK_CUDA(cudaFree(d_l));
    CHECK_CUDA(cudaFree(d_nf));


    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}