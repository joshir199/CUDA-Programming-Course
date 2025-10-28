#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cfloat>
#include <cuda_runtime.h>
using namespace std;


#define n 512 // row for Key, Value
#define D 128 // common dimension
#define m 512 // row for Query
#define threadsPerBlock 256
#define Tile 16


#define CHECK_CUDA(call) do {                    \
    cudaError_t e = (call);                      \
    if(e != cudaSuccess) {                        \
        cout<<"CUDA Error: "<<cudaGetErrorString(e) \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl; \
        exit(1);                                     \
    }                                               \
} while(0)

// softmax function for matrix A[MxN] with casual self-attention inclusion
// Per Row softmax calculation
__global__ void softmaxFunction(float* a, float* c, int M, int N) {

    extern __shared__ float cache[];  // one dynamic shared memory of size 2*N
    float* cacheMax = cache;  // start at 0-th index of cache array till N-1 index
    float* cacheSum = &cache[N];  // start at N-th index of cache array till 2N-1 index
    int rowOffset = blockIdx.x * N; // one row per block

    // each threads loads multiple elements (as N can be 100000)
    for(int i=threadIdx.x; i<N; i+=blockDim.x) {
        cacheMax[i] = a[i + rowOffset];
    }
    __syncthreads();

    // reduction to find max
    // In case where single threads loads multiple elements, each thread is responsible for multiple values
    // Therefore, we need to apply two-stage reduction strategy
    // Stage 1 — Intra-thread reduction (done in your loop)
    //  - Each thread finds the maximum of its chunk(number of elements each process).
    //  - O(N / blockDim.x) time per thread.
    float localMaxVal = -150.0f; // per thread, all elements are in range [-100.0, 100.0]
    for(int i=threadIdx.x; i<N; i+=blockDim.x) {
        localMaxVal = fmaxf(cacheMax[i], localMaxVal);
    }
    __syncthreads();

    // After getting per thread Max value, we can call stage 2
    // Stage 2 — Inter-thread reduction (could be tree-based)
    //  - Combine blockDim.x local maxima.
    //  - O(log₂(blockDim.x)) time if you use tree-based reduction in shared memory
    __shared__ float blockMax[256];  // threads Per block = 256
    blockMax[threadIdx.x] = localMaxVal;
    __syncthreads();

    // Use parallel reduction
    for(int i = blockDim.x/2; i>0; i/=2) {
        if(threadIdx.x < i) {
            blockMax[threadIdx.x] = fmaxf(blockMax[threadIdx.x], blockMax[threadIdx.x + i]);
        }
        __syncthreads();
    }


    // ******************* Part 2 *********************************
    // Now, we will do the similar operation for getting exponential sums per block and later
    // calculate normalised softmax values
    // get the exponential of (elements - maxElement per row) for numerical stability
    // each threads loads multiple elements (as N can be 100000)
    for(int i=threadIdx.x; i<N; i+=blockDim.x) {
        float maskedElement = (a[i + rowOffset] == -150.0f) ? 0.0f : 1.0f;  // set masked element to 0
        cacheSum[i] = maskedElement * expf(a[i + rowOffset] - blockMax[0]);
    }
    __syncthreads();

    // Since each thread loads multiple elements, we will apply 2-stage strategy
    // 1. Intra thread accumulation of values
    float localSum = 0.0f;
    for(int i=threadIdx.x; i<N; i+=blockDim.x) {
        localSum += cacheSum[i];
    }
    __syncthreads();


    // 2. For Inter-thread processing,
    __shared__ float blockSum[256];  // threads Per block = 256
    blockSum[threadIdx.x] = localSum;
    __syncthreads();


    // Use parallel reduction
    for(int i = blockDim.x/2; i>0; i/=2) {
        if(threadIdx.x < i) {
            blockSum[threadIdx.x] += blockSum[threadIdx.x + i];
        }
        __syncthreads();
    }

    // copy the final softmax value to output matrix
    for(int i=threadIdx.x; i<N; i+=blockDim.x) {
       // check for zero to avoid division by zero
       // FLT_EPSILON: it’s the smallest representable float difference.
       c[i + rowOffset] = cacheSum[i] / fmaxf(blockSum[0], FLT_EPSILON);
    }
}

// calculate matrix transpose for Matrix A[Nxd] to get C[dxN]
__global__ void transposeMat(float* a, float* c, int N, int d) {
    int x_c = threadIdx.x + blockDim.x * blockIdx.x; // d
    int y_r = threadIdx.y + blockDim.y * blockIdx.y;  // N

    if(x_c < d && y_r < N) {
        c[x_c * N + y_r] = a[y_r * d + x_c];  // transpose by changing the values
                                              // at transpose indexes C[j][i] = A[i][j]
    }
}

//(d_qks, d_v, d_c, M, N, d, false)
// Given dense matrix A[Mxd] and B[dxN], we do matrix multiplication
// using tile to produce matrix C[MxN]
__global__ void matMul(float* a, float* b, float* c, int M, int d, int N, bool scaled) {
    int x_c = threadIdx.x + blockDim.x * blockIdx.x;
    int y_r = threadIdx.y + blockDim.y * blockIdx.y;

    __shared__ float cacheTileA[Tile][Tile+1];
    __shared__ float cacheTileB[Tile][Tile+1];

    float partialsum = 0.0f;
    int numtiles = (d+ Tile -1)/Tile;

    for(int i = 0; i<numtiles; i++) {
        int localx_c = threadIdx.x;
        int localy_r = threadIdx.y;

        int globalx_c = i*Tile + localx_c; // For Matrix A[y_r, globalx_c]
        int globaly_r = i*Tile + localy_r; // For Matrix B[globaly_r, x_c]

        //Load data from matrix A[Mxd] to tileA
        if(y_r<M && globalx_c< d){
            cacheTileA[localy_r][localx_c] = a[y_r * d + globalx_c];
        } else {
            cacheTileA[localy_r][localx_c] = 0.0f;
        }

        //Load data from matrix A to tileA
        if(globaly_r<d && x_c< N){
            cacheTileB[localy_r][localx_c] = b[globaly_r * N + x_c];
        } else {
            cacheTileB[localy_r][localx_c] = 0.0f;
        }
        __syncthreads();

        for(int k=0; k< Tile; k++) {
            partialsum += cacheTileA[localy_r][k] * cacheTileB[k][localx_c];
        }
        __syncthreads();

    }

    if(scaled) { // If scaled matrix multiplication as in attention
        // To Implement Causal (masked) Self-Attention used in Decoder of transformer model,
        // It uses a causal mask that sets all positions corresponding to keys after the current query to
        // -INFINITY, which in turn becomes 0 during softmax layer.
        // Include casual mask in QK_T, QK_T[row][col] = -INFINITY { row (query) < col (key)
        // and for other cases, it will be as usual
        if(x_c< N & y_r < M) {
            float mask = -150.0f;  // elements are sampled from [-100.0, 100.0]
            //To avoid warp divergence, you can compute the same result branchlessly (to avoid if())
            float condition = (x_c > y_r) ? 1.0f : 0.0f;
            c[y_r * N + x_c] = condition * mask + (1.0f - condition) * partialsum * rsqrtf(d); // 1/sqrt(d) - reciprocal of sqrt
        }

    } else {
        if(x_c< N & y_r < M) {
            c[y_r * N + x_c] = partialsum;
        }
    }
}


int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    int M = m;
    int N = n;
    int d = D;


    float h_q[M*d], h_k[N*d], h_v[N*d], h_c[M*d];
    float *d_q, *d_k, *d_kt, *d_v, *d_qk, *d_qks, *d_c;

    CHECK_CUDA(cudaMalloc(&d_q, M*d*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_k, N*d*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_kt, d*N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_v, N*d*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_qk, M*N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_qks, M*N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, M*d*sizeof(float)));

    for(int i=0;i<M;i++) {
        for(int j =0; j<d; j++) {
            h_q[i*d + j] = (rand() + j ) % (99) * 1.0f + 1.0f;
        }
    }
    for(int i=0;i<N;i++) {
        for(int j =0; j<d; j++) {

            if(i==j) {
                h_k[i*d + j] = 1.0f;
            } else {
                h_k[i*d + j] = 0.0f;
            }
        }
    }
    for(int i=0;i<N;i++) {
        for(int j =0; j<d; j++) {
            h_v[i*d + j] = ((i*d + j) % 9 )*1.0f + 1.0f;
        }
    }

    CHECK_CUDA(cudaEventRecord(start, 0));

    CHECK_CUDA(cudaMemcpy(d_q, h_q, M*d*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_k, h_k, N*d*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_v, h_v, N*d*sizeof(float), cudaMemcpyHostToDevice));

    // get the transpose of Key matrix
    dim3 block(16, 16);
    dim3 grid((d+15)/16, (N+15)/16);
    transposeMat<<<grid, block>>>(d_k, d_kt, N, d);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Calculate the matrix multiplication of Query with Key^T = QxK_T (dot product)
    dim3 block1(16, 16);
    dim3 grid1((N+15)/16, (M+15)/16);
    matMul<<<grid1, block1>>>(d_q, d_kt, d_qk, M, d, N, true);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Calculate softmax values for each query showing attention towards different keys
    // This is scaled softmax and softmax are done in row-wise manner = softmax(QxK_T / sqrt(d))
    int blocksPerGrid = M; // each block process single row
    size_t shmem = (2*N)*sizeof(float);
    softmaxFunction<<<blocksPerGrid, threadsPerBlock, shmem>>>(d_qk, d_qks, M, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Do the final Matrix multiplication of attention weights with each key elements
    // attention weights x Value
    dim3 block2(16, 16);
    dim3 grid2((d+15)/16, (M+15)/16);
    matMul<<<grid2, block2>>>(d_qks, d_v, d_c, M, N, d, false);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // copy the final attention values for M queries
    CHECK_CUDA(cudaMemcpy(h_c, d_c, M*d*sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"ELapsed time(in ms) is "<<elapsed_time<<endl; //3.05

    for(int i = 0; i< M*d && i<20; i++) {
        cout<<"Query matrix at i:"<<i<<", is "<<h_q[i]<<endl;
    }
    for(int i = 0; i< N*d && i<20; i++) {
        cout<<"Value Matrix at i:"<<i<<", is "<<h_v[i]<<endl;
    }
    for(int i = 0; i< M*d && i<20; i++) {
        cout<<"Attention output at i:"<<i<<" is "<<h_c[i]<<endl;
    }

    CHECK_CUDA(cudaFree(d_q));
    CHECK_CUDA(cudaFree(d_k));
    CHECK_CUDA(cudaFree(d_v));
    CHECK_CUDA(cudaFree(d_kt));
    CHECK_CUDA(cudaFree(d_qk));
    CHECK_CUDA(cudaFree(d_qks));
    CHECK_CUDA(cudaFree(d_c));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}