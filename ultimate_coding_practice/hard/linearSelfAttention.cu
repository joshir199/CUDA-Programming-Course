#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;


#define D 512 // common dimension
#define m 512 // row for Query, Key, Value
#define Tile 16


#define CHECK_CUDA(call) do {                    \
    cudaError_t e = (call);                      \
    if(e != cudaSuccess) {                        \
        cout<<"CUDA Error: "<<cudaGetErrorString(e) \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl; \
        exit(1);                                     \
    }                                               \
} while(0)

// Matrix - Vector multiplication A[d x N] x B [N x 1]  => C[d x 1]
__global__ void matrixVecCalWithTransformOld(float* a, float* b, float* c, int d, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float partialSum = 0.0f;

    if(tid < d) {
        for(int i = 0; i<N; i++) {
            // use the kernel transform while calculating the key sums per feature
            float valA = a[tid * N + i];
            float sumA = (valA > 0) ? (valA + 1.0f) : (expf(valA));
            partialSum += sumA * b[i];
        }
    }
    __syncthreads();

    if(tid<d) {
        c[tid] = partialSum;
    }

}

// Get Matrix-Vector multiplication using per block dot product for each row
// A[M, N], B[N]  => C[M]
__global__ void matrixVecCalWithTransform(float* a, float* b, float* c, int M, int N) {

    float threadSum = 0.0f;

    // each block stores single row of A[M,N]
    for(int i = threadIdx.x; i<N; i+= blockDim.x) {

        int gIdx = blockIdx.x * N + i;
        // use the kernel transform while calculating the key sums per feature
        float valA = a[gIdx];
        float sumA = (valA > 0) ? (valA + 1.0f) : (expf(valA));
        threadSum += sumA * b[i];  // element-wise dot product
    }

    // initialize the block Sum
    __shared__ float blockSum[256];
    if(threadIdx.x < blockDim.x) {
        blockSum[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // copy the perthread sum
    blockSum[threadIdx.x] = threadSum;
    __syncthreads();

    // parallel reduction for sum
    for(int i=blockDim.x/2; i>0; i/=2) {

        if(threadIdx.x < i) {
            blockSum[threadIdx.x] += blockSum[threadIdx.x + i];
        }
        __syncthreads();
    }

    // get the per row dot product as final output C[M]
    if(threadIdx.x == 0) {
        c[blockIdx.x] = blockSum[0];
    }
}



// calculate matrix transpose for Key Matrix A[Nxd] to get C[dxN]
__global__ void transposeMat(float* a, float* c, int N, int d) {
    int x_c = threadIdx.x + blockDim.x * blockIdx.x;  // d
    int y_r = threadIdx.y + blockDim.y * blockIdx.y;  // N

    if(x_c < d && y_r < N) {
        c[x_c * N + y_r] = a[y_r * d + x_c];  // transpose by changing the values
                                              // at transpose indexes C[j][i] = A[i][j]
    }
}

//(d_qks, d_v, d_c, M, N, d, false)
// Given dense matrix A[Mxd] and B[dxN], we do matrix multiplication
// using tile to produce matrix C[MxN] with kernel transformation of matrix A
__global__ void matMulWithKernelTransform(float* a, float* b, float* c, int M, int d, int N) {
    int x_c = threadIdx.x + blockDim.x * blockIdx.x;
    int y_r = threadIdx.y + blockDim.y * blockIdx.y;

    __shared__ float cacheTileA[Tile][Tile+1];
    __shared__ float cacheTileB[Tile][Tile+1];

    float partialsum = 0.0f;
    int numtiles = (d + Tile -1)/Tile;

    for(int i = 0; i<numtiles; i++) {
        int localx_c = threadIdx.x;
        int localy_r = threadIdx.y;

        int globalx_c = i*Tile + localx_c; // For Matrix A[y_r, globalx_c]
        int globaly_r = i*Tile + localy_r; // For Matrix B[globaly_r, x_c]

        //Load data from matrix A[Mxd] to tileA
        if(y_r<M && globalx_c< d){
            //  ϕ(x) kernel for matrix A
            float valA = a[y_r * d + globalx_c];
            cacheTileA[localy_r][localx_c] = (valA > 0) ? (valA + 1.0f) : (expf(valA));
        } else {
            cacheTileA[localy_r][localx_c] = 0.0f;
        }

        //Load data from matrix B to tileB
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

    if(x_c< N && y_r < M) {
        c[y_r * N + x_c] = partialsum;
    }

}

// Implement Linear Attention for a given set of matrices, following the method described
// in "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention".
// It replaces the softmax function with a kernel feature map ϕ(x) to get linear attention (O(N))
// ϕ(x) is a function that makes all features non-negative (so it can act like softmax).
// Example: ϕ(x)=elu(x)+1  = { x + 1; x>0   |  exp(x) + 1 ; x<=0
// attention(Q,K,V) = ϕ(Q)(ϕ(K_t)V) / ϕ(Q)(ϕ(K_t)1_vec)  ; 1_vec is a vector of ones (for normalization).
// So, steps to calculate linear self-attention includes:
// 1. MatMul : ϕ(K_t)xV [dxM, Mxd], ϕ(Q)x(result) [Mxd, dxd]
// 2. MatVecMul : ϕ(K_t)x1_vec [dxM, Mx1], ϕ(Q)x(result) [Mxd, dx1]
// 3. Row wise division of Numerator with denominator

// Analysis: In standard attention: You’d compute QK_t -> O(N^2D),
// but if we change order: This changes order:
// (QK_t)V  →  (K_tV)Q :=> cost goes from O(N^2D) -> O(ND^2) [No pairwise computation)


// Numerator Matrix A[Mxd] divided row-wise by denominator vector B[M]  => C[Mxd]
__global__ void finalDivisionKernel(float* a, float* b, float* c, int M, int d) {
    int tid = threadIdx.x; // per block will handle feature side d  (each row)
    int gIdx = blockIdx.x * d + tid;

    if(tid < d && gIdx<M*d) {
        float denom = (b[blockIdx.x] == 0.0f) ? 0.00001f : b[blockIdx.x]; // check for division by zero
        c[gIdx] = a[gIdx] / denom;
    }

}



int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    int M = m;
    int N = m;
    int d = D;


    float h_q[M*d], h_k[N*d], h_v[N*d], h_one[N], h_c[M*d];
    float *d_q, *d_k, *d_kt, *d_v, *d_ktv, *d_kts, *d_qkn, *d_qkd, *d_one, *d_c;

    CHECK_CUDA(cudaMalloc(&d_q, M*d*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_k, N*d*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_kt, d*N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_v, N*d*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ktv, d*d*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_kts, d*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_qkn, M*d*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_qkd, M*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_one, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, M*d*sizeof(float)));

    for(int i=0;i<M;i++) {
        for(int j =0; j<d; j++) {
            h_q[i*d + j] = (rand() + j ) % (99) * 1.0f + 1.0f;
        }
    }

    for(int i=0;i<N;i++) {

        for(int j =0; j<d; j++) {
            h_k[i*d + j] = ((rand() + 3)%9)*0.3f;
        }
        h_one[i] = 1.0f;
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
    CHECK_CUDA(cudaMemcpy(d_one, h_one, N*sizeof(float), cudaMemcpyHostToDevice));

    // Step 1: get the transpose of Key matrix
    dim3 block(Tile, Tile);
    dim3 grid((d + Tile - 1)/Tile, (N + Tile - 1)/Tile);
    transposeMat<<<grid, block>>>(d_k, d_kt, N, d);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());


    // Step 2
    // For Numerator: Calculate the matrix multiplication of Key transpose with Value = (ϕ(K_T) x V)
    // with kernel transformation
    dim3 block1(Tile, Tile);
    dim3 grid1((d+Tile-1)/Tile, (d+Tile -1)/Tile);
    matMulWithKernelTransform<<<grid1, block1>>>(d_kt, d_v, d_ktv, d, M, d);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());


    // Step 3
    // For Numerator: Calculate the matrix multiplication of Query Matrix with (Key, Value) = ϕ(Q) x (ϕ(K_T) x V)
    // with kernel transformation
    dim3 block2(Tile, Tile);
    dim3 grid2((d+Tile-1)/Tile, (M+Tile -1)/Tile);
    matMulWithKernelTransform<<<grid2, block2>>>(d_q, d_ktv, d_qkn, M, d, d);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());


    // Step 4
    // For Denominator: Calculate Matrix Vector multiplication with vector consisting of ones only.
    // To calculate : ϕ(K_t) x 1_vec  [d, M].[M]
    int threadsPerBlock = 256;
    matrixVecCalWithTransform<<<d, threadsPerBlock>>>(d_kt, d_one, d_kts, d, M);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());



    // Step 5
    // For Denominator: Calculate Matrix Vector multiplication with vector from above.
    // To calculate : ϕ(Q) x (ϕ(K_t) x 1_vec)  [M, d].[d]
    matrixVecCalWithTransform<<<M, threadsPerBlock>>>(d_q, d_kts, d_qkd, M, d);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    //float h_test[M];
    //CHECK_CUDA(cudaMemcpy(h_test, d_qkd, M*sizeof(float), cudaMemcpyDeviceToHost));
    //for(int i = 0; i<M; i++) {
    //    cout<<" h_test: "<<h_test[i]<<endl;
    //}

    // Do the final linear attention by dividing with denominator row-wise
    int blocksPerGrid1 = M;
    finalDivisionKernel<<<blocksPerGrid1, threadsPerBlock>>>(d_qkn, d_qkd, d_c, M, d);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // copy the final attention values for M queries
    CHECK_CUDA(cudaMemcpy(h_c, d_c, M*d*sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"ELapsed time(in ms) is "<<elapsed_time<<endl; //1.45

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
    CHECK_CUDA(cudaFree(d_kts));
    CHECK_CUDA(cudaFree(d_qkd));
    CHECK_CUDA(cudaFree(d_qkn));
    CHECK_CUDA(cudaFree(d_one));
    CHECK_CUDA(cudaFree(d_c));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}