#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;


#define N 1024 // column
#define M 1024 // row
#define threadsPerBlock 256
#define Warp_Size 32 // define warp size

#define CHECK_CUDA(call) do {                    \
    cudaError_t e = (call);                      \
    if(e != cudaSuccess) {                        \
        cout<<"CUDA Error: "<<cudaGetErrorString(e) \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl; \
        exit(1);                                     \
    }                                               \
} while(0)

// Define constant memory to store Vector which is accessed multiple times
__constant__ int vec[N];

// CSR based SpMV using Warp level computation for :
// 1. Better load balancing across threads within a row.
// 2. Coalesced memory accesses when rows are long.
// 3. Fewer idle threads.
__global__ void sparseMatVecMul(int* values, int* colInd, int* rowptr, int* c) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = tid/Warp_Size;  // per warp, there will be Warp_Size amounts of threads
    int lane = threadIdx.x % Warp_Size; // to get actual thread lane per warp


    if(warpId<M) { // Warp_Size threads per row (M= number of rows)
        int sum = 0;  // sum variable for result per Row
        int start = rowptr[warpId];  // inclusive
        int end = rowptr[warpId + 1];  // exclusive

        // each threads in warp will process different elements of the row
        for(int i = start + lane; i<end; i += Warp_Size) {
            sum += values[i] * vec[colInd[i]];  // values * value at same index in vector
        }

        // Warp-level reduction using shuffle
        // (sum all partial sums within the warp)
        //Warp Reduction (__shfl_down_sync) : An intrinsic that allows threads in the same warp
        // to exchange register values and thus, no shared memory needed.
        // T __shfl_down_sync(unsigned mask, T var, unsigned delta, int width=warpSize);
        // mask — bitmask of active threads in the warp. Typically 0xFFFFFFFF (for 32 active threads)
        // var — the register value to shuffle (here "sum")
        // delta — how far to shift values down (offset as "i")
        // width — optional, limits the shuffle to smaller groups (default = 32)
        int i = Warp_Size/2;
        while(i>0) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, i);
            i = i/2;
        }
        // Resultant sum is accumulated at very first thread in warp
        if(lane == 0) {
            c[warpId] = sum;
        }
    }
}

void getCSRpointers(int* mat, vector<int> &values, vector<int> &colIndex, vector<int> &rowptrs) {

    int x_val = 0;
    rowptrs.clear();
    rowptrs.push_back(0);
    int temp = 0;

    for(int i = 0; i<M; i++) {
        for(int j = 0; j<N; j++) {
            temp = mat[i*M + j];
            if(temp != 0) {
                values.push_back(temp);
                colIndex.push_back(j);
                x_val++;
            }
        }
        rowptrs.push_back(x_val);
    }
}

int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));


    int h_a[M*N], h_b[N], h_c[M];
    int *d_c;

    CHECK_CUDA(cudaMalloc(&d_c, M*sizeof(int)));

    for(int i=0;i<M;i++) {
        for(int j =0; j<N; j++) {
            h_a[i*M + j] = ((i + j)/2) % 2;
        }
    }

    for(int i=0;i<N;i++) {
        h_b[i] = (2*i + 1)%999;
    }

    // Assign values to constant memory (vector) which won't be changing
    // Since, vector is getting accessed multiple times, so better store it as constant
    CHECK_CUDA(cudaMemcpyToSymbol(vec, h_b, N*sizeof(int)));

    //all non-zero elements, row by row ? length = nnz
    vector<int> h_values;
    // the column index of each value ? length = nnz
    vector<int> h_colInd;
    //cumulative count of non-zeros up to each row ? length = M + 1
    vector<int> h_rowPtr;

    getCSRpointers(h_a, h_values, h_colInd, h_rowPtr);

    int nnz = h_values.size();

    int *d_val, *d_colInd, *d_rowptr;

    CHECK_CUDA(cudaMalloc(&d_val, nnz*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_colInd, nnz*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_rowptr, (M+1)*sizeof(int)));


    CHECK_CUDA(cudaEventRecord(start, 0));

    CHECK_CUDA(cudaMemcpy(d_val, h_values.data(), nnz*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_colInd, h_colInd.data(), nnz*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_rowptr, h_rowPtr.data(), (M+1)*sizeof(int), cudaMemcpyHostToDevice));

    // Need to include warp_size as each row will have warp_size threads
    int blocksPerGrid = (M*Warp_Size + threadsPerBlock - 1)/threadsPerBlock;

    sparseMatVecMul<<<blocksPerGrid, threadsPerBlock>>>(d_val, d_colInd, d_rowptr, d_c);


    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c, d_c, M*sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"ELapsed time(in ms) is "<<elapsed_time<<endl; //0.37

    for(int i = 0; i< nnz && i<20; i++) {
        cout<<"CSR indexing format at i:"<<i<<", colId:"<<h_colInd[i]<<", value:"<<h_values[i]<<endl;
    }
    for(int i = 0; i< N && i<20; i++) {
        cout<<"Vector at i:"<<i<<", is "<<h_b[i]<<endl;
    }
    for(int i = 0; i< M && i<20; i++) {
        cout<<"Sparse Matrix-Vector multiplication using at i:"<<i<<" is "<<h_c[i]<<endl;
    }

    CHECK_CUDA(cudaFree(d_rowptr));
    CHECK_CUDA(cudaFree(d_val));
    CHECK_CUDA(cudaFree(d_colInd));
    CHECK_CUDA(cudaFree(d_c));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}