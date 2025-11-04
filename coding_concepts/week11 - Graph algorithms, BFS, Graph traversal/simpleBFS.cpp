#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;


#define m 768 // row
#define threadsPerBlock 256
#define UINT_VERTEX (m+1)

#define CHECK_CUDA(call) do {                    \
    cudaError_t e = (call);                      \
    if(e != cudaSuccess) {                        \
        cout<<"CUDA Error: "<<cudaGetErrorString(e) \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl; \
        exit(1);                                     \
    }                                               \
} while(0)


// Vertex-Centric BFS algorithm which follows Top-Down approach. It visits the vertex and finds all the non-visited
// neighboring vertex and assign them to next level and update about more iteration needed using the flag
__global__ void bfsVertexCentric(int* colInd, int* rowptr, int* level, int* newVertexVisited, int curLevel, int M) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x; // each vertex assigned one thread

    if(tid>=M) { return;}
    // process only those vertices which have been saved in previous iteration as new level vertices
    // or cal it current frontier
    if(level[tid] == curLevel - 1) {
        // loop through all connected vertex from this
        int startId = rowptr[tid];  // starting index in row tid (inclusive)
        int endId = rowptr[tid + 1];   // starting index in row (tid + 1)

        for(int i = startId; i<endId; i++) {
            int dstId = colInd[i]; // get the destination vertex from column index pointer

            // check if the destination vertex is not visited yet
            if(level[dstId] == UINT_VERTEX) {
                // update the level of this destination vertex
                level[dstId] = curLevel;

                // Update Flag that at least one new vertex was discovered to iterate more
                *newVertexVisited = 1;
            }
        }

    }
}

// CSR pointers where each values are 1 only in un-weighted graph
void getCSRpointers(int* mat, int M, vector<int> &values, vector<int> &colIndex, vector<int> &rowptrs) {

    int x_val = 0;
    rowptrs.clear();
    rowptrs.push_back(0);
    int temp = 0;

    for(int i = 0; i<M; i++) { // row
        for(int j = 0; j<M; j++) {  // col
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

    int M=m;
    int rootVertex = 0; // rootVertex or starting vertex (< M)

    int h_a[M*M], h_l[M], h_nf, h_c[M];
    int *d_l, *d_nf;

    CHECK_CUDA(cudaMalloc(&d_l, M*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_nf, sizeof(int))); // flag to track if needed more iteration or not

    for(int i=0;i<M;i++) {
        for(int j =0; j<M; j++) {
            h_a[i*M + j] = (rand() + j) % 2;
            //cout<<"h_a[i]: "<<h_a[i*M + j]<<endl;
        }
    }

    for(int i=0;i<M;i++) {
        h_l[i] = UINT_VERTEX;
    }
    int curLevel = 0;
    h_l[rootVertex] = curLevel; // set the root vertex to level 0

    h_nf = (M>0) ? 1 : 0; // it root vertex exists

    //all non-zero elements, row by row ? length = nnz
    vector<int> h_values;
    // the column index of each value ? length = nnz
    vector<int> h_colInd;
    //cumulative count of non-zeros up to each row ? length = M + 1
    vector<int> h_rowPtr;

    getCSRpointers(h_a, M, h_values, h_colInd, h_rowPtr);

    int nnz = h_colInd.size();

    int *d_colInd, *d_rowptr; // No need of values

    CHECK_CUDA(cudaMalloc(&d_colInd, nnz*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_rowptr, (M+1)*sizeof(int)));


    CHECK_CUDA(cudaEventRecord(start, 0));

    CHECK_CUDA(cudaMemcpy(d_l, h_l, M*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_colInd, h_colInd.data(), nnz*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_rowptr, h_rowPtr.data(), (M+1)*sizeof(int), cudaMemcpyHostToDevice));

    int blocksPerGrid = (M + threadsPerBlock - 1)/threadsPerBlock;

    curLevel = 1;

    do {
        // update the flag for next iteration
        h_nf = 0;
        CHECK_CUDA(cudaMemcpy(d_nf, &h_nf, sizeof(int), cudaMemcpyHostToDevice));

        // kernel to do bfs iteration level-wise
        bfsVertexCentric<<<blocksPerGrid, threadsPerBlock>>>(d_colInd, d_rowptr, d_l, d_nf, curLevel, M);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(&h_nf, d_nf, sizeof(int), cudaMemcpyDeviceToHost));
        curLevel += 1;

    } while (h_nf);


    CHECK_CUDA(cudaMemcpy(h_c, d_l, M*sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"ELapsed time(in ms) is "<<elapsed_time<<endl; //0.56

    for(int i = 0; i< nnz && i<20; i++) {
        cout<<"CSR indexing format at i:"<<i<<", colId:"<<h_colInd[i]<<endl;
    }

    for(int i = 0; i< M && i<20; i++) {
        cout<<"Vertex levels in graph search with (0 as root vertex) using at i:"<<i<<" is "<<h_c[i]<<endl;
    }

    CHECK_CUDA(cudaFree(d_rowptr));
    CHECK_CUDA(cudaFree(d_colInd));
    CHECK_CUDA(cudaFree(d_l));
    CHECK_CUDA(cudaFree(d_nf));


    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}