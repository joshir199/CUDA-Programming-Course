// conv1d_shared_const.cu
// 1D convolution: shared memory tiling + kernel in constant memory
// Supports input length up to ~1.5M and kernel length up to 2047
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do {                                   \
    cudaError_t _e = (call);                                    \
    if (_e != cudaSuccess) {                                    \
        fprintf(stderr, "CUDA error: %s (%d) at %s:%d\n",       \
                cudaGetErrorString(_e), (int)_e, __FILE__, __LINE__); \
        exit(1);                                                \
    }                                                           \
} while (0)

// Maximum kernel size we'll support in constant memory (2048 to be safe)
constexpr int KMAX = 2048;
__constant__ float d_kernel_const[KMAX]; // host will copy at runtime

// Shared-memory convolution kernel
// in: input array (length N)
// out: output array (length N), "same" convolution with zero padding
// N: input length
// K: kernel length
// r: kernel radius = K/2 (floor)
// s_mem: extern shared memory pointer (size: blockDim.x + 2*r floats)
__global__ void conv1d_shared_const(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    int N, int K, int r)
{
    extern __shared__ float s[]; // tile + halo
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int B = blockDim.x;

    // global index of the output this thread will compute
    int gx = bx * B + tx;

    // start index of tile in global coordinates (leftmost element that goes into s[0])
    // For tile covering outputs at gx = bx*B .. bx*B + B-1, we need inputs from
    // start = bx*B - r up to end = bx*B + B -1 + r
    int tileStart = bx * B - r;
    int tileWidth = B + 2 * r;

    // cooperative load of tileWidth elements into shared memory
    // each thread loads multiple elements strided by blockDim.x
    for (int i = tx; i < tileWidth; i += B) {
        int gidx = tileStart + i; // global input index corresponding to s[i]
        float v = 0.0f;
        if (gidx >= 0 && gidx < N) v = in[gidx];
        s[i] = v;
    }
    __syncthreads();

    if (gx >= N) return; // outside output range

    // compute convolution for output index gx
    // s index for gx corresponds to offset = (gx - tileStart)
    int sbase = gx - tileStart;
    float sum = 0.0f;

    // accumulate using kernel in constant memory
    // kernel index k runs 0..K-1, multiplies s[sbase - r + k]
    // Note: sbase - r = gx - tileStart - r = gx - (bx*B - r) - r = gx - bx*B
    // but simpler to index directly:
    int sIdx = sbase - r;
    for (int k = 0; k < K; ++k) {
        // s index is sIdx + k ; we are guaranteed s array size = tileWidth
        sum += s[sIdx + k] * d_kernel_const[k];
    }

    out[gx] = sum;
}

// CPU reference convolution (same semantics: zero padded "same")
void conv1d_reference(const float* in, float* out, int N, const float* kernel, int K) {
    int r = K / 2;
    for (int i = 0; i < N; ++i) {
        float s = 0.0f;
        for (int k = 0; k < K; ++k) {
            int idx = i - r + k;
            if (idx >= 0 && idx < N) s += in[idx] * kernel[k];
        }
        out[i] = s;
    }
}

int main(int argc, char** argv) {
    // parse args: input size and kernel size optional
    int N = 1 << 20; // default ~1,048,576
    int K = 31;      // default kernel size
    if (argc >= 2) N = atoi(argv[1]);
    if (argc >= 3) K = atoi(argv[2]);
    if (N < 1) { fprintf(stderr,"Invalid N\n"); return 1; }
    if (K < 1 || K >= KMAX) { fprintf(stderr,"K must be 1..%d\n", KMAX-1); return 1; }

    printf("1D conv: N=%d, K=%d\n", N, K);

    const int r = K/2;
    const int BLOCK = 256; // good balance for Tesla GPUs; tune experimentally
    const int grid = (N + BLOCK - 1) / BLOCK;

    // check shared memory requirement per block
    size_t shmem_bytes = (size_t)(BLOCK + 2 * r) * sizeof(float);
    cudaDeviceProp prop;
    int dev; CHECK_CUDA(cudaGetDevice(&dev));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s, sharedMemPerBlock=%zu bytes\n", prop.name, prop.sharedMemPerBlock);

    if (shmem_bytes > prop.sharedMemPerBlock) {
        fprintf(stderr, "Required shared memory %zu > device limit %zu. Reduce BLOCK or K.\n",
                shmem_bytes, (size_t)prop.sharedMemPerBlock);
        return 1;
    }

    // Allocate host pinned memory for input/output (good for streaming / H2D)
    float *h_in = nullptr, *h_out = nullptr;
    CHECK_CUDA(cudaHostAlloc((void**)&h_in, N * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc((void**)&h_out, N * sizeof(float), cudaHostAllocDefault));

    // fill input with some test data
    srand(123);
    for (int i = 0; i < N; ++i) h_in[i] = (float)(rand() & 0xFF) / 255.0f;

    // Host kernel array (copy to constant)
    float* h_kernel = (float*)malloc(K * sizeof(float));
    // example: simple box or gaussian-like weights; here simple normalized box
    for (int k = 0; k < K; ++k) h_kernel[k] = 1.0f / K;

    // copy kernel to constant memory
    CHECK_CUDA(cudaMemcpyToSymbol(d_kernel_const, h_kernel, K * sizeof(float)));

    // device buffers
    float *d_in = nullptr, *d_out = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_in, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_out, N * sizeof(float)));

    // copy input to device
    CHECK_CUDA(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    // warmup + timed launch
    CHECK_CUDA(cudaDeviceSynchronize());
    const int iterations = 3;

    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));
    CHECK_CUDA(cudaEventRecord(t0));

    for (int it = 0; it < iterations; ++it) {
        conv1d_shared_const<<<grid, BLOCK, shmem_bytes>>>(d_in, d_out, N, K, r);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaEventRecord(t1));
    CHECK_CUDA(cudaEventSynchronize(t1));
    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
    float avg_ms = ms / iterations;
    double bytesMoved = (double)N * sizeof(float) * 2 + (double)K * sizeof(float); // rough
    printf("Avg kernel time: %.3f ms (grid=%d,B=%d,shmem=%.1f KB)\n",
           avg_ms, grid, BLOCK, shmem_bytes / 1024.0);

    // copy result back
    CHECK_CUDA(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    // verify correctness on a small sample
    const int CHECKN = std::min(N, 512);
    float* ref = (float*)malloc(CHECKN * sizeof(float));
    conv1d_reference(h_in, ref, CHECKN, h_kernel, K);
    bool ok = true;
    for (int i = 0; i < CHECKN; ++i) {
        float a = h_out[i];
        float b = ref[i];
        if (fabsf(a - b) > 1e-3f) { ok = false; break; }
    }
    printf("Validation (first %d): %s\n", CHECKN, ok ? "PASS" : "FAIL");

    // cleanup
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFreeHost(h_in));
    CHECK_CUDA(cudaFreeHost(h_out));
    free(h_kernel);
    free(ref);
    CHECK_CUDA(cudaEventDestroy(t0));
    CHECK_CUDA(cudaEventDestroy(t1));

    return 0;
}
