#include <iostream>
#include <cstdlib>     // for rand()
#include <ctime>       // for seeding rand()
#include <cuda_runtime.h>

using namespace std;

#define INF 2e10f
#define SPHERES 20
#define N 16

struct sphere {
    float r, g, b; // color properties
    float x, y, z; // position in 3D space
    float radius;  // radius of the sphere

    // check for the spheres getting hit by the ray originating from pixel at (ox, oy)
    __device__ float intersect(float ox, float oy, float* n_out) {

        float dx = x - ox;
        float dy = y - oy;
        float rr = radius*radius;
        // ray-hit condition
        if(dx*dx + dy*dy < rr) {
            float dz = sqrtf(rr - dx*dx - dy*dy);
            *n_out = dz / radius;
            return z - dz;
        }
        return INF;
    }
};

#define CHECK_CUDA(call) do {                                \
    cudaError_t e = (call);                                  \
    if (e != cudaSuccess) {                                  \
        std::cerr << "CUDA error: " << cudaGetErrorString(e) \
                  << " at " << __FILE__ << ":" << __LINE__   \
                  << std::endl;                              \
        exit(1);                                             \
    }                                                        \
} while(0)


// Kernel: each thread computes one pixel
__global__ void render(float* dev, sphere* spheres, int num_spheres) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    //int offset = x + y * gridDim.x * blockDim.x;
    // do operations of calculating the pixel colors for valid x and y coordinates
    if(x<N && y<N){
        // map pixel to xy-plane coords (centered)
        // shift the x, y so that z-axis runs through center of image
        float fx = (float)x - (float)N / 2.0f;
        float fy = (float)y - (float)N / 2.0f;
        int offset = (y * N + x) * 3;


        float maxz = INF;
        int hit = -1;
        float bestN = 0.0f;
        // iterate over all the spheres
        for(int i = 0; i< num_spheres;i++) {
            float n=0.0f;
            float t = spheres[i].intersect(fx, fy, &n); // calculate the hit
            if(t<maxz) {
                bestN = n; // calculate the amount of color for the pixel due to angle of hitting
                maxz = t;
                hit = i;
            }
        }

        if(hit>=0){
            // set the color for that pixel hitting the spheres
            float rr = spheres[hit].r * bestN;
            float gg = spheres[hit].g * bestN;
            float bb = spheres[hit].b * bestN;
            dev[offset*3 + 0] = float(rr * 255);
            dev[offset*3 + 1] = float(gg * 255);
            dev[offset*3 + 2] = float(bb * 255);
        } else {
            // dark background
            dev[offset*3 + 0] = float(0);
            dev[offset*3 + 1] = float(0);
            dev[offset*3 + 2] = float(0);
        }

    }
}

int main() {

    srand((unsigned)time(NULL));

    cudaEvent_t start, stop; // define the variable for cuda event
    cudaEventCreate(&start); // create event for start
    cudaEventCreate(&stop);  // create event for stop
    cudaEventRecord(start, 0);  // start recording the event

    float *dev_image = nullptr; // define variable for device
    float *image_map = new float[N*N*3];  // define image patch
    cudaMalloc(&dev_image, N*N*3*sizeof(float)); // allocate global memory for device variable

    // initialize the 3D spheres with its properties
    sphere *temp_s = (sphere*)malloc(SPHERES*sizeof(sphere));
    for (int i = 0; i < SPHERES; ++i) {
        // colors in [0,1]
        temp_s[i].r = (rand() % 256) / 255.0f;
        temp_s[i].g = (rand() % 256) / 255.0f;
        temp_s[i].b = (rand() % 256) / 255.0f;
        // x,y in roughly -N/2 .. +N/2 so they can intersect image plane
        temp_s[i].x = (rand() % (N * 2)) - (float)N;
        temp_s[i].y = (rand() % (N * 2)) - (float)N;
        // z behind the image plane (negative), between -3 and -12
        temp_s[i].z = - (3.0f + (rand() % 10));
        // radius reasonable
        temp_s[i].radius = 0.5f + (rand() % 30) / 10.0f;
    }

    sphere* d_spheres = nullptr;
    // allocate global memory for shperes variable on device
    CHECK_CUDA(cudaMalloc(&d_spheres, sizeof(sphere) * SPHERES));

    // transfer the data from host to device
    cudaMemcpy(d_spheres, temp_s, SPHERES*sizeof(sphere), cudaMemcpyHostToDevice);

    free(temp_s); // free the temporary host memory

    // initialize the image_map
    for(int i =0;i<N*N*3;i++){
        image_map[i] = 0.0f;
    }

    dim3 block(N,N); // define the block dimensions
    dim3 grid((N+15)/16, (N+15)/16); // define the required grid dimension

    // call the kernel
    render<<<grid, block>>>(dev_image, d_spheres, SPHERES);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // collect the output result from device to host
    cudaMemcpy(image_map, dev_image, N*N*3*sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);  // record the timestamp of completion of event
    cudaEventSynchronize(stop); // ensure no other event starts before the event to be recorded is completed

    float time_elapsed;
    cudaEventElapsedTime(&time_elapsed, start, stop);
    cout<<"time (in ms) elapsed for the ray tracing event: "<<time_elapsed<<endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    for (int i = 0; i < 200 && i < N*N; ++i) {
        int o = i * 3;
        cout << i << ": " << image_map[o+0] << " " << image_map[o+1] << " " << image_map[o+2] << endl;
    }

    delete[] image_map;

    CHECK_CUDA(cudaFree(dev_image));
    CHECK_CUDA(cudaFree(d_spheres));
    return 0;
}