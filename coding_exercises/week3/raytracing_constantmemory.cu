#include <iostream>
#include <cstdlib>     // for rand()
#include <ctime>       // for seeding rand()
#include <cuda_runtime.h>

using namespace std;

#define INF 2e10f
#define SPHERES 20

struct sphere {
    float r, g, b; // color properties
    float x, y, z; // position in 3D space
    float radius;  // radius of the sphere

    // check for the spheres getting hit by the ray originating from pixel at (ox, oy)
    __device__ float intersect(float ox, float oy, float* n) {

        float dx = x - ox;
        float dy = y - oy;
        // ray-hit condition
        if(dx*dx + dy*dy < radius*radius) {
            float dz = sqrtf(radius*radius - dx*dx - dy*dy);
            *n = dz / r;
            return z - dz;
        }
        return INF;
    }
};

__constant__ sphere s[SPHERES]; // **Note**: Added qualifier for constant memoru
#define N 16

__global__ void render(float* dev, sphere* s) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * gridDim.x * blockDim.x;
    // do operations of calculating the pixel colors for valid x and y coordinates
    if(x<N && y<N){
        float fx = x - N/2; // shift the x, y so that z-axis runs through center of image
        float fy = y - N/2;
        float r=0, g=0, b=0;
        float maxz = INF;
        int hit = -1;
        // iterate over all the spheres
        for(int i = 0; i< SPHERES;i++) {
            float n;
            float t = s[i].intersect(fx, fy, &n); // calculate the hit
            if(t<maxz) {
                float fscale = n; // calculate the amount of color for the pixel due to angle of hitting
                maxz = t;
                r = s[i].r * fscale;
                g = s[i].g * fscale;
                b = s[i].b * fscale;
                hit = i;
            }
        }

        if(hit>=0){
            // set the color for that pixel hitting the spheres
            dev[offset*3 + 0] = float(r * 255);
            dev[offset*3 + 1] = float(g * 255);
            dev[offset*3 + 2] = float(b * 255);
        } else {
            // dark background
            dev[offset*3 + 0] = float(0);
            dev[offset*3 + 1] = float(0);
            dev[offset*3 + 2] = float(0);
        }

    }
}

int main() {

    float *dev_image; // define variable for device
    float image_map[N*N*3];  // define image patch
    cudaMalloc(&dev_image, N*N*3*sizeof(float)); // allocate global memory for device variable

    // initialize the 3D spheres with its properties
    sphere *temp_s = (sphere*)malloc(SPHERES*sizeof(sphere));
    for(int i=0;i<SPHERES;i++){
        temp_s[i].r = (rand() % 200 - 100) / 20.0f;   // -5 .. +5
        temp_s[i].g = (rand() % 200 - 100) / 20.0f;   // -5 .. +5
        temp_s[i].b = - (rand() % 10 + 3);            // -3 .. -12
        temp_s[i].x = rand() % 256;
        temp_s[i].y = rand() % 256;
        temp_s[i].z = rand() % 256;
        temp_s[i].radius = (rand() % 30) / 10.0f + 0.5f;   // 0.5 .. 3.5
    }


    // **Note**: Use cudaMemcpyToSymbol for constant memory.
    // It requires no declaration og HostToDevice flag because it
    // directly copies values to constant memory
    cudaMemcpyToSymbol(s, temp_s, SPHERES*sizeof(sphere));

    free(temp_s); // free the temporary host memory

    // initialize the image_map
    for(int i =0;i<N*N*3;i++){
        image_map[i] = 0;
    }

    dim3 block(N,N); // define the block dimensions
    dim3 grid((N+15)/16, (N+15)/16); // define the required grid dimension

    // call the kernel
    render<<<grid, block>>>(dev_image, s);

    // collect the output result from device to host
    cudaMemcpy(image_map, dev_image, N*N*3*sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0;i<N*N*3;i++){
        cout<<"image pixels at i= "<<i<<" : "<<image_map[i]<<endl;
    }

    cudaFree(dev_image);
    cudaFree(s);
    return 0;
}