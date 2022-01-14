#include <iostream>
#include <cstdlib>

// Includes CUDA
#include <cuda_runtime.h>


// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

// Define the files that are to be save and the reference images for validation
const char *sampleName    = "simpleTexture";

////////////////////////////////////////////////////////////////////////////////
// Constants
const float angle = 0.5f;        // angle to rotate image by (in radians)


__global__ void transformKernel(float *outputData,
                                int i_width,
                                int i_height,
                                float theta,
                                cudaTextureObject_t tex)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    float u = (float)x - (float)(i_width) / 2.0f;
    float v = (float)y - (float)(i_height) / 2.0f;

    float tu = u * cosf(theta) - v * sinf(theta);
    float tv = v * cosf(theta) + u * sinf(theta);

    tu /= (float)i_width;
    tv /= (float)i_height;

    outputData[y * i_width + x] = tex2D<float>(tex, tu + 0.5f, tv + 0.5f);
}


void runTest();

int main()
{
    std::cout << sampleName<<" starting ..." << std::endl;
    runTest();
    std::cout << sampleName<<" completed ..." << std::endl;
    return 0;
}


void runTest()
{
    //load image from disk
    float * hData = nullptr;
    unsigned int width,height;

    const char *image_path = "../../../testdata/lena_bw.pgm";

    sdkLoadPGM(image_path, &hData, &width, &height);
    unsigned int size  = width * height * sizeof(float);

    std::cout << "Load " << image_path << " " << width << "*" << height << "pixels " << std::endl;

    //allocate device memory for result
    float *dData = nullptr;
    checkCudaErrors(cudaMalloc((void**)&dData,size));

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);

    cudaArray *pCudaArray;
    checkCudaErrors(cudaMallocArray(&pCudaArray, &channelDesc, width, height));
    checkCudaErrors(cudaMemcpyToArray(pCudaArray, 0, 0, hData, size, cudaMemcpyHostToDevice));

    cudaTextureObject_t tex;
    cudaResourceDesc texRes = {};
    memset(&texRes,0,sizeof(cudaResourceDesc));

    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = pCudaArray;

    cudaTextureDesc textureDesc={};
    memset(&textureDesc,0,sizeof(cudaTextureDesc));

    textureDesc.normalizedCoords = true;
    textureDesc.mipmapFilterMode = cudaFilterModeLinear;
    textureDesc.addressMode[0] = cudaAddressModeWrap;
    textureDesc.addressMode[1] = cudaAddressModeWrap;
    textureDesc.readMode = cudaReadModeElementType;

    checkCudaErrors(cudaCreateTextureObject(&tex,&texRes,&textureDesc,nullptr));

    dim3 dimBlock(8,8,1);
    dim3 dimGrid(width/dimBlock.x ,height/dimBlock.y ,1);

    transformKernel<<<dimGrid,dimBlock,0>>>(dData, (int)width, (int)height, angle, tex);

    checkCudaErrors(cudaDeviceSynchronize()) ;

    StopWatchInterface *timer = nullptr;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    // Execute the kernel
    transformKernel<<<dimGrid, dimBlock, 0>>>(dData, (int)width, (int)height, angle, tex);

    // Check if kernel execution generated an error
    getLastCudaError("Kernel execution failed");
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer);

    // Allocate mem for the result on host side
    auto * hOutputData = (float *) malloc(size);
    // copy result from device to host
    checkCudaErrors(cudaMemcpy(hOutputData,
                               dData,
                               size,
                               cudaMemcpyDeviceToHost));

    // Write result to file
    char outputFilename[1024];
    strcpy_s(outputFilename, image_path);
    strcpy(outputFilename + strlen(image_path) - 4, "_out.pgm");
    sdkSavePGM(outputFilename, hOutputData, width, height);
    printf("Write '%s'\n", outputFilename);

    checkCudaErrors(cudaDestroyTextureObject(tex));
    checkCudaErrors(cudaFree(dData));
    checkCudaErrors(cudaFreeArray(pCudaArray));
}
