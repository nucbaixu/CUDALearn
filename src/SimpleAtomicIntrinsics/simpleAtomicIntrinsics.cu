/* A simple program demonstrating trivial use of global memory atomic
 * device functions (atomic*() functions).
 */

// includes, system
#include <cstdlib>
#include <cstdio>
#include <string>
#include <cmath>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

// Includes, kernels
#include "simpleAtomicIntrinsics_kernel.cuh"

const char *sampleName = "simpleAtomicIntrinsics";

////////////////////////////////////////////////////////////////////////////////
// Auto-Verification Code
bool testResult = true;

////////////////////////////////////////////////////////////////////////////////
// Declaration, forward
void runTest(int argc, char **argv);

extern "C" bool computeGold(int *gpuData, int len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    printf("%s starting...\n", sampleName);

    runTest(argc, argv);

    printf("%s completed, returned %s\n",
           sampleName,
           testResult ? "OK" : "ERROR!");
    exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char **argv)
{
    cudaStream_t stream;
    // This will pick the best possible CUDA capable device
    findCudaDevice(argc, (const char **)argv);

    StopWatchInterface *timer;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    unsigned int numThreads = 256;
    unsigned int numBlocks = 64;
    unsigned int numData = 11;
    unsigned int memSize = sizeof(int) * numData;

    //allocate mem for the result on host side
    int *hOData = (int *) malloc(memSize);

    //initialize the memory
    for (unsigned int i = 0; i < numData; i++)
        hOData[i] = 0;

    //To make the AND and XOR tests generate something other than 0...
    hOData[8] = hOData[10] = 0xff;

    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    // allocate device memory for result
    int *dOData;
    checkCudaErrors(cudaMalloc((void **) &dOData, memSize));
    // copy host memory to device to initialize to zero
    checkCudaErrors(cudaMemcpyAsync(dOData, hOData, memSize,
                                    cudaMemcpyHostToDevice, stream));

    // execute the kernel
    testKernel<<<numBlocks, numThreads, 0, stream>>>(dOData);
    getLastCudaError("Kernel execution failed");

    //Copy result from device to host
    checkCudaErrors(cudaMemcpyAsync(hOData, dOData, memSize,
                                    cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);

    // Compute reference solution
    testResult = computeGold(hOData, numThreads * numBlocks);

    // Cleanup memory
    free(hOData);
    checkCudaErrors(cudaFree(dOData));
}
