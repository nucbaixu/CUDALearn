#include <iostream>
#include <cstdlib>

// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

// Define the files that are to be save and the reference images for validation
const char *sampleNamePitchLinear   = "simplePitchLinearTexture";

#define NUM_REPS 100  // number of repetitions performed
#define TILE_DIM 16   // tile/block size

/*
 * Shifts matrix elements using pitch linear array
 * @param odata  output data in global memory
 */
__global__ void shiftPitchLinear(float *odata,
                                 int pitch,
                                 int width,
                                 int height,
                                 int shiftX,
                                 int shiftY,
                                 cudaTextureObject_t texRefPL)
{
    unsigned int xid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int yid = blockDim.y * blockIdx.y + threadIdx.y;

    odata[yid*pitch + xid] = tex2D<float>(texRefPL,
                                          float(xid+shiftX)/(float)width,
                                          float(yid+shiftY)/(float)height );
}

__global__ void shiftArray(float *odata,
                           int pitch,
                           int width,
                           int height,
                           int shiftX,
                           int shiftY,
                           cudaTextureObject_t texRefArray)
{
    unsigned int xid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yid = blockIdx.y * blockDim.y + threadIdx.y;

    odata[yid * pitch + xid] = tex2D<float>(texRefArray,
                                            float (xid + shiftX) / (float) width,
                                            float (yid + shiftY) / (float) height);
}

void Run_simplePitchLinearTexture();

int main()
{
    printf("%s starting...\n\n", sampleNamePitchLinear);

    Run_simplePitchLinearTexture();

    printf("%s completed, returned \n",
           sampleNamePitchLinear);
}

void Run_simplePitchLinearTexture()
{
    // Set array size
    const int nx = 2048;
    const int ny = 2048;

    // Setup shifts applied to x and y data
    const int x_shift = 5;
    const int y_shift = 7;

    // Setup execution configuration parameters
    dim3 dimGrid(nx / TILE_DIM, ny / TILE_DIM), dimBlock(TILE_DIM, TILE_DIM);

    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    // Host allocation and initialization
    auto * h_idata = (float *) malloc(sizeof(float) * nx * ny);
    auto * h_odata = (float *) malloc(sizeof(float) * nx * ny);
    auto * gold = (float *) malloc(sizeof(float) * nx * ny);

    for (int i = 0; i < nx * ny; ++i){
        h_idata[i] = (float) i;
    }

    // Device memory allocation Pitch linear input data
    float *d_idataPL;
    size_t d_pitchBytes;

    checkCudaErrors(cudaMallocPitch((void **) &d_idataPL,
                                    &d_pitchBytes,
                                    nx * sizeof(float),
                                    ny));

    std::cout<<"d_pitchBytes:"<<d_pitchBytes<<std::endl;

    // Array input data
    cudaArray *d_idataArray;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    checkCudaErrors(cudaMallocArray(&d_idataArray, &channelDesc, nx, ny));

    // Pitch linear output data
    //cudaError_t cudaMallocPitch( void** devPtr，size_t* pitch，size_t widthInBytes，size_t height )
    // 向设备分配至少widthInBytes*height字节的线性存储器，并以*devPtr的形式返回指向所分配存储器的指针。
    // 该函数可以填充所分配的存储器，以确保在地址从一行更新到另一行时，给定行的对应指针依然满足对齐要求。
    // cudaMallocPitch()以*pitch的形式返回间距，即所分配存储器的宽度，以字节为单位
    //如果给定一个T类型数组元素的行和列，可按如下方法计算地址
    //T* pElement = (T*)((char*)BaseAddress + Row * pitch) + Column;
    float *d_odata;
    checkCudaErrors(cudaMallocPitch((void **) &d_odata,
                                    &d_pitchBytes,
                                    nx * sizeof(float),
                                    ny));


    // Copy host data to device
    // Pitch linear
    size_t h_pitchBytes = nx * sizeof(float);

    checkCudaErrors(cudaMemcpy2D(d_idataPL,
                                 d_pitchBytes,
                                 h_idata,
                                 h_pitchBytes,
                                 nx * sizeof(float),
                                 ny,
                                 cudaMemcpyHostToDevice));

    // Array
    //cudaMemcpyToArray is desperate so use cudaMemcpy2DToArray instead
    checkCudaErrors(cudaMemcpy2DToArray(d_idataArray,
                        0,
                        0,
                        h_idata,
                        nx*sizeof(float) , nx*sizeof(float ), ny, cudaMemcpyHostToDevice));

    /*
    checkCudaErrors(cudaMemcpyToArray(d_idataArray,
                                      0,
                                      0,
                                      h_idata,
                                      nx * ny * sizeof(float),
                                      cudaMemcpyHostToDevice));
    */

    cudaTextureObject_t         texRefPL;
    cudaTextureObject_t         texRefArray;
    cudaResourceDesc            texRes = {};
    memset(&texRes,0,sizeof(cudaResourceDesc));
    texRes.resType          = cudaResourceTypePitch2D;
    texRes.res.pitch2D.devPtr	= d_idataPL;
    texRes.res.pitch2D.desc     = channelDesc;
    texRes.res.pitch2D.width	= nx;
    texRes.res.pitch2D.height   = ny;
    texRes.res.pitch2D.pitchInBytes = h_pitchBytes;

    cudaTextureDesc         texDescr = {};
    memset(&texDescr,0,sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = true;
    texDescr.filterMode       = cudaFilterModePoint;
    texDescr.addressMode[0] = cudaAddressModeWrap;
    texDescr.addressMode[1] = cudaAddressModeWrap;
    texDescr.readMode = cudaReadModeElementType;


    checkCudaErrors(cudaCreateTextureObject(&texRefPL, &texRes, &texDescr, nullptr));
    memset(&texRes,0,sizeof(cudaResourceDesc));
    memset(&texDescr,0,sizeof(cudaTextureDesc));


    texRes.resType            = cudaResourceTypeArray;
    texRes.res.array.array    = d_idataArray;
    texDescr.normalizedCoords = true;
    texDescr.filterMode       = cudaFilterModePoint;
    texDescr.addressMode[0] = cudaAddressModeWrap;
    texDescr.addressMode[1] = cudaAddressModeWrap;
    texDescr.readMode = cudaReadModeElementType;
    checkCudaErrors(cudaCreateTextureObject(&texRefArray, &texRes, &texDescr, nullptr));

    // Reference calculation
    for (int j = 0; j < ny; ++j)
    {
        int jshift = (j + y_shift) % ny;

        for (int i = 0; i < nx; ++i)
        {
            int ishift = (i + x_shift) % nx;
            gold[j * nx + i] = h_idata[jshift * nx + ishift];
        }
    }

    // Run ShiftPitchLinear kernel
    checkCudaErrors(cudaMemset2D(d_odata,
                                 d_pitchBytes,
                                 0,
                                 nx * sizeof(float),
                                 ny));

    checkCudaErrors(cudaEventRecord(start, nullptr));

    for (int i = 0; i < NUM_REPS; ++i)
    {
        shiftPitchLinear<<<dimGrid, dimBlock>>>
        (d_odata,
         (int)(d_pitchBytes / sizeof(float)),
         nx,
         ny,
         x_shift,
         y_shift, texRefPL);
    }

    checkCudaErrors(cudaEventRecord(stop, nullptr));
    checkCudaErrors(cudaEventSynchronize(stop));
    float timePL;
    checkCudaErrors(cudaEventElapsedTime(&timePL, start, stop));

    // Check results
    checkCudaErrors(cudaMemcpy2D(h_odata,
                                 h_pitchBytes,
                                 d_odata,
                                 d_pitchBytes,
                                 nx * sizeof(float),
                                 ny,
                                 cudaMemcpyDeviceToHost));

    compareData(gold, h_odata, nx*ny, 0.0f, 0.15f);

    // Run ShiftArray kernel
    checkCudaErrors(cudaMemset2D(d_odata,
                                 d_pitchBytes,
                                 0,
                                 nx * sizeof(float),
                                 ny));
    checkCudaErrors(cudaEventRecord(start, nullptr));

    for (int i = 0; i < NUM_REPS; ++i)
    {
        shiftArray<<<dimGrid, dimBlock>>>
        (d_odata,
         (int)(d_pitchBytes / sizeof(float)),
         nx,
         ny,
         x_shift,
         y_shift, texRefArray);
    }

    checkCudaErrors(cudaEventRecord(stop, nullptr));
    checkCudaErrors(cudaEventSynchronize(stop));
    float timeArray;
    checkCudaErrors(cudaEventElapsedTime(&timeArray, start, stop));

    // Check results
    checkCudaErrors(cudaMemcpy2D(h_odata,
                                 h_pitchBytes,
                                 d_odata,
                                 d_pitchBytes,
                                 nx * sizeof(float),
                                 ny,
                                 cudaMemcpyDeviceToHost));
    compareData(gold, h_odata, nx*ny, 0.0f, 0.15f);

    float bandwidthPL =
            2.f * 1000.f * nx * ny * sizeof(float) /
            (1.e+9f) / (timePL / NUM_REPS);
    float bandwidthArray =
            2.f * 1000.f * nx * ny * sizeof(float) /
            (1.e+9f) / (timeArray / NUM_REPS);

    printf("\nBandwidth (GB/s) for pitch linear: %.2e; for array: %.2e\n",
           bandwidthPL, bandwidthArray);

    float fetchRatePL =
            nx * ny / 1.e+6f / (timePL / (1000.0f * NUM_REPS));
    float fetchRateArray =
            nx * ny / 1.e+6f / (timeArray / (1000.0f * NUM_REPS));

    printf("\nTexture fetch rate (Mpix/s) for pitch linear: "
           "%.2e; for array: %.2e\n\n",
           fetchRatePL, fetchRateArray);

    // Cleanup
    free(h_idata);
    free(h_odata);
    free(gold);

    checkCudaErrors(cudaDestroyTextureObject(texRefPL));
    checkCudaErrors(cudaDestroyTextureObject(texRefArray));
    checkCudaErrors(cudaFree(d_idataPL));
    checkCudaErrors(cudaFreeArray(d_idataArray));
    checkCudaErrors(cudaFree(d_odata));

    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));





















}