
// Matrix multiplication: C = A * B.
// Host code.
//
// This sample implements matrix multiplication as described in Chapter 3
// of the programming guide and uses the CUBLAS library to demonstrate
// the best performance.

// SOME PRECAUTIONS:
// IF WE WANT TO CALCULATE ROW-MAJOR MATRIX MULTIPLY C = A * B,
// WE JUST NEED CALL CUBLAS API IN A REVERSE ORDER: cublasSegemm(B, A)!
// The reason is explained as follows:

// CUBLAS library uses column-major storage, but C/C++ use row-major storage.
// When passing the matrix pointer to CUBLAS, the memory layout alters from
// row-major to column-major, which is equivalent to an implicit transpose.
// Utilities and system includes


#include <helper_string.h>  // helper for shared functions common to CUDA Samples

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>

#ifndef min
#define min(a,b) ((a < b) ? a : b)
#endif
#ifndef max
#define max(a,b) ((a > b) ? a : b)
#endif

typedef struct _matrixSize      // Optional Command-line multiplier for matrix sizes
{
    unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
} sMatrixSize;

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set matrix multiply on CPU
//! C = A * B
//! @param C          reference data, computed but preallocated
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param hA         height of matrix A
//! @param wB         width of matrix B
////////////////////////////////////////////////////////////////////////////////
void matrixMulCPU(float *C, const float *A, const float *B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    for (unsigned int i = 0 ; i < hA; i++)
    {
        for(unsigned int j = 0; j < wB; j++)
        {
            float  f_sum = 0;

            for(int k = 0; k < wA;k++)
            {
                f_sum += A[i * wA + k] * B[k * wB + j];
            }

            C[i * wB + j] = f_sum;
        }
    }
}

// Allocates a matrix with random float entries.
void randomInit(float *data,  unsigned int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = float( rand())  / (float)RAND_MAX;
}

void printDiff(float *data1, float *data2, int width, int height, int iListLength, float fListTol)
{
    printf("Listing first %d Differences > %.6f...\n", iListLength, fListTol);
    int i,j,k;
    int error_count=0;

    for (j = 0; j < height; j++)
    {
        if (error_count < iListLength)
        {
            printf("\n  Row %d:\n", j);
        }

        for (i = 0; i < width; i++)
        {
            k = j * width + i;
            float fDiff = fabs(data1[k] - data2[k]);

            if (fDiff > fListTol)
            {
                if (error_count < iListLength)
                {
                    printf("    Loc(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j, data1[k], data2[k], fDiff);
                }

                error_count++;
            }
        }
    }

    printf(" \n  Total Errors = %d\n", error_count);
}

void initializeCUDA(int argc, char **argv, int &devID, int &iSizeMultiple, sMatrixSize &matrix_size)
{
    // By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
    cudaError_t error;
    devID = 0;

    devID = findCudaDevice(argc, (const char **)argv);

    if (checkCmdLineFlag(argc, (const char **)argv, "sizemult"))
    {
        iSizeMultiple = getCmdLineArgumentInt(argc, (const char **)argv, "sizemult");
    }

    iSizeMultiple = min(iSizeMultiple, 10);
    iSizeMultiple = max(iSizeMultiple, 1);

    cudaDeviceProp deviceProp{};

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);

    int block_size = 32;

    matrix_size.uiWA = 3 * block_size * iSizeMultiple;
    matrix_size.uiHA = 4 * block_size * iSizeMultiple;
    matrix_size.uiWB = 2 * block_size * iSizeMultiple;
    matrix_size.uiHB = 3 * block_size * iSizeMultiple;
    matrix_size.uiWC = 2 * block_size * iSizeMultiple;
    matrix_size.uiHC = 4 * block_size * iSizeMultiple;

    printf("MatrixA(%u,%u), MatrixB(%u,%u), MatrixC(%u,%u)\n",
           matrix_size.uiHA, matrix_size.uiWA,
           matrix_size.uiHB, matrix_size.uiWB,
           matrix_size.uiHC, matrix_size.uiWC);

    if( matrix_size.uiWA != matrix_size.uiHB ||
    matrix_size.uiHA != matrix_size.uiHC ||
    matrix_size.uiWB != matrix_size.uiWC)
    {
        printf("ERROR: Matrix sizes do not match!\n");
        exit(-1);
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test matrix multiply using CUBLAS
////////////////////////////////////////////////////////////////////////////////
int matrixMultiply_V1(int argc, char **argv, int devID, sMatrixSize &matrix_size)
{
    cudaDeviceProp deviceProp{};

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

    int block_size = 32;

    // set seed for rand()
    srand(2006);

    // allocate host memory for matrices A and B
    unsigned int size_A = matrix_size.uiWA * matrix_size.uiHA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    auto *h_A = (float *)malloc(mem_size_A);
    unsigned int size_B = matrix_size.uiWB * matrix_size.uiHB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    auto *h_B = (float *)malloc(mem_size_B);

    // set seed for rand()
    srand(2006);

    // initialize host memory
    randomInit(h_A, size_A);
    randomInit(h_B, size_B);

    // allocate device memory
    float *d_A, *d_B, *d_C;
    unsigned int size_C = matrix_size.uiWC * matrix_size.uiHC;
    unsigned int mem_size_C = sizeof(float) * size_C;

    // allocate host memory for the result
    auto *h_C      = (float *) malloc(mem_size_C);
    auto *h_CUBLAS = (float *) malloc(mem_size_C);

    checkCudaErrors(cudaMalloc((void **) &d_A, mem_size_A));
    checkCudaErrors(cudaMalloc((void **) &d_B, mem_size_B));
    checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &d_C, mem_size_C));

    // setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(matrix_size.uiWC / threads.x, matrix_size.uiHC / threads.y);

    // create and start timer
    printf("Computing result using CUBLAS...");

    int nIter = 30;

    //CUBLAS Version 2.0
    {
        const float alpha = 1.0f;
        const float beta = 0.0;

        cublasHandle_t handle;
        cudaEvent_t start,stop;

        checkCudaErrors(cublasCreate(&handle));

        //preform warmup operation with cublas
        //cublasSgemm是CUDA的cublas库的矩阵相乘函数
        //https://blog.csdn.net/u011197534/article/details/78378536
        //https://blog.csdn.net/HaoBBNuanMM/article/details/103054357
        //当我们选择CUBLAS_OP_N时表示不转置，按列优先存储；当我们选择CUBLAS_OP_T时表示需要转置，按行优先存储

        checkCudaErrors(cublasSgemm(handle
                                    , CUBLAS_OP_N
                                    , CUBLAS_OP_N
                                    , matrix_size.uiWC
                                    , matrix_size.uiHC
                                    , matrix_size.uiWA
                                    , &alpha
                                    , d_B
                                    , matrix_size.uiWB
                                    , d_A
                                    , matrix_size.uiWA
                                    , &beta
                                    , d_C
                                    , matrix_size.uiWC));

        // Allocate CUDA events that we'll use for timing
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));

        // Record the start event
        checkCudaErrors(cudaEventRecord(start, nullptr));

        for (int j = 0; j < nIter; j++)
        {
            //note cublas is column primary!
            //need to transpose the order
            checkCudaErrors(cublasSgemm(handle
                                        , CUBLAS_OP_N
                                        , CUBLAS_OP_N
                                        , matrix_size.uiWC
                                        , matrix_size.uiHC
                                        , matrix_size.uiWA
                                        , &alpha
                                        , d_B
                                        , matrix_size.uiWB
                                        , d_A
                                        , matrix_size.uiWA
                                        , &beta
                                        , d_C
                                        , matrix_size.uiWC));
        }

        printf("done.\n");

        // Record the stop event
        checkCudaErrors(cudaEventRecord(stop, nullptr));

        // Wait for the stop event to complete
        checkCudaErrors(cudaEventSynchronize(stop));

        float msecTotal = 0.0f;
        checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

        // Compute and print the performance
        float msecPerMatrixMul = msecTotal / nIter;
        double flopsPerMatrixMul = 2.0 * (double)matrix_size.uiHC * (double)matrix_size.uiWC * (double)matrix_size.uiHB;
        double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
        printf(
                "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
                gigaFlops,
                msecPerMatrixMul,
                flopsPerMatrixMul);

        // copy result from device to host
        checkCudaErrors(cudaMemcpy(h_CUBLAS, d_C, mem_size_C, cudaMemcpyDeviceToHost));

        // Destroy the handle
        checkCudaErrors(cublasDestroy(handle));

    }

    // compute reference solution
    printf("Computing result using host CPU...");
    float *reference = (float *)malloc(mem_size_C);
    matrixMulCPU(reference, h_A, h_B, matrix_size.uiHA, matrix_size.uiWA, matrix_size.uiWB);
    printf("done.\n");

    // check result (CUBLAS)
    bool resCUBLAS = sdkCompareL2fe(reference, h_CUBLAS, size_C, 1.0e-6f);

    if (resCUBLAS != true)
    {
        printDiff(reference, h_CUBLAS, matrix_size.uiWC, matrix_size.uiHC, 100, 1.0e-5f);
    }

    printf("Comparing CUBLAS Matrix Multiply with CPU results: %s\n", (true == resCUBLAS) ? "PASS" : "FAIL");

    printf("\nNOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n");

    // clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(reference);
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

    if (resCUBLAS == true)
    {
        return EXIT_SUCCESS;    // return value = 1
    }
    else
    {
        return EXIT_FAILURE;     // return value = 0
    }

}

template<class T>
void transpose(T* a, T* b,int rows,int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            b[j*rows+i] = a[i*cols+j];
            //b[j][i] = a[i][j];
        }
    }
}

int matrixMultiply_V2(int argc, char **argv, int devID, sMatrixSize &matrix_size)
{
    cudaDeviceProp deviceProp{};

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

    int block_size = 32;

    // set seed for rand()
    srand(2006);

    // allocate host memory for matrices A and B
    unsigned int size_A = matrix_size.uiWA * matrix_size.uiHA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    auto *h_A = (float *)malloc(mem_size_A);
    unsigned int size_B = matrix_size.uiWB * matrix_size.uiHB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    auto *h_B = (float *)malloc(mem_size_B);

    // set seed for rand()
    srand(2006);

    // initialize host memory
    randomInit(h_A, size_A);
    randomInit(h_B, size_B);

    // allocate device memory
    float *d_A, *d_B, *d_C;
    unsigned int size_C = matrix_size.uiWC * matrix_size.uiHC;
    unsigned int mem_size_C = sizeof(float) * size_C;

    // allocate host memory for the result
    auto *h_C      = (float *) malloc(mem_size_C);
    auto *h_CUBLAS = (float *) malloc(mem_size_C);
    auto *h_CUBLAS_T = (float *) malloc(mem_size_C);

    checkCudaErrors(cudaMalloc((void **) &d_A, mem_size_A));
    checkCudaErrors(cudaMalloc((void **) &d_B, mem_size_B));
    checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &d_C, mem_size_C));

    // setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(matrix_size.uiWC / threads.x, matrix_size.uiHC / threads.y);

    // create and start timer
    printf("Computing result using CUBLAS...");

    int nIter = 30;

    //CUBLAS Version 2.0
    {
        const float alpha = 1.0f;
        const float beta = 0.0;

        cublasHandle_t handle;
        cudaEvent_t start,stop;

        checkCudaErrors(cublasCreate(&handle));

        //preform warmup operation with cublas
        //cublasSgemm是CUDA的cublas库的矩阵相乘函数
        //https://blog.csdn.net/u011197534/article/details/78378536
        //https://blog.csdn.net/HaoBBNuanMM/article/details/103054357
        //当我们选择CUBLAS_OP_N时表示不转置，按列优先存储；当我们选择CUBLAS_OP_T时表示需要转置，按行优先存储

        checkCudaErrors(cublasSgemm(handle
                                    , CUBLAS_OP_T
                                    , CUBLAS_OP_T
                                    , matrix_size.uiHC
                                    , matrix_size.uiWC
                                    , matrix_size.uiWA
                                    , &alpha
                                    , d_A
                                    , matrix_size.uiWA
                                    , d_B
                                    , matrix_size.uiWB
                                    , &beta
                                    , d_C
                                    , matrix_size.uiHC));

        // Allocate CUDA events that we'll use for timing
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));

        // Record the start event
        checkCudaErrors(cudaEventRecord(start, nullptr));

        for (int j = 0; j < nIter; j++)
        {
            //note cublas is column primary!
            //need to transpose the order
            checkCudaErrors(cublasSgemm(handle
                                        , CUBLAS_OP_T
                                        , CUBLAS_OP_T
                                        , matrix_size.uiHC
                                        , matrix_size.uiWC
                                        , matrix_size.uiWA
                                        , &alpha
                                        , d_A
                                        , matrix_size.uiWA
                                        , d_B
                                        , matrix_size.uiWB
                                        , &beta
                                        , d_C
                                        , matrix_size.uiHC));

        }

        printf("done.\n");

        // Record the stop event
        checkCudaErrors(cudaEventRecord(stop, nullptr));

        // Wait for the stop event to complete
        checkCudaErrors(cudaEventSynchronize(stop));

        float msecTotal = 0.0f;
        checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

        // Compute and print the performance
        float msecPerMatrixMul = msecTotal / nIter;
        double flopsPerMatrixMul = 2.0 * (double)matrix_size.uiHC * (double)matrix_size.uiWC * (double)matrix_size.uiHB;
        double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
        printf(
                "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
                gigaFlops,
                msecPerMatrixMul,
                flopsPerMatrixMul);

        // copy result from device to host
        checkCudaErrors(cudaMemcpy(h_CUBLAS, d_C, mem_size_C, cudaMemcpyDeviceToHost));

        // Destroy the handle
        checkCudaErrors(cublasDestroy(handle));

    }

    // compute reference solution
    printf("Computing result using host CPU...");
    float *reference = (float *)malloc(mem_size_C);
    matrixMulCPU(reference, h_A, h_B, matrix_size.uiHA, matrix_size.uiWA, matrix_size.uiWB);
    printf("done.\n");

    //transpose resCUBLAS
    transpose<float>(h_CUBLAS,h_CUBLAS_T,matrix_size.uiWC,matrix_size.uiHC);

    // check result (CUBLAS)
    bool resCUBLAS = sdkCompareL2fe(reference, h_CUBLAS_T, size_C, 1.0e-6f);
    //transpose<float,float>(h_CUBLAS,h_CUBLAS_T,matrix_size.uiHC,matrix_size.uiWC);

    if (resCUBLAS != true)
    {
        printDiff(reference, h_CUBLAS_T, matrix_size.uiWC, matrix_size.uiHC, 100, 1.0e-5f);
    }

    printf("Comparing CUBLAS Matrix Multiply with CPU results: %s\n", (true == resCUBLAS) ? "PASS" : "FAIL");

    printf("\nNOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n");

    // clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_CUBLAS_T);
    free(reference);
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

    if (resCUBLAS == true)
    {
        return EXIT_SUCCESS;    // return value = 1
    }
    else
    {
        return EXIT_FAILURE;     // return value = 0
    }

}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    printf("[Matrix Multiply CUBLAS] - Starting...\n");

    int devID = 0, sizeMult = 5;
    sMatrixSize matrix_size;

    initializeCUDA(argc, argv, devID, sizeMult, matrix_size);

    int matrix_result = matrixMultiply_V2(argc, argv, devID, matrix_size);

    return matrix_result;
}
























