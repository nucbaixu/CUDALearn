#include <iostream>
#include <cstdlib>

// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

const char *sSampleName = "simpleStreams";

#define N (1024*1024)
#define FULL_DATA_SIZE (N*20)

__global__ void kernel(const int* a, const int *b, int*c)
{
    unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadID < N)
    {
        c[threadID] = (a[threadID] + b[threadID]) / 2;
    }
}

int TestNoStream()
{
    //启动计时器
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, nullptr);

    int *host_a, *host_b, *host_c;
    int *dev_a, *dev_b, *dev_c;

    //在GPU上分配内存
    cudaMalloc((void**)&dev_a, FULL_DATA_SIZE * sizeof(int));
    cudaMalloc((void**)&dev_b, FULL_DATA_SIZE * sizeof(int));
    cudaMalloc((void**)&dev_c, FULL_DATA_SIZE * sizeof(int));

    //在CPU上分配可分页内存
    host_a = (int*)malloc(FULL_DATA_SIZE * sizeof(int));
    host_b = (int*)malloc(FULL_DATA_SIZE * sizeof(int));
    host_c = (int*)malloc(FULL_DATA_SIZE * sizeof(int));

    //主机上的内存赋值
    for (int i = 0; i < FULL_DATA_SIZE; i++)
    {
        host_a[i] = i;
        host_b[i] = FULL_DATA_SIZE - i;
    }

    //从主机到设备复制数据
    cudaMemcpy(dev_a, host_a, FULL_DATA_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, FULL_DATA_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    kernel <<<FULL_DATA_SIZE / 1024, 1024 >>> (dev_a, dev_b, dev_c);

    //数据拷贝回主机
    cudaMemcpy(host_c, dev_c, FULL_DATA_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    //计时结束
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    std::cout << "use time： " << elapsedTime << std::endl;

    //输出前10个结果
    for (int i = 0; i < 10; i++)
    {
        std::cout << host_c[i] << std::endl;
    }

    cudaFreeHost(host_a);
    cudaFreeHost(host_b);
    cudaFreeHost(host_c);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}

int TestWithStream()
{
    //get cuda prop
    cudaDeviceProp prop{};
    int deviceID;
    cudaGetDevice(&deviceID);
    cudaGetDeviceProperties(&prop,deviceID);

    //if device support overlap function
    if(!prop.deviceOverlap)
    {
        std::cout<<"No device will handle overlaps.so no speed up from stream."<<std::endl;
        return 0;
    }

    //start timer
    cudaEvent_t start,stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, nullptr);

    //creat stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int *host_a, *host_b, *host_c;
    int *dev_a, *dev_b, *dev_c;

    //在GPU上分配内存
    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * sizeof(int));

    //stream need host paned memory
    cudaHostAlloc((void**)&host_a,FULL_DATA_SIZE*sizeof(int),cudaHostAllocDefault);
    cudaHostAlloc((void**)&host_b,FULL_DATA_SIZE*sizeof(int),cudaHostAllocDefault);
    cudaHostAlloc((void**)&host_c,FULL_DATA_SIZE*sizeof(int),cudaHostAllocDefault);

    for (int i = 0; i < FULL_DATA_SIZE; i++)
    {
        host_a[i] = i;
        host_b[i] = FULL_DATA_SIZE - i;
    }

    for(int i = 0; i < FULL_DATA_SIZE; i+= N)
    {
        cudaMemcpyAsync(dev_a,host_a+i,N * sizeof(int),cudaMemcpyHostToDevice,stream);
        cudaMemcpyAsync(dev_b,host_b+i,N * sizeof(int),cudaMemcpyHostToDevice,stream);
        kernel<<<N / 1024,1024,0,stream>>>(dev_a,dev_b,dev_c);
        cudaMemcpyAsync(host_c + i,dev_c,N*sizeof (int),cudaMemcpyDeviceToHost,stream);
    }

    // wait until gpu execution finish
    cudaStreamSynchronize(stream);

    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    std::cout << "use time:" << elapsedTime << std::endl;
    for (int i = 0; i < 10; i++)
    {
        std::cout << host_c[i] << std::endl;
    }

    cudaFreeHost(host_a);
    cudaFreeHost(host_b);
    cudaFreeHost(host_c);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    cudaStreamDestroy(stream);

    return  0;
}

int main()
{
    TestNoStream();
    TestWithStream();
    return 0;
}