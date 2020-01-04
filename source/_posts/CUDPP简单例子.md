---
title: CUDPP简单例子
date: 2018-05-17 09:59:11
tags: CUDPP
---
### scan
在使用CUDPP库之前，必须先初始化cudppCreate()返回一个CUDPP对象句柄，在创建cudppPlan时要用。
```
 // Initialize the CUDPP Library
    CUDPPHandle theCudpp;
    cudppCreate(&theCudpp);

```

<!--more-->

配置cudpp执行什么算法依赖于cudppPlan的概念。cudppPlan用于维护算法的中间存储的数据结构，以及CUDPP可用于优化当前硬件的执行的信息。当调用一个算法时，cudppPlan会将配置细节传递给当前要执行的算法，并且生成一个内部的plan对象。返回一个CUDPPHandle（一个指针对象，用于引用plan对象，必须传递给其他CUDPP函数才能执行算法）。
下面这个例子中对numElements个元素进行exclusive sum-scan。首先将配置信息存储在CUDPPConfiguration 结构对象中。然后将CUDPPConfiguration 传递给cudppPlan，同时传递的参数还有要扫描的元素个数numElements, 最后传递1,0，作为numRows and rowPitch参数，因为这个例子中只是扫描一维数组。
```
    CUDPPConfiguration config;
    config.op = CUDPP_ADD;
    config.datatype = CUDPP_FLOAT;
    config.algorithm = CUDPP_SCAN;
    config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
    
    CUDPPHandle scanplan = 0;
    CUDPPResult res = cudppPlan(theCudpp, &scanplan, config, numElements, 1, 0);  

    if (CUDPP_SUCCESS != res)
    {
        printf("Error creating CUDPPPlan\n");
        exit(-1);
    }
```
现在有了一个scanplan对象句柄，接下来传递plan句柄、input和output设备数组、numElements，调用cudppScan()让CUDPP工作,确保cudppPlan不会运行错误。
```
    // Run the scan
    res = cudppScan(scanplan, d_odata, d_idata, numElements);
    if (CUDPP_SUCCESS != res)
    {
        printf("Error in cudppScan()\n");
        exit(-1);
    }
```
接下来将结果从GPU端读出来，与串行算法做对比验证。
```
    // allocate mem for the result on host side
    float* h_odata = (float*) malloc( memSize);
    // copy result from device to host
    result = cudaMemcpy( h_odata, d_odata, memSize, cudaMemcpyDeviceToHost);
    if (result != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(result));
        exit(-1);
    }
```
最后让CUDPP用cudppDestroyPlan()来清理用于plan对象的内存。然后传递从cudppCreate获取的句柄给cudppDestroy()来关闭库。使用free和cudaFree来释放host端和device端的内存。
```
    // compute reference solution
    float* reference = (float*) malloc( memSize);
    computeSumScanGold( reference, h_idata, numElements, config);

    // check result
    bool passed = true;
    for (unsigned int i = 0; i < numElements; i++)
        if (reference[i] != h_odata[i]) passed = false;
        
    printf( "Test %s\n", passed ? "PASSED" : "FAILED");

    res = cudppDestroyPlan(scanplan);
    if (CUDPP_SUCCESS != res)
    {
        printf("Error destroying CUDPPPlan\n");
        exit(-1);
    }

    // shut down the CUDPP library
    cudppDestroy(theCudpp);
    
    free( h_idata);
    free( h_odata);
    free( reference);
    cudaFree(d_idata);
    cudaFree(d_odata);
}

```
完整代码
```
/*
 * This is a basic example of how to use the CUDPP library.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "cudpp.h"

#include <string>

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

extern "C" 
void computeSumScanGold( float *reference, const float *idata, 
                        const unsigned int len,
                        const CUDPPConfiguration &config);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
    runTest( argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    int dev = 0;
    if (argc > 1) {
        std::string arg = argv[1];
        size_t pos = arg.find("=");
        if (arg.find("device") && pos != std::string::npos) {
            dev = atoi(arg.c_str() + (pos + 1));
        }
    }
    if (dev < 0) dev = 0;
    if (dev > deviceCount-1) dev = deviceCount - 1;
    cudaSetDevice(dev);

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
    {
        printf("Using device %d:\n", dev);
        printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
               prop.name, (int)prop.totalGlobalMem, (int)prop.major, 
               (int)prop.minor, (int)prop.clockRate);
    }

    unsigned int numElements = 32768;
    unsigned int memSize = sizeof( float) * numElements;

    // allocate host memory
    float* h_idata = (float*) malloc( memSize);
    // initalize the memory
    for (unsigned int i = 0; i < numElements; ++i) 
    {
        h_idata[i] = (float) (rand() & 0xf);
    }

    // allocate device memory
    float* d_idata;
    cudaError_t result = cudaMalloc( (void**) &d_idata, memSize);
    if (result != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(result));
        exit(-1);
    }
    
    // copy host memory to device
    result = cudaMemcpy( d_idata, h_idata, memSize, cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(result));
        exit(-1);
    }
     
    // allocate device memory for result
    float* d_odata;
    result = cudaMalloc( (void**) &d_odata, memSize);
    if (result != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(result));
        exit(-1);
    }

    // Initialize the CUDPP Library
    CUDPPHandle theCudpp;
    cudppCreate(&theCudpp);

    CUDPPConfiguration config;
    config.op = CUDPP_ADD;
    config.datatype = CUDPP_FLOAT;
    config.algorithm = CUDPP_SCAN;
    config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
    
    CUDPPHandle scanplan = 0;
    CUDPPResult res = cudppPlan(theCudpp, &scanplan, config, numElements, 1, 0);  

    if (CUDPP_SUCCESS != res)
    {
        printf("Error creating CUDPPPlan\n");
        exit(-1);
    }

    // Run the scan
    res = cudppScan(scanplan, d_odata, d_idata, numElements);
    if (CUDPP_SUCCESS != res)
    {
        printf("Error in cudppScan()\n");
        exit(-1);
    }

    // allocate mem for the result on host side
    float* h_odata = (float*) malloc( memSize);
    // copy result from device to host
    result = cudaMemcpy( h_odata, d_odata, memSize, cudaMemcpyDeviceToHost);
    if (result != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(result));
        exit(-1);
    }
    
    // compute reference solution
    float* reference = (float*) malloc( memSize);
    computeSumScanGold( reference, h_idata, numElements, config);

    // check result
    bool passed = true;
    for (unsigned int i = 0; i < numElements; i++)
        if (reference[i] != h_odata[i]) passed = false;
        
    printf( "Test %s\n", passed ? "PASSED" : "FAILED");

    res = cudppDestroyPlan(scanplan);
    if (CUDPP_SUCCESS != res)
    {
        printf("Error destroying CUDPPPlan\n");
        exit(-1);
    }

    // shut down the CUDPP library
    cudppDestroy(theCudpp);
    
    free( h_idata);
    free( h_odata);
    free( reference);
    cudaFree(d_idata);
    cudaFree(d_odata);
}
```
### compact
```
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "include\cudpp.h"
#include <string>

void runTest(int argc, char** argv);
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv)
{
    runTest(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest(int argc, char** argv)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    int dev = 0;
    if (argc > 1) {
        std::string arg = argv[1];
        size_t pos = arg.find("=");
        if (arg.find("device") && pos != std::string::npos) {
            dev = atoi(arg.c_str() + (pos + 1));
        }
    }
    if (dev < 0) dev = 0;
    if (dev > deviceCount - 1) dev = deviceCount - 1;
    cudaSetDevice(dev);

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
    {
        printf("Using device %d:\n", dev);
        printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
            prop.name, (int)prop.totalGlobalMem, (int)prop.major,
            (int)prop.minor, (int)prop.clockRate);
    }
    const int numElements = 10;
    unsigned int memSize = sizeof(int)*numElements;
    
    unsigned int h_isValid[numElements] = { 0, 0, 1, 1, 0, 1, 0, 1, 1, 1 };
    int             h_vals[numElements] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    int* h_output = NULL;
    size_t h_numValidElements = 0;

    //allocate host memory t store the input data
    unsigned int *d_isValid;
    int *d_vals;
    int *d_output = NULL;
    size_t* d_numValid = NULL;

    cudaMalloc((void**)&d_isValid, sizeof(unsigned int)*numElements);
    cudaMemcpy(d_isValid, h_isValid, memSize, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_vals, memSize);
    cudaMemcpy(d_vals, h_vals, memSize, cudaMemcpyHostToDevice);
    
    cudaMalloc((void**)&d_numValid, sizeof(size_t));
    cudaMalloc((void**)&d_output, memSize);
    cudaMemset(d_output, 0, memSize);


    // Initialize the CUDPP Library
    CUDPPResult result = CUDPP_SUCCESS;
    CUDPPHandle theCudpp;
    result = cudppCreate(&theCudpp);

    CUDPPConfiguration config;
    config.datatype = CUDPP_INT;
    config.algorithm = CUDPP_COMPACT;
    config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;

    CUDPPHandle compactplan = 0;
    CUDPPResult res = cudppPlan(theCudpp, &compactplan, config, numElements, 1, 0);

    if (CUDPP_SUCCESS != res)
    {
        printf("Error creating CUDPPPlan\n");
        exit(-1);
    }

    // Run the compact
    res = cudppCompact(compactplan, d_output, d_numValid, d_vals, d_isValid, numElements);
    if (CUDPP_SUCCESS != res)
    {
        printf("Error in cudppScan()\n");
        exit(-1);
    }
    cudaMemcpy(&h_numValidElements, d_numValid, sizeof(size_t), cudaMemcpyDeviceToHost);
    h_output = (int*)malloc(sizeof(int)*h_numValidElements);
    cudaMemcpy(h_output, d_output, sizeof(int)*h_numValidElements, cudaMemcpyDeviceToHost);
    printf("numValidElements: %ld\n", h_numValidElements);
    int i = 0;
    for (i = 0; i < h_numValidElements; i++)
    {
        printf("ValidElements[%d]: %ld\n", i, h_output[i]);
    }

    // cleanup memory
    cudppDestroyPlan(compactplan);
    cudppDestroy(theCudpp);
    free(h_output);
    cudaFree(d_isValid);
    cudaFree(d_vals);
    cudaFree(d_output);
    cudaFree(d_numValid);

}
```
### cudpp_hash
```
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "cudpp.h"
#include "cudpp_hash.h"

int main() {
    const int N = 10;

    int keys[N] = { 1, 6, 4, 9, 0, 3, 7, 2, 5, 8 };
    int vals[N] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

    int *d_keys, *d_vals;
    cudaMalloc((void**)&d_keys, sizeof(int)* N);
    cudaMemcpy(d_keys, keys, sizeof(int)* N, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_vals, sizeof(int)* N);
    cudaMemcpy(d_vals, vals, sizeof(int)* N, cudaMemcpyHostToDevice);

    int input[N] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    int output[N];

    int *d_input, *d_output;
    cudaMalloc((void**)&d_input, sizeof(int)* N);
    cudaMemcpy(d_input, input, sizeof(int)* N, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_output, sizeof(int)* N);
    cudaMemset(d_output, 0, sizeof(int)* N);

    // CUDPP初始化
    CUDPPHandle cudpp;
    cudppCreate(&cudpp);

    // 設定hashtable
    CUDPPHashTableConfig config;
    config.type = CUDPP_BASIC_HASH_TABLE;
    config.kInputSize = N;
    config.space_usage = 2.0;

    CUDPPHandle hash_table_handle;
    cudppHashTable(cudpp, &hash_table_handle, &config);

    cudppHashInsert(hash_table_handle, d_keys, d_vals, N);

    cudppHashRetrieve(hash_table_handle, d_input, d_output, N);

    cudaMemcpy(output, d_output, sizeof(int)* N, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i) {
        printf("%d\n", output[i]);
    }

    cudppDestroyHashTable(cudpp, hash_table_handle);

    cudppDestroy(cudpp);

    return 0;
}
```

### 混合reduce和compact的compact例子

通过reduce计算出总数，这样可以少占用GPU端显存

```
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "include\cudpp.h"
#include <string>

void runTest(int argc, char** argv);
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    runTest(argc, argv);
}
////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char** argv)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    int dev = 0;
    if (argc > 1) {
        std::string arg = argv[1];
        size_t pos = arg.find("=");
        if (arg.find("device") && pos != std::string::npos) {
            dev = atoi(arg.c_str() + (pos + 1));
        }
    }
    if (dev < 0) dev = 0;
    if (dev > deviceCount - 1) dev = deviceCount - 1;
    cudaSetDevice(dev);

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
    {
        printf("Using device %d:\n", dev);
        printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
            prop.name, (int)prop.totalGlobalMem, (int)prop.major,
            (int)prop.minor, (int)prop.clockRate);
    }
    const int numElements = 10;
    unsigned int memSize = sizeof(int)*numElements;
    
    unsigned int h_isValid[numElements] = { 0, 0, 1, 1, 0, 1, 0, 1, 1, 1 };
    int             h_vals[numElements] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    int* h_output = NULL;
    size_t h_numValidElements = 0;

    //allocate host memory t store the input data
    unsigned int *d_isValid;
    int *d_vals;
    int *d_output = NULL;
    size_t* d_numValid = NULL;

    cudaMalloc((void**)&d_isValid, sizeof(unsigned int)*numElements);
    cudaMemcpy(d_isValid, h_isValid, memSize, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_vals, memSize);
    cudaMemcpy(d_vals, h_vals, memSize, cudaMemcpyHostToDevice);
    
    cudaMalloc((void**)&d_numValid, sizeof(size_t));



    // Initialize the CUDPP Library
    CUDPPResult result = CUDPP_SUCCESS;
    CUDPPHandle theCudpp;
    result = cudppCreate(&theCudpp);


    //有效元素数量
    int* d_reduceSum = NULL;
    int h_reduceSum = 0;
    cudaMalloc((void**)&d_reduceSum, sizeof(int));
    cudaMemset(d_reduceSum, 0, sizeof(int));
    CUDPPConfiguration configReduce;
    configReduce.op == CUDPP_ADD;
    configReduce.datatype = CUDPP_INT;
    configReduce.algorithm = CUDPP_REDUCE;
    configReduce.options = 0;
    CUDPPHandle reduceplan = 0;
    cudppPlan(theCudpp, &reduceplan, configReduce, numElements, 1, 0);
    cudppReduce(reduceplan, d_reduceSum, d_isValid, numElements);
    cudaMemcpy(&h_reduceSum, d_reduceSum, sizeof(int), cudaMemcpyDeviceToHost);
    printf("h_reduceSum %d ", h_reduceSum);

    cudaMalloc((void**)&d_output, h_reduceSum*sizeof(int));
    cudaMemset(d_output, 0, h_reduceSum*sizeof(int));


    CUDPPConfiguration config;
    //config.op == CUDPP_ADD;
    config.datatype = CUDPP_INT;
    config.algorithm = CUDPP_COMPACT;
    config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;

    CUDPPHandle compactplan = 0;
    CUDPPResult res = cudppPlan(theCudpp, &compactplan, config, numElements, 1, 0);

    if (CUDPP_SUCCESS != res)
    {
        printf("Error creating CUDPPPlan\n");
        exit(-1);
    }

    // Run the compact
    res = cudppCompact(compactplan, d_output, d_numValid, d_vals, d_isValid, numElements);
    if (CUDPP_SUCCESS != res)
    {
        printf("Error in cudppScan()\n");
        exit(-1);
    }
    cudaMemcpy(&h_numValidElements, d_numValid, sizeof(size_t), cudaMemcpyDeviceToHost);
    
    
    h_output = (int*)malloc(sizeof(int)*h_numValidElements);
    cudaMemcpy(h_output, d_output, sizeof(int)*h_numValidElements, cudaMemcpyDeviceToHost);
    printf("numValidElements: %ld\n", h_numValidElements);
    int i = 0;
    for (i = 0; i < h_numValidElements; i++)
    {
        printf("ValidElements[%d]: %ld\n", i, h_output[i]);
    }

    // cleanup memory
    cudppDestroyPlan(reduceplan);
    cudppDestroyPlan(compactplan);
    cudppDestroy(theCudpp);
    free(h_output);
    cudaFree(d_isValid);
    cudaFree(d_vals);
    cudaFree(d_output);
    cudaFree(d_numValid);
}
```

