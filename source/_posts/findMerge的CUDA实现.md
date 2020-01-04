---
title: findMerge的CUDA实现
date: 2018-05-19 10:07:37
tags: CUDPP
---

### 算法目标
根据原图像值，在两个相邻分块的标记值中寻找要合并的标记，并将其分别存放在father和child两个数组中。寻找8连通的合并对，其实就是以第一行为核心按照如下模板进行遍历，如果图像pixel的原值相同，就说明这两个pixel所在的标记应该合并，就将上一行计入father数组，将下一行计入child数组。

<!--more-->

![图片](1.png)
例如：
```
input：
    const int width = 5;
    int h_rowOneValue[width] = { 1, 1, 2, 1, 3 };
    int h_rowTwoValue[width] = { 4, 1, 1, 1, 1 };

    int h_rowOneLabel[width] = { 1, 1, 4, 2, 5 };
    int h_rowTwoLabel[width] = { 6, 3, 3, 3, 3 };

output
  father {1，2}
  child  {3，3}
```
### CUDA并行算法思想
将每个pixel映射到一个CUDA线程上，将可能要合并的pair先都存储，然后将重复项和无效项去除。以上面的例子作为输入来说明，首先会产生一个3倍于原输入大小的father和child，因为每个线程都要对应遍历模板产生3个输出。
```
father {-1，-1，1，  -1，1，1，  -1，-1，-1，  2，2，2，  -1，-1，-1}
child  {-1，-1，3，  -1，3，3，  -1，-1，-1，  3，3，3，  -1，-1，-1}
IsValid{ 0， 0，1，   0，0，0，   0， 0， 0，  1，0，0，   0， 0， 0}
```
然后用CUDPP compact 操作根据中间数组IsValid将无效项（-1，-1）和重复项（1,3）（2,3）去掉。
最终输出
```
  father {1，2}
  child  {3，3}
```
### 程序实现过程中遇到的问题
一、怎样在一个函数中再调用另外的函数对传进来的表示数组的指针申请内存空间并进行修改赋值等操作。
之前的经验告诉我，要在一个函数内申请内存空间并使得赋值有效传回，得传递双指针。脑子里有指针传递和值传递这个概念，但一直不是很清晰，对函数的调用机制也不清晰。下面就来搞清楚这中间到底发生了什么。
指针就是一个变量，存放的是地址，也是一个值而已。函数参数的传递实际上是一个拷贝的过程。以前是在被调用的函数中改变调用者中实参的值，用指向实参的指针传值就可以，现在，要在被调用函数中改变指针的指向，就要用指向指针的指针（双重指针）。在被调用函数中**ptr代表指针指向的值，*ptr代表指针，ptr代表指针的指针就是指针的地址。
![图片](2.png)
其实在函数传递中我们只是将指针的地址复制了过来，但是这个地址仍然指向这个指针。所以当需要嵌套的在函数中传递一个数组，并要在最内层的函数调用中申请数组空间或修改数组的值的时候，要将指针的指针作为函数的参数传递，这样每次向内层的调用都会复制指针的地址，修改指针指向的地址中的值。

二、怎样让一个线程调用另外一个核函数，我使用了__device__函数，但是启动失败。这个问题待解决

三、遇到CUDA 的报错 invalid device function.
cudpp中可能用到了比较高级的函数，必须要用计算能力大于3.5的显卡，并且在编译的时候设置 -arch=sm_35.原因是有些特性在低端的机器上是不支持的，需要指明显卡的计算能力。所以首先你得知道自己的显卡的计算能力，在编译的时候将-arch改为与自己的显卡相匹配的数值。
### 并行算法实现
在实现中将原先的数组前后各添加一个无效项，避免边界处理操作。然后对每个线程使用模板，将产生的pair存放于father和child，然后得到IsValid数组，最后去重得到有效的pair。
每个线程产生的pair的存放位置如下:
![图片](3.png)

```
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "cudpp.h"
#include "cudpp_hash.h"

__global__ void getFatherAndChild(int* d_rowOneValue, int* d_rowTwoValue, int* d_rowOneLabel, int* d_rowTwoLabel, int width,
                                 int* d_father, int* d_child)
{
    const int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    /*//gid是针对于第一行的
    要遍历的模板是下面这种
         A
    A   A   A
    */
    //将原先的数组前后各添加一个无效项，避免边界处理操作
    if ((gid == 0) || (gid == width+1))
    {
        d_rowOneValue[gid] = -1;
        d_rowTwoValue[gid] = -1;
        d_rowOneLabel[gid] = -1;
        d_rowTwoLabel[gid] = -1;
    }
    if ((gid > 0) && (gid < width+1))
    {
        if (d_rowOneValue[gid] == d_rowTwoValue[gid - 1])
        {
            d_father[3 * (gid - 1)] = d_rowOneLabel[gid];
            d_child[3 * (gid - 1)] = d_rowTwoLabel[gid - 1];
        }
        else
        {
            d_father[3 * (gid - 1)] = -1;
            d_child[3 * (gid - 1)] = -1;
        }

        if (d_rowOneValue[gid] == d_rowTwoValue[gid])
        {
            d_father[3 * (gid - 1) + 1] = d_rowOneLabel[gid];
            d_child[3 * (gid - 1) + 1] = d_rowTwoLabel[gid];
        }
        else
        {
            d_father[3 * (gid - 1) + 1] = -1;
            d_child[3 * (gid - 1) + 1] = -1;
        }

        if (d_rowOneValue[gid] == d_rowTwoValue[gid + 1])
        {
            d_father[3 * (gid - 1) + 2] = d_rowOneLabel[gid];
            d_child[3 * (gid - 1) + 2] = d_rowTwoLabel[gid + 1];
        }
        else
        {
            d_father[3 * (gid - 1) + 2] = -1;
            d_child[3 * (gid - 1) + 2] = -1;
        }
    }

}

__device__ void reLabelRepeatItem(int* d_father, int* d_child, int currentId)
{
    const int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    if ((gid < currentId) && (d_father[gid] == d_father[currentId]) && (d_child[gid] == d_child[currentId]))
    {
        d_father[currentId] = -1;
        d_child[currentId] = -1;
    }
}
__global__ void getIsvalid(int* d_father, int* d_child, unsigned int* d_isValid, int N, int numThreads, int numBlocks)
{
    const int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int currentId;
    if (gid < N)
    {
        currentId = gid;
        int i = 0;
        for (i = currentId - 1; i >= 0; i--)
        {
            if ((d_father[i] == d_father[currentId]) && (d_child[i] == d_child[currentId]))
            {
                d_father[currentId] = -1;
                d_child[currentId] = -1;
                break;
            }
        }
        //reLabelRepeatItem << <numBlocks, numThreads >> >(d_father, d_child, currentId);
    }
    __syncthreads();
    if (gid < N)
    {
        if (d_father[gid] > 0)
        {
            d_isValid[gid] = 1;
        }
    }
}

size_t compactArray(int* d_Val, unsigned int* d_isValid, int totalNum, CUDPPHandle& compactplan,
    int** h_ValValid)
{
    int memSize = sizeof(int)* totalNum;
    
    size_t* d_numValid = NULL;
    cudaMalloc((void**)&d_numValid, sizeof(size_t));
    int *d_ValValid;
    cudaMalloc((void**)&d_ValValid, memSize);
    cudaMemset(d_ValValid, 0, memSize);

    // Run the compact
    CUDPPResult res = cudppCompact(compactplan, d_ValValid, d_numValid, d_Val, d_isValid, totalNum);
    if (CUDPP_SUCCESS != res)
    {
        printf("Error in cudppScan()\n");
        exit(-1);
    }
    size_t h_numValidElements = 0;
    cudaMemcpy(&h_numValidElements, d_numValid, sizeof(size_t), cudaMemcpyDeviceToHost);

    *h_ValValid = (int*)malloc(sizeof(int)*h_numValidElements);
    cudaMemcpy(*h_ValValid, d_ValValid, sizeof(int)*h_numValidElements, cudaMemcpyDeviceToHost);

    cudaFree(d_numValid);
    cudaFree(d_ValValid);
    return h_numValidElements;
}



size_t compactFatherAndChild(int* d_father, int* d_child, unsigned int* d_isValid, int totalNum,
                            int** h_fatherValid, int** h_childValid)
{
    CUDPPHandle cudppForCompact;
    cudppCreate(&cudppForCompact);

    CUDPPConfiguration compactConfig;
    compactConfig.datatype = CUDPP_INT;
    compactConfig.algorithm = CUDPP_COMPACT;
    compactConfig.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
    
    CUDPPHandle compactplan = 0;
    CUDPPResult res = cudppPlan(cudppForCompact, &compactplan, compactConfig, totalNum, 1, 0);
    if (CUDPP_SUCCESS != res)
    {
        printf("Error creating CUDPPPlan\n");
        exit(-1);
    }
    compactArray(d_father, d_isValid, totalNum, compactplan, h_fatherValid);
    size_t h_numValidElements = compactArray(d_child, d_isValid, totalNum, compactplan, h_childValid);
    
    cudppDestroy(cudppForCompact);
    cudppDestroyPlan(compactplan);
    
    return h_numValidElements;
}
void checkDeviceArrayOfTypeInt(int* d_val, int length, char* name)
{
    int memSize = sizeof(int)*length;
    int *h_val = (int*)malloc(memSize);
    cudaMemcpy(h_val, d_val, memSize, cudaMemcpyDeviceToHost);
    printf("%s:\n",name);
    int i = 0;
    for (i = 0; i < length; i++)
    {
        printf("%d\t", h_val[i]);
    }
    printf("\n");
    free(h_val);
}

size_t findMergeFunc(int* h_rowOneValue, int* h_rowTwoValue, int* h_rowOneLabel, int* h_rowTwoLabel, int width, int numThreads, int numBlocks,
    int** h_fatherValid, int** h_childValid)
{
    int memSize = sizeof(int)* (width + 2);//这里设置数组长度为width+2是为了在前后都加一个无效项，避免边界处理操作
    
    int *d_rowOneValue = NULL;
    cudaMalloc((void**)&d_rowOneValue, memSize);
    cudaMemcpy(d_rowOneValue+1, h_rowOneValue, sizeof(int)* (width), cudaMemcpyHostToDevice);
    
    int *d_rowTwoValue = NULL;
    cudaMalloc((void**)&d_rowTwoValue, memSize);
    cudaMemcpy(d_rowTwoValue + 1, h_rowTwoValue, sizeof(int)* (width), cudaMemcpyHostToDevice);

    int *d_rowOneLabel = NULL;
    cudaMalloc((void**)&d_rowOneLabel, memSize);
    cudaMemcpy(d_rowOneLabel + 1, h_rowOneLabel, sizeof(int)* (width), cudaMemcpyHostToDevice);

    int *d_rowTwoLabel = NULL;
    cudaMalloc((void**)&d_rowTwoLabel, memSize);
    cudaMemcpy(d_rowTwoLabel + 1, h_rowTwoLabel, sizeof(int)* (width), cudaMemcpyHostToDevice);


    int memSizeFatherChild = sizeof(int)* width * 3;
    
    int* d_father = NULL;
    cudaMalloc((void**)&d_father, memSizeFatherChild);
    cudaMemset(d_father, 0, memSizeFatherChild);
    int* d_child = NULL;
    cudaMalloc((void**)&d_child, memSizeFatherChild);
    cudaMemset(d_child, 0, memSizeFatherChild);

    //接下来就是给d_father和d_child赋值
    getFatherAndChild << <numBlocks, numThreads >> >(d_rowOneValue, d_rowTwoValue, d_rowOneLabel, d_rowTwoLabel, width, d_father, d_child);
    
    char str_father[] = "d_father";
    checkDeviceArrayOfTypeInt(d_father, width * 3, str_father);
    char str_child[] = "d_father";
    checkDeviceArrayOfTypeInt(d_child, width * 3, str_child);

    //到这里为止father和child设置完毕，接下来要进行去重操作
    //首先声明一个有效数组用来标记是否有效d_isValid
    unsigned int* d_isValid = NULL;
    cudaMalloc((void**)&d_isValid, sizeof(unsigned int)* width * 3);
    cudaMemset(d_isValid, 0, memSizeFatherChild);
    getIsvalid << <numBlocks, numThreads >> >(d_father, d_child, d_isValid, width * 3, numThreads, numBlocks);

    int *h_isValid = (int*)malloc(sizeof(unsigned int)* width * 3);
    cudaMemcpy(h_isValid, d_isValid, sizeof(unsigned int)* width * 3, cudaMemcpyDeviceToHost);
    printf("h_isValid:\n");
    int i;
    for (i = 0; i < width * 3; i++)
    {
        printf("%d\t", h_isValid[i]);
    }
    printf("\n");


    //现在已经有了d_father和d_child，以及d_isValid，接下来需要对d_father和d_child根据d_isValid进行compact操作
    
    size_t h_numValidElements = compactFatherAndChild(d_father, d_child, d_isValid, width * 3, h_fatherValid, h_childValid);

    cudaFree(d_rowOneValue);
    cudaFree(d_rowTwoValue);
    cudaFree(d_rowOneLabel);
    cudaFree(d_rowTwoLabel);
    cudaFree(d_father);
    cudaFree(d_child);
    cudaFree(d_isValid);

    return h_numValidElements;
}
void initCUDA(int& devIdx)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    int dev = 0;
    if (devIdx < 0) dev = 0;
    if (devIdx > deviceCount - 1) dev = deviceCount - 1;
    else dev = devIdx;
    cudaSetDevice(dev);


    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
    {
        printf("Using device %d:\n", dev);
        printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
            prop.name, (int)prop.totalGlobalMem, (int)prop.major,
            (int)prop.minor, (int)prop.clockRate);
    }
}
void findMerge()
{
 int gpuIdx = 1;
 initCUDA(gpuIdx);
    const int width = 5;
    int h_rowOneValue[width] = { 1, 1, 2, 1, 3 };
    int h_rowTwoValue[width] = { 4, 2, 1, 1, 1 };

    int h_rowOneLabel[width] = { 1, 1, 4, 2, 5 };
    int h_rowTwoLabel[width] = { 6, 7, 3, 3, 3 };

    int numThreads = 1024;
    int numBlocks = (width + numThreads - 1) / numThreads;
    size_t h_numValidElements = 0;
    int* h_fatherValid = NULL;
    int* h_childValid = NULL;

    h_numValidElements = findMergeFunc(h_rowOneValue, h_rowTwoValue, h_rowOneLabel, h_rowTwoLabel, width, numThreads, numBlocks, &h_fatherValid, &h_childValid);
    

    printf("numValidElements: %ld\n", h_numValidElements);
    int i = 0;
    for (i = 0; i < h_numValidElements; i++)
    {
        printf("%d\t", h_fatherValid[i]);
    }
    printf("\n");
    for (i = 0; i < h_numValidElements; i++)
    {
        printf("%d\t", h_childValid[i]);
    }
    printf("\n");
    free(h_fatherValid);
    free(h_childValid);
}
```

