---
title: 用CUDPP实现类似并查集的find和union功能
date: 2018-04-23 12:38:48
tags: CUDA
---

## 需求说明
对于给定的键值对key-value，根据另一个给定的pair（father，child）将key进行合并。pair代表key数组中等价的key。要做的就是将所有等价的key用其中最小的值代替，并将对应的value值都累积到这个key对应的value中。
实现这个过程的目的是在计算景观指数时，keys对应与在各个分块中的标记值，vals对应与每个斑块的属性（面积、周长等），而pair对应与将两个分块进行合并时所需合并的斑块对。比如下图中（1,3）（2,4）（3,5）（4,5）都是需要合并的斑块，最终合并为一个斑块标记值key为1，值为vals[1]+vals[2]+vals[3]+vals[4]+vals[5];


| input         | 名称                |  示例               |
| --------- | -------- | ----------------------------- |
| 键   |  keys    | int keys[totalNum] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };   | 
| 值   |  vals   | int vals[totalNum]  = { 1, 6, 4, 9, 0, 3, 7, 2, 5, 8 };   | 
| 元素总数   | totalNum   | const int totalNum = 10;   | 
| pair----father   | father   | int father[4] = { 1, 2, 3, 4 };   | 
| pair----child   | child   | int child[4]  =  { 3, 4, 5, 5 };   | 

<!--more-->

以上输入对应的需要进行合并的部分为{1,2,3,4,5}.图示如下
![图片](1.png)
合并之后的ouput如下


| output   | 名称   | 示例（对于上面的input有以下输出）   | 
|:----|:----|:----|
| 有效的key   | h_rootValid   | 0       1       6       7       8       9   | 
| 累积后的value   | h_SumValVaild   | 1       22      7       2       5       8   | 

## 串行算法思想
采用并查集，可以实现O(1)的查找效率。
## 并行算法思想
由于采用并查集的方法，存储利用率太低，只有不到5%，所以想到用hash的方法存储，这样可以保证只要槽的个数不是远小于插入的元素的个数（实际上在cudpp_hash中槽的个数至少为输入元素个数的1.05倍），查询时间就会保持在常数级。也就是说能在常数级完成对动态集合中的操作。由于连通域标记的值是不连续的，在我之前的程序中是用一个连通域第一个pixel的一维索引作为整个连通域的标记值，所以如果用并查集数组的方法存储，需要申请一个与整个图像pixel数一样多的一个数组，而且这个数组是非常稀疏的，空间利用率太低，所以想到了用hash table的方法存储。需要保持在用较少的内存的情况下查询效率几乎不受影响。而且要实现像并查集一样的两个重要功能：
1、寻找根节点find
2、合并根节点相同的节点
### find-union算法思想
输入：d_keys, d_father, d_child, totalNum（总的元素个数）， numberOfMerge（要合并的pair数量）
输出：d_root
中间变量： int* d_fatherroot, d_childroot, d_keyIndex, d_updateIndex,
int  d_ischanged

* d_keys, d_father, d_child是从CPU端拷贝到GPU端（d_keys相当于是进行连通域标记后的标记值，d_father和d_child为按照相邻两个分块处标记值和原图像值对比而得到的需要合并的标记）。
* d_root初始化为d_keys本身（借鉴union-find）数组长度为totalNum
* d_fatherroot和d_childroot，数组长度为numberOfMerge,用于存放father的根节点和child的根节点。所谓根节点就是与当前标记值等价的标记值中的最小值。
* d_keyIndex为d_keys的一维索引值，就是按照顺序的{1,2,3,4,5,6……}
* d_updateIndex为d_root中需要更新的位置索引
* d_ischanged，初始化为false。用于标记这一轮是否还有元素改变了，如果没有，就交换father和child

程序中要构造两个hashtable

1. （d_key,d_root）便于通过d_key快速的找到所对应的d_root。
2. （d_key,d_keyIndex）便于通过d_key快速找到它的一维索引值（也就是要修改的位置）

寻找根节点的步骤如下

1. 首先在d_root中找到了father的root，分别存放于d_fatherroot中
2. 然后找到d_child中的根的一维索引，这些位置的d_root是接下来需要更新的位置，存放在d_updateIndex中
3. 更新d_root，修改d_ischanged值。如果d_fatherroot的值比要更新的位置的值小，也就是说如果father的根比child的根值要小，就修改child的根的值。father的root和child的root是存放在同一个数组d_root中的。

```
__global__  void    updateRoot(int* d_updateIndex, int* d_root, int* d_fatherroot, int N_update, bool* ischanged)
{
    const int tid = threadIdx.x;
    int globalid = blockIdx.x * blockDim.x + tid;
    if (globalid < N_update)
    {
        if (d_root[d_updateIndex[globalid]] > d_fatherroot[globalid])
        {
            *ischanged = true;
            atomicMin(&d_root[d_updateIndex[globalid]], d_fatherroot[globalid]);
        }
    }
}
```

4. 将d_ischanged传回主机端，如果这次d_root已经没有更新了，则退出while循环，否则更新hashtable（d_key,d_root），进入新一轮的循环。直到d_root没有更新为止，也就是子节点的root<=父节点的root
5. 接下来交换father和child的角色，更新d_root直到父节点的root<=子节点的root，这样就结束了整个查找根节点的过程。

```
    do{
        h_ischanged = false;
        cudaMemcpy(d_ischanged, &h_ischanged, sizeof(bool), cudaMemcpyHostToDevice);
        cudppHashRetrieve(hash_table_handle, d_father, d_fatherroot, totalNum); //在d_root中找到了father的root        d_fatherroot
        //在d_root中修改，将child的root修改为father的root
        //在d_keyIndex中找要修改的元素的索引，即找d_childroot的索引，也就是检测是否要修改的元素的索引；
        cudppHashRetrieve(hash_table_handleForFindIndex, d_child, d_updateIndex, numberOfMerge);
        updateRoot << <numBlocks, numThreads >> >(d_updateIndex, d_root, d_fatherroot, numberOfMerge, d_ischanged);
        cudaMemcpy(&h_ischanged, d_ischanged, sizeof(bool), cudaMemcpyDeviceToHost);
        if (h_ischanged)
            cudppHashInsert(hash_table_handle, d_keys, d_root, totalNum);
    } while (h_ischanged);

    do{
        h_ischanged = false;
        cudaMemcpy(d_ischanged, &h_ischanged, sizeof(bool), cudaMemcpyHostToDevice);
        cudppHashRetrieve(hash_table_handle, d_child, d_childroot, totalNum);       //在d_root中找到了child的root     d_childroot
        //在d_root中更新，按照大小对比，将father的root修改为child的root
        //在d_keyIndex中找要修改的元素的索引，即找d_fatherroot的索引，也就是检测是否要修改的元素的索引；
        cudppHashRetrieve(hash_table_handleForFindIndex, d_father, d_updateIndex, numberOfMerge);
        updateRoot << <numBlocks, numThreads >> >(d_updateIndex, d_root, d_childroot, numberOfMerge, d_ischanged);
        cudaMemcpy(&h_ischanged, d_ischanged, sizeof(bool), cudaMemcpyDeviceToHost);
        if (h_ischanged)
            cudppHashInsert(hash_table_handle, d_keys, d_root, totalNum);
    } while (h_ischanged);
```

整个过程的图示为


| 初始值   | 中间变量   | 
|:----|:----|
| d_root = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };   | d_fatherroot = {1,2,3,4}   | 
| int father[4] = { 1, 2, 3, 4 }; | d_childroot = {3,4,5,5}   | 
| int child[4]  =  { 3, 4, 5, 5 }; | d_updateIndex = {3,4,5,5}   | 

| 第一轮 | d_fatherroot = {1,2,1,2} | 
| ------- | ------ |
| d_root = { 0, 1, 2, 1, 2, 3, 6, 7, 8, 9 }; | d_childroot = {1,2,3,3} | 
| d_ischanged = true   | d_updateIndex = {3,4,5,5} | 

| 第二轮 | d_fatherroot = {1,2,1,1} | 
| ------- | ------ |
| d_root = { 0, 1, 2, 1, 1, 1, 6, 7, 8, 9 }; | d_childroot = {1,1,1,1} | 
| d_ischanged = true | d_updateIndex = {3,4,5,5} | 

| 第三轮 | d_fatherroot = {1,2,1,1} | 
| ------- | ------ |
| d_root = { 0, 1, 2, 1, 1, 1, 6, 7, 8, 9 }; | d_childroot = {1,1,1,1} | 
| d_ischanged = false   | d_updateIndex = {3,4,5,5} | 

此时child的root<=father的root，d_ischanged=false；结束第一个while循环。交换father和child的角色，开始第二个while循环。

| 第四轮 | d_fatherroot = {1,2,1,1} | 
| ------- | ------ |
| d_root = { 0, 1, 1, 1, 1, 1, 6, 7, 8, 9 }; | d_childroot = {1,1,1,1} | 
| d_ischanged = true | d_updateIndex = {1,2,3,4} | 


| 第五轮 | d_fatherroot = {1,1,1,1} | 
| ------- | ------ |
| d_root = { 0, 1, 1, 1, 1, 1, 6, 7, 8, 9 }; | d_childroot = {1,1,1,1} | 
| d_ischanged = false   | d_updateIndex = {1,2,3,4} | 

此时father的root<=child的root，d_ischanged=false；结束第二个while循环。至此寻找根节点结束。形成d_root = { 0, 1, 1, 1, 1, 1, 6, 7, 8, 9 };
### 合并值算法思想
输入：d_root, d_vals
输出：d_rootValid, d_SumValsValid，d_numValid
中间变量：int* d_SumVals, d_isValid


| input   | 预期output   | 
| ------- | ------ |
| d_root = { 0, 1, 1, 1, 1, 1, 6, 7, 8, 9 }; | d_rootVaild         = { 0, 1,  6, 7, 8, 9 }; | 
| d_vals = { 1, 6, 4, 9, 0, 3, 7, 2, 5, 8 }; | d_SumValsValid = {1, 22, 7, 2, 5, 8}   | 

1. 先通过hashtable（d_key,d_keyIndex）在d_root中找到每个root的一维索引d_rootIndex。
2. getSumVals函数将d_vals中的值都加到索引对应的位置上得到d_SumVals 。
3. getIsvalid函数获得有效的位置，也就是d_SumVals > 0的位置。
4. 然后用compact操作获得有效的d_root 和d_SumVals。分别存储于d_rootValid和h_SumValsValid


| d_root         = { 0, 1, 1, 1, 1, 1, 6, 7, 8, 9 }; | 
|:----|
| d_rootIndex = { 0, 1, 1, 1, 1, 1, 6, 7, 8, 9 };   | 
| d_vals          ={ 1, 6, 4, 9, 0, 3, 7, 2, 5, 8 }; | 
| d_SumVals  ={ 1, 22, 0, 0, 0,0, 7, 2, 5, 8 };    | 
| d_isValid      = {1, 1,    0, 0, 0,0, 1, 1, 1, 1}   | 
| d_SumValsValid = {1, 22, 7, 2,5, 8}   | 
| d_rootVaild         = { 0, 1,  6, 7, 8, 9 };   | 

上述过程用到了cudpp中的compact功能，去掉值为0 的元素。原理就是通过scan操作找出每个非零元素应该存放的位置，并获取非零元素的总数d_numValid。然后申请对应大小的空间。将结果从设备端拷贝回去。
其实这样初始情况下在设备端还是申请了与原数组长度相同的totalNum大小的d_valValid数组，因为d_numValid是在compact函数中返回的我们事先并不知道有效元素有多少个，所以新申请的用于存放有效元素的数组d_valValid中只有前d_numValid的空间是有效的。如果设备端的空间很有限，可以采用先对isvalid数组进行规约操作获得d_numValid。然后申请对应大小的d_valValid，传入compact函数，同样可以得到compact后的结果，这样做相当于用时间换取空间，我们多reduce计算了一次有效元素的数量，换来的是只需要申请d_numValid大小的d_valValid。
![图片](2.png)![图片](3.png)

## 并行算法实现源码
在实现过程中遇到的一个坑：在main函数中声明了一个用于存放结果的指针，
int* h_rootValid = NULL;将这个指针作为参数传递给findValidRootAndValid函数，作为findValidRootAndValid的输出值。要想在函数内修改数组的值，就需要传递指针的指针进去
size_t h_numValidElements = findValidRootAndValid(keys, vals, father, child, totalNum, numberOfMerge, numBlocks, numThreads, &h_rootValid, &h_SumValsValid);
在函数内取值的时候要用(*h_rootValid)[id]，括号一定不要忘记，因为 [ ] 的优先级比 * 高，如果写成* h_rootValid[id]会产生未知的结果。

```
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "cudpp.h"
#include "cudpp_hash.h"


__global__ void setKeyIndex(int* d_keyIndex, int N)
{
    const int tid = threadIdx.x;
    int globalid = blockIdx.x * blockDim.x + tid;
    if (globalid < N)
    {
        d_keyIndex[globalid] = globalid;
    }
}
__global__ void updateRoot(int* d_updateIndex, int* d_root, int* d_fatherroot, int N_update, bool* ischanged)
{
    const int tid = threadIdx.x;
    int globalid = blockIdx.x * blockDim.x + tid;
    if (globalid < N_update)
    {
        if (d_root[d_updateIndex[globalid]] > d_fatherroot[globalid])
        {
            *ischanged = true;
            atomicMin(&d_root[d_updateIndex[globalid]], d_fatherroot[globalid]);
        }
    }
}
__global__ void getSumVals(int* d_SumVals, int* d_rootIndex, int* d_vals, int N)
{
    const int tid = threadIdx.x;
    int globalid = blockIdx.x * blockDim.x + tid;
    if (globalid < N)
    {
        //d_root中存储的是key，我必须根据这个Key找到对应的位置索引
        atomicAdd(&d_SumVals[d_rootIndex[globalid]], d_vals[globalid]);
    }
}
__global__ void getIsvalid(int* d_SumVals, unsigned int* d_isValid, int N)
{
    const int tid = threadIdx.x;
    int globalid = blockIdx.x * blockDim.x + tid;
    if (globalid < N)
    {
        if (d_SumVals[globalid] > 0)
        {
            d_isValid[globalid] = 1;
        }
    }
}

size_t findValidRootAndValid(int* keys, int* vals, int* father, int* child, int totalNum, int numberOfMerge, int numBlocks, int numThreads,
              int** h_rootValid, int** h_SumValsValid)
{

    int *d_keys;
    cudaMalloc((void**)&d_keys, sizeof(int)* totalNum);
    cudaMemcpy(d_keys, keys, sizeof(int)* totalNum, cudaMemcpyHostToDevice);

    int* d_vals;
    cudaMalloc((void**)&d_vals, sizeof(int)* totalNum);
    cudaMemcpy(d_vals, vals, sizeof(int)* totalNum, cudaMemcpyHostToDevice);

    int *d_father = NULL;
    cudaMalloc((void**)&d_father, sizeof(int)* numberOfMerge);
    cudaMemcpy(d_father, father, sizeof(int)* numberOfMerge, cudaMemcpyHostToDevice);
    
    int *d_child = NULL;
    cudaMalloc((void**)&d_child, sizeof(int)* numberOfMerge);
    cudaMemcpy(d_child, child, sizeof(int)* numberOfMerge, cudaMemcpyHostToDevice);

    int *d_root = NULL;
    cudaMalloc((void**)&d_root, sizeof(int)* totalNum);
    cudaMemcpy(d_root, d_keys, sizeof(int)* totalNum, cudaMemcpyDeviceToDevice);

    int *d_fatherroot = NULL;
    cudaMalloc((void**)&d_fatherroot, sizeof(int)* numberOfMerge);
    cudaMemset(d_fatherroot, 0, sizeof(int)* numberOfMerge);
    
    int *d_childroot = NULL;
    cudaMalloc((void**)&d_childroot, sizeof(int)* numberOfMerge);
    cudaMemset(d_childroot, 0, sizeof(int)* numberOfMerge);

    int *d_keyIndex;
    cudaMalloc((void**)&d_keyIndex, sizeof(int)* totalNum);
    cudaMemset(d_keyIndex, 0, sizeof(int)* totalNum);

    int *d_updateIndex;
    cudaMalloc((void**)&d_updateIndex, sizeof(int)* numberOfMerge);
    cudaMemset(d_updateIndex, 0, sizeof(int)* numberOfMerge);

    // CUDPP初始化
    CUDPPHandle cudpp;
    cudppCreate(&cudpp);

    // 设定hashtable
    CUDPPHashTableConfig config;
    config.type = CUDPP_BASIC_HASH_TABLE;
    config.kInputSize = totalNum;
    config.space_usage = 2.0;

    CUDPPHandle hash_table_handle;
    cudppHashTable(cudpp, &hash_table_handle, &config);
    cudppHashInsert(hash_table_handle, d_keys, d_root, totalNum);

    CUDPPHandle cudppForFindIndex;
    cudppCreate(&cudppForFindIndex);
    CUDPPHandle hash_table_handleForFindIndex;
    cudppHashTable(cudppForFindIndex, &hash_table_handleForFindIndex, &config);
    setKeyIndex << <numBlocks, numThreads >> >(d_keyIndex, totalNum);
    cudppHashInsert(hash_table_handleForFindIndex, d_keys, d_keyIndex, totalNum);

    bool h_ischanged = false;
    bool *d_ischanged;
    cudaMalloc((void**)&d_ischanged, sizeof(bool));
    cudaMemcpy(d_ischanged, &h_ischanged, sizeof(bool), cudaMemcpyHostToDevice);
    
 bool h_hasdif = false;
    bool *d_hasdif;
    cudaMalloc((void**)&d_hasdif, sizeof(bool));
    cudaMemcpy(d_hasdif, &h_hasdif, sizeof(bool), cudaMemcpyHostToDevice);
    //找father_root和child_root，如果有不同，则进行下面的操作

    cudppHashRetrieve(hash_table_handle, d_father, d_fatherroot, totalNum); //在d_root中找到了father的root  d_fatherroot
    cudppHashRetrieve(hash_table_handle, d_child, d_childroot, totalNum);  //在d_root中找到了child的root  d_childroot
 hasDiff << <numBlocks, numThreads >> >(d_childroot, d_fatherroot, numberOfMerge, d_hasdif);    cudaMemcpy(&h_hasdif, d_hasdif, sizeof(bool), cudaMemcpyDeviceToHost);
    while(h_hasdif)
    {
 //初始化结束，接下来开始找每个key的根节点
    do{
        h_ischanged = false;
        cudaMemcpy(d_ischanged, &h_ischanged, sizeof(bool), cudaMemcpyHostToDevice);
        cudppHashRetrieve(hash_table_handle, d_father, d_fatherroot, totalNum); //在d_root中找到了father的root        d_fatherroot
        //在d_root中修改，将child的root修改为father的root
        //在d_keyIndex中找要修改的元素的索引，即找d_childroot的索引，也就是检测是否要修改的元素的索引；
        cudppHashRetrieve(hash_table_handleForFindIndex, d_child, d_updateIndex, numberOfMerge);
        updateRoot << <numBlocks, numThreads >> >(d_updateIndex, d_root, d_fatherroot, numberOfMerge, d_ischanged);
        cudaMemcpy(&h_ischanged, d_ischanged, sizeof(bool), cudaMemcpyDeviceToHost);
        if (h_ischanged)
            cudppHashInsert(hash_table_handle, d_keys, d_root, totalNum);
    } while (h_ischanged);

    do{
        h_ischanged = false;
        cudaMemcpy(d_ischanged, &h_ischanged, sizeof(bool), cudaMemcpyHostToDevice);
        cudppHashRetrieve(hash_table_handle, d_child, d_childroot, totalNum);       //在d_root中找到了child的root     d_childroot
        //在d_root中更新，按照大小对比，将father的root修改为child的root
        //在d_keyIndex中找要修改的元素的索引，即找d_fatherroot的索引，也就是检测是否要修改的元素的索引；
        cudppHashRetrieve(hash_table_handleForFindIndex, d_father, d_updateIndex, numberOfMerge);
        updateRoot << <numBlocks, numThreads >> >(d_updateIndex, d_root, d_childroot, numberOfMerge, d_ischanged);
        cudaMemcpy(&h_ischanged, d_ischanged, sizeof(bool), cudaMemcpyDeviceToHost);
        if (h_ischanged)
            cudppHashInsert(hash_table_handle, d_keys, d_root, totalNum);
    } while (h_ischanged);
      
   h_hasdif = false;
      cudaMemcpy(d_hasdif, &h_hasdif, sizeof(bool), cudaMemcpyHostToDevice);
      cudppHashRetrieve(hash_table_handle, d_father, d_fatherroot, totalNum); //在d_root中找到了father的root  d_fatherroot
      cudppHashRetrieve(hash_table_handle, d_child, d_childroot, totalNum);  //在d_root中找到了child的root  d_childroot


      hasDiff << <numBlocks, numThreads >> >(d_childroot, d_fatherroot, numberOfMerge, d_hasdif);
      cudaMemcpy(&h_hasdif, d_hasdif, sizeof(bool), cudaMemcpyDeviceToHost);


    }
    //开始求需要的元素，并compact输出===================================================================================
    unsigned int *d_isValid;
    cudaMalloc((void**)&d_isValid, sizeof(unsigned int)*totalNum);
    cudaMemset(d_isValid, 0, sizeof(unsigned int)* totalNum);

    size_t* d_numValid = NULL;
    cudaMalloc((void**)&d_numValid, sizeof(size_t));

    int* d_SumVals;
    cudaMalloc((void**)&d_SumVals, sizeof(int)* totalNum);
    cudaMemset(d_SumVals, 0, sizeof(int)* totalNum);


    int *d_rootIndex;
    cudaMalloc((void**)&d_rootIndex, sizeof(int)* totalNum);
    cudaMemset(d_rootIndex, 0, sizeof(int)* totalNum);
    cudppHashRetrieve(hash_table_handleForFindIndex, d_root, d_rootIndex, totalNum);        //在d_keyIndex中找到d_droot 所对应的key的索引,也就是要修改的元素的索引

    getSumVals << <numBlocks, numThreads >> >(d_SumVals, d_rootIndex, d_vals, totalNum);
    getIsvalid << <numBlocks, numThreads >> >(d_SumVals, d_isValid, totalNum);

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

    int *d_rootValid;
    cudaMalloc((void**)&d_rootValid, sizeof(int)*totalNum);
    cudaMemset(d_rootValid, 0, sizeof(int)* totalNum);
    int *d_SumValsValid;
    cudaMalloc((void**)&d_SumValsValid, sizeof(int)*totalNum);
    cudaMemset(d_SumValsValid, 0, sizeof(int)* totalNum);
    // Run the compact
    res = cudppCompact(compactplan, d_rootValid, d_numValid, d_root, d_isValid, totalNum);
    if (CUDPP_SUCCESS != res)
    {
        printf("Error in cudppScan()\n");
        exit(-1);
    }
    res = cudppCompact(compactplan, d_SumValsValid, d_numValid, d_SumVals, d_isValid, totalNum);
    if (CUDPP_SUCCESS != res)
    {
        printf("Error in cudppScan()\n");
        exit(-1);
    }
    size_t h_numValidElements = 0;
    cudaMemcpy(&h_numValidElements, d_numValid, sizeof(size_t), cudaMemcpyDeviceToHost);

    *h_rootValid = (int*)malloc(sizeof(int)*h_numValidElements);
    cudaMemcpy(*h_rootValid, d_rootValid, sizeof(int)*h_numValidElements, cudaMemcpyDeviceToHost);
    *h_SumValsValid = (int*)malloc(sizeof(int)*h_numValidElements);
    cudaMemcpy(*h_SumValsValid, d_SumValsValid, sizeof(int)*h_numValidElements, cudaMemcpyDeviceToHost);

    //clear
    cudppDestroyHashTable(cudpp, hash_table_handle);
    cudppDestroyHashTable(cudppForFindIndex, hash_table_handleForFindIndex);
    cudppDestroyHashTable(cudppForCompact, compactplan);

    cudppDestroy(cudpp);
    cudppDestroy(cudppForFindIndex);
    cudppDestroy(cudppForCompact);

    cudaFree(d_root);
    cudaFree(d_keys);
    cudaFree(d_vals);
    cudaFree(d_keyIndex);
    cudaFree(d_father);
    cudaFree(d_child);
    cudaFree(d_fatherroot);
    cudaFree(d_childroot);
    cudaFree(d_updateIndex);

    cudaFree(d_isValid);
    cudaFree(d_numValid);
    cudaFree(d_rootValid);
    cudaFree(d_SumValsValid);
    cudaFree(d_rootIndex);
    
    return h_numValidElements;
}
void initCUDA(int& devIdx)  //设置计算能力大于3.5的GPU
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
int main() {
  int gpuIdx = 1;
 initCUDA(gpuIdx);
    const int totalNum = 10;
    const int numberOfMerge = 4;
    int numThreads = 1024;
    int numBlocks = (totalNum + numThreads - 1) / numThreads;
    int keys[totalNum] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };//只关心1，2，3，4，5在merge后的值
    int vals[totalNum] = { 1, 6, 4, 9, 0, 3, 7, 2, 5, 8 };
    /*
    father: 1 2 3 4
    child:  3 4 5 5
    
    merge后应该输出
    keys:0  1  6 7 8 9
    vals:1 22  7 2 5 8
    */
    int father[4] = { 1, 2, 3, 4 };
    int child[4] =  { 3, 4, 5, 5 };
    int* h_rootValid = NULL;
    int* h_SumValsValid = NULL;
    size_t h_numValidElements = findValidRootAndValid(keys, vals, father, child, totalNum, numberOfMerge, numBlocks, numThreads, &h_rootValid, &h_SumValsValid);
    
    printf("numValidElements: %ld\n", h_numValidElements);
    int i = 0;
    for (i = 0; i < h_numValidElements; i++)
    {
        printf("%d\t", h_rootValid[i]);
    }
    printf("\n");
    for (i = 0; i < h_numValidElements; i++)
    {
        printf("%d\t", h_SumValsValid[i]);
    }
    printf("\n");
    free(h_rootValid);
    free(h_SumValsValid);
    return 0;
}
```

