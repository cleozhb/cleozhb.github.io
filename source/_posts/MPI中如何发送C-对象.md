---
title: MPI中如何发送C++对象
date: 2018-11-21 16:11:59
tags: [MPI, c]
---

现在考虑用MPI+CUDA实现多GPU同时计算景观指数。在编写代码的过程中遇到了一个问题就是如何将原先定义的dataBlock对象传递到子进程中去。
在做这个问题的过程中产生了以下几点疑问：        
1. 怎样发送带有指针的类，并在从进程中修改后再传回
2. 发送结构体（非连续数据类型，带有指针），发送过去以后类的函数是否可以正常调用。
3. 发送到从进程的结构体在从进程中修改后再传回主进程大小是否改变，因为有指针成员变量在从进程中被改变

<!--more-->

通过查看一些资料，知道上述疑问的提法是完全错误的，每个进程都有自己的地址空间，在一个进程上的有效地址很可能指向另一个进程上的无效内存。所以发送指针是没有意义的，必须发送数据本身。
MPI是消息传递接口，这里面的消息基本上是指原始的字节流。而C++对象是一个挺复杂的东西，不光代表了一块字节，还和一些函数（也就是代码）绑定在一起。所以，一般没有发送C++对象这种说法。你要先把对象转化成字节消息，然后发给另一个进程，然后另一个进程在这个消息的基础上重建这个对象。总之就是要先用protobuf或者Boost库将其序列化成字节数组。
* 可以用MPI_Type_struct()定义一个数据类型，就可以直接用MPI收发。
* 另一种方法就是将对象中的元素存到一个数组中，将数组作为整体发送，接收端根据数组中数据的顺序重新建立一个对象。
* 用Boost.MPI+BoostSerialization.不用考虑发送对象的大小，数据类型等。要先对类进行serialize，使用Boost的话，list和vector都可以直接发。

所以我现在的想法是将原先的dataBlock类拆开，控制每个分块大小的那一部分在主进程中定义，并且发送给从进程。从进程根据主进程传来的分块信息构建一个标记结构体。作为结果传回主进程。主进程根据传回来的信息重建类对象。
现在要做的是分别构造两个存储数据的结构，如果不方便的话就一个一个的进行传输。这个过程是比较麻烦的，要将原先的大类拆成小类，还要保证小类之间的对应关系。原先的大类将计算之前和计算之后的数据都放在一个类中了，如果用静态负载均衡的方法，可以让每个进程分别求计算前、后的数据，但是如果想采用动态负载均衡方法，就必须在主进程将任务进行分块，然后动态的派发，这样就必须将一个分块处理前的信息和处理后的结果分开保存。所以可以将原先的大类拆成如下两个小类。
还有一种方法是原先的大类不变，另外定义两个结构体，专用于传输数据，或者不定义结构体，采用一个一个传的方式，然后在接收方重建对象。也就是说，传的时候只是传对象的数据。在主从进程中都用接受到的数据重建对象。
为了尽量少的改动数据结构，减少代码的调整，我在程序中采用了最笨的一个一个传的方法，
```
//dataBlock
    int mnDataStart;
        int mnDataEnd;
        int mnTaskStart;
        int mnTaskEnd;
        int mnSubDataHeight;
        int mnSubTaskHeight;
        int mnStartTag;
        int mnWidth;
        int mnNodata;
```

```
//dataBlockResult
        int* mh_holoUp = NULL;                  //当前分块的上一行，若为第一行则一直保持NULL
        int* mh_holoDown = NULL;                //当前分块的下一行，若为最后一行则一直保持NULL
        int* mh_SubData = NULL;                 //保存当前分块的src信息
        
        int* mh_LabelVal = NULL;                //保存当前分块的标记值，从GPU传回的标记值
        int* mh_RelabelVal = NULL;              //保存当前分块的标记值,经过处理后的连续标记值
        int  mh_curPatchNum;                    //当前分块中斑块的数量
        
        int* mh_compactSrc = NULL;              //记录原始类型
        int* mh_compactLabel = NULL;            //标记值集合
        int* mh_compactRelabel = NULL;          //重标记值的集合
        int* mh_compactAreaByPixel = NULL;      //当前分块中每个斑块的面积（用像元数量表示）
        int* mh_compactPerimeterByPixel = NULL; //当前分块中每个斑块的周长（用像元为单位长度表示）
```

```
#include "basestruct.h"

#include <gdal_priv.h>
#include <cpl_conv.h>
#include "GDALRead.h"
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudaConfig.cuh"
#include "mpi.h"
#include <sys/stat.h>



#include "basestruct.h"
#include "utils.h"

void LineCCL(CuLSM::dataBlock& curBlock,
    dim3 blockSize, dim3 gridSize);
size_t findMerge(int* h_rowOneValue, int* h_rowTwoValue, int* h_rowOneLabel, int* h_rowTwoLabel, int pairNum, int cols,
    int** h_fatherValid, int** h_childValid);
int findMerge(int width, int BGvalue, int* Meg, int* h_subDataFirst, int *h_subDataSecond, int* lastRowLabel, int* firstRowLabel);

int findRoot(int* keys, int* father, int* child, int totalNum, int numberOfMerge, int** h_continueRoot);//相当于main函数

void mergePatch(int width, int blockNum, int BGvalue, int* h_rowOneValue, int* h_rowTwoValue, int* h_rowOneLabel, int* h_rowTwoLabel, CuLSM::UnionFind *Quf);

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
int getDevideInfo(int gpuIdx, int width, int height, int nodata, CuLSM::dataBlock** dataBlockArray)
{
    initCUDA(gpuIdx);
    int maxnum;     //可以读入的像元的个数
    size_t freeGPU, totalGPU;
    cudaMemGetInfo(&freeGPU, &totalGPU);//size_t* free, size_t* total
    cout << "(free,total)" << freeGPU << "," << totalGPU << endl;

    maxnum = (freeGPU) / (sizeof(int)* 10);//每个pixel基本上要开辟6个中间变量，变量类型都是int
    // maxnum = (freeGPU) / (sizeof(int)* 6 * 2);//每个pixel基本上要开辟6个中间变量，变量类型都是int
    int sub_height = maxnum / width - 5;    //每个分块的高度sub_height
    //sub_height = 2;
    int blockNum = height / sub_height + 1; //总的分块个数

    //*dataBlockArray = new CuLSM::dataBlock[blockNum];
    *dataBlockArray = (CuLSM::dataBlock*)malloc(blockNum*sizeof(CuLSM::dataBlock));

    int subIdx = 0;
    for (int height_all = 0; height_all < height; height_all += sub_height)
    {
        int task_start = subIdx*sub_height;
        int task_end;
        if ((subIdx + 1)*sub_height - height <= 0)
            task_end = (subIdx + 1)*sub_height - 1;
        else
            task_end = height - 1;
        int data_start, data_end;
        if (task_start - 1 <= 0)
            data_start = 0;
        else
            data_start = task_start - 1;
        if (task_end + 1 >= height - 1)
            data_end = height - 1;
        else
            data_end = task_end + 1;
        int data_height = data_end - data_start + 1;
        int task_height = task_end - task_start + 1;

        (*dataBlockArray)[subIdx].mnDataStart = data_start;
        (*dataBlockArray)[subIdx].mnDataEnd = data_end;
        (*dataBlockArray)[subIdx].mnTaskStart = task_start;
        (*dataBlockArray)[subIdx].mnTaskEnd = task_end;
        (*dataBlockArray)[subIdx].mnSubTaskHeight = task_height;
        (*dataBlockArray)[subIdx].mnSubDataHeight = data_height;
        (*dataBlockArray)[subIdx].mnStartTag = task_start*width;//当前分块的起始标记值，也就是该分块的第一个栅格的一维索引值
        (*dataBlockArray)[subIdx].mnWidth = width;
        (*dataBlockArray)[subIdx].mnNodata = nodata;

        subIdx++;
    }
    return blockNum;
}

void recordBoundary(CuLSM::dataBlock &curBlock, int iBlock, int width,
    int** vecOriginValRow1, int** vecOriginValRow2, int** vecLabelValRow1, int** vecLabelValRow2);


MPI_Datatype dataBlock_MPI;


void getCCLmpi_manager(int commSize, MPI_Comm comm, CGDALRead* pread,
    CuLSM::dataBlock** dataBlockArray,CuLSM::UnionFind* Quf, int *blockNum)
{
    int width = pread->cols();
    int height = pread->rows();
    int nodata = (int)pread->invalidValue();
    CuLSM::dataBlock *dataBlockArray = NULL;
    //用一个GPU得到总的分块个数 blockNums
    int blockNums = getDevideInfo(1, width, height, nodata, dataBlockArray);
    /*
    *blockNums是总共的分块个数
    *每个分块的起始信息都被保存在了dataBlockArray中，需要发送给子进程进行处理
    *同时发送blockId作为是否处理完的标识
    *子进程处理完了之后将dataBlock对象传回    
    */
    int blockId = 0;
    int *power;
    
    //先接收从子进程传回来的有效元素的数量，在主进程分配内存空间，然后才能接收从子进程传回来的数组
    int getValidNum = -1;

    MPI_Status status;
    int terminate = 0;//终止主进程的标识

    power = (int*)malloc(sizeof(int)* 2);
    memset(power, 0, sizeof(int)* 2);
    vector<CuLSM::Patch> vecAllLabel;//用于记录不重复的label
    //记录边界holo区域的原始数据和标记值，用于合并斑块用的
    //blockNum个分块，总共有blockNum-1个交界
    int* vecOriginValRow1 = (int*)malloc(sizeof(int)* width * (*blockNum - 1));
    int* vecOriginValRow2 = (int*)malloc(sizeof(int)* width * (*blockNum - 1));
    int* vecLabelValRow1 = (int*)malloc(sizeof(int)* width * (*blockNum - 1));
    int* vecLabelValRow2 = (int*)malloc(sizeof(int)* width * (*blockNum - 1));
    checkMemAlloc(vecOriginValRow1);
    checkMemAlloc(vecOriginValRow2);
    checkMemAlloc(vecLabelValRow1);
    checkMemAlloc(vecLabelValRow2);

    do
    {
        printf("------------------------ I am manager ------------------------\n");
        MPI_Recv(power, 2, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);   // Manager 接收到Worker的请求信息
        printf("I received the request from process :%d\n", power[1]);

        if ((power[0] == 0) && (blockId < blockNums))
        {
            //怎样发送带有指针的类，并在从进程中修改后再传回
            //问题1：发送结构体（非连续数据类型，带有指针），发送过去以后类的函数是否可以正常调用。
            //发送到从进程的结构体在从进程中修改后再传回主进程大小是否改变，因为有指针成员变量在从进程中被改变
            /*
            这种想法是完全错误的，每个进程都有自己的地址空间，在一个进程上的有效地址很可能指向另一个进程上的无效内存
            所以发送指针是没有意义的，必须发送数据本身，参考
            https://stackoverflow.com/questions/10419990/creating-an-mpi-datatype-for-a-structure-containing-pointers/10421892#10421892
            */
            
            MPI_Send(&blockId, 1, MPI_INT, power[1], 0, comm);//发送当前处理的block的序号，tag 0
            
            //////////////////////////////////////////////////////////////////////////////////////////////
            //发送分块信息，tag 80-88, 这个分块结构体是从dataBlock中拆出来的
            //////////////////////////////////////////////////////////////////////////////////////////////
            CuLSM::dataBlock curBlock = (*dataBlockArray)[blockId];
            int curDataStart = curBlock.mnDataStart;
            int curDataEnd = curBlock.mnDataEnd;
            int curTaskStart = curBlock.mnTaskStart;
            int curTaskEnd = curBlock.mnTaskEnd;
            int curSubDataHeight = curBlock.mnSubDataHeight;
            int curSubTaskHeight = curBlock.mnSubTaskHeight;
            int curStartTag = curBlock.mnStartTag;
            int curWidth = curBlock.mnWidth;
            int curNodata = curBlock.mnNodata;
            MPI_Send(&curDataStart, 1, MPI_INT, power[1], 80, comm);
            MPI_Send(&curDataEnd, 1, MPI_INT, power[1], 81, comm);
            MPI_Send(&curTaskStart, 1, MPI_INT, power[1], 82, comm);
            MPI_Send(&curTaskEnd, 1, MPI_INT, power[1], 83, comm);
            MPI_Send(&curSubDataHeight, 1, MPI_INT, power[1], 84, comm);
            MPI_Send(&curSubTaskHeight, 1, MPI_INT, power[1], 85, comm);
            MPI_Send(&curStartTag, 1, MPI_INT, power[1], 86, comm);
            MPI_Send(&curWidth, 1, MPI_INT, power[1], 87, comm);
            MPI_Send(&curNodata, 1, MPI_INT, power[1], 88, comm);

            printf("I sent the block%d to the process :%d\n", blockId, power[1]);
            blockId++;

            //////////////////////////////////////////////////////////////////////////////////////////////
            //接受从进程返回的结果数组tag 100.
            //先接收有效元素的个数，并在主进程上分配合适大小的空间
            //////////////////////////////////////////////////////////////////////////////////////////////
            MPI_Recv(&curBlock.mh_curPatchNum, 1, MPI_INT, power[1], 94, comm, &status);
            if (curBlock.mh_curPatchNum > 0)
            {
                //////////////////////////////////////////////////////////////////////////////////////////////
                //分配空间
                //////////////////////////////////////////////////////////////////////////////////////////////
                curBlock.mh_holoUp = (int*)malloc(sizeof(int)* width);
                curBlock.mh_holoDown = (int*)malloc(sizeof(int)* width);
                curBlock.mh_SubData = (int*)malloc(sizeof(int)* width * curSubTaskHeight);
                curBlock.mh_LabelVal = (int*)malloc(sizeof(int)* width * curSubTaskHeight);

                curBlock.mh_compactSrc = (int*)malloc(sizeof(int)* curBlock.mh_curPatchNum);
                curBlock.mh_compactLabel = (int*)malloc(sizeof(int)* curBlock.mh_curPatchNum);
                curBlock.mh_compactAreaByPixel = (int*)malloc(sizeof(int)* curBlock.mh_curPatchNum);
                curBlock.mh_compactPerimeterByPixel = (int*)malloc(sizeof(int)* curBlock.mh_curPatchNum);

                getValidNum = 1;//可以发送有效数组给我了！
                MPI_Send(&getValidNum, 1, MPI_INT, power[1], 100, comm);
                
                //////////////////////////////////////////////////////////////////////////////////////////////
                //接收结果
                //////////////////////////////////////////////////////////////////////////////////////////////
                MPI_Recv(curBlock.mh_holoUp, width, MPI_INT, power[1], 90, comm, &status);
                MPI_Recv(curBlock.mh_holoDown, width, MPI_INT, power[1], 91, comm, &status);
                MPI_Recv(curBlock.mh_SubData, width * curSubTaskHeight, MPI_INT, power[1], 92, comm, &status);
                MPI_Recv(curBlock.mh_LabelVal, width * curSubTaskHeight, MPI_INT, power[1], 93, comm, &status);

                MPI_Recv(curBlock.mh_compactSrc, curBlock.mh_curPatchNum, MPI_INT, power[1], 95, comm, &status);
                MPI_Recv(curBlock.mh_compactLabel, curBlock.mh_curPatchNum, MPI_INT, power[1], 96, comm, &status);
                MPI_Recv(curBlock.mh_compactAreaByPixel, curBlock.mh_curPatchNum, MPI_INT, power[1], 97, comm, &status);
                MPI_Recv(curBlock.mh_compactPerimeterByPixel, curBlock.mh_curPatchNum, MPI_INT, power[1], 98, comm, &status);
            }
            
            for (int i = 0; i < curBlock.mh_curPatchNum; i++)
            {
                CuLSM::Patch temp;
                temp.nLabel = curBlock.mh_compactLabel[i];
                temp.nType = curBlock.mh_compactSrc[i];
                temp.nAreaByPixel = curBlock.mh_compactAreaByPixel[i];
                temp.nPerimeterByPixel = curBlock.mh_compactPerimeterByPixel[i];
                temp.isUseful = false;
                vecAllLabel.push_back(temp);
            }
            
            //记录分块的边界信息
            recordBoundary(curBlock, blockId, width, &vecOriginValRow1, &vecOriginValRow2, &vecLabelValRow1, &vecLabelValRow2);

            curBlock.freeSubData();
            curBlock.freeHoloUp();
            curBlock.freeHoloDown();

        }
        if (blockId > blockNums - 1)
        {
            printf("Now no block need process , send terminate sign to process:%d\n", power[1]);
            blockId = -1;
            MPI_Send(&blockId, 1, MPI_INT, power[1], 0, comm);
        }

        /******************** worker处理完发送处理完的信号**********************/
        if (power[0] == 1)
        {
            terminate++;
        }
        printf("------------------------------------------------\n\n\n");
    } while (terminate < (commSize - 1));
    
    //合并斑块的操作只能在主进程上执行
    Quf->initUF(vecAllLabel);
    mergePatch(width, *blockNum, (int)pread->invalidValue(), vecOriginValRow1, vecOriginValRow2, vecLabelValRow1, vecLabelValRow2, Quf);
    free(vecOriginValRow1);
    free(vecOriginValRow2);
    free(vecLabelValRow1);
    free(vecLabelValRow2);
    printf("------------------------ The all work have done ---------------------------\n");

}


void getCCLmpi_worker(int commRank, MPI_Comm comm, CGDALRead* pread)
{
    //将一个进程与一块GPU绑定
    int gpuIdx = commRank - 1;
    initCUDA(gpuIdx);


    int *power;
    MPI_Status status;
    power = (int*)malloc(sizeof(int)* 2);
    memset(power, 0, sizeof(int)* 2);
    
    int blockId; //用于表示当前处理的block的序号，当接收的blockId = -1时，表示没有分开要处理了，发送结束标志给主进程
    power[0] = 0;
    power[1] = commRank;
    int getValidNum = -1;
    //记录当前分块的信息
    CuLSM::dataBlock curBlock;
    
    int curDataStart;
    int curDataEnd;
    int curTaskStart;
    int curTaskEnd;
    int curSubDataHeight;
    int curSubTaskHeight;
    int curStartTag;
    int curWidth;
    int curNodata;

    for (;;)
    {
        printf("-----------------------I am worker : %d -------------------------\n", power[1]);
        MPI_Send(power, 2, MPI_INT, 0, commRank, MPI_COMM_WORLD);     // 给Manager发送请求信息 --- tag commRank
        MPI_Recv(&blockId, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);// 接受当前处理的Block的序号，blockId --- tag 0
        if (blockId == -1)
        {
            printf("I have not received from Manager , shoud be terminated .\n");
            power[0] = 1;
            MPI_Send(power, 2, MPI_INT, 0, commRank, MPI_COMM_WORLD);
            break;
        }
        else
        {
            printf("I received the block%d from Manager.\n", blockId);
            //////////////////////////////////////////////////////////////////////////////////////////////
            //接收分块信息，tag 80-88，然后做相应的处理，并生成结果结构体 dataBlockAfterCCL，并传回主机端
            //////////////////////////////////////////////////////////////////////////////////////////////
            MPI_Recv(&curDataStart, 1, MPI_INT, 0, 80, comm, &status);
            MPI_Recv(&curDataEnd, 1, MPI_INT, 0, 81, comm, &status);
            MPI_Recv(&curTaskStart, 1, MPI_INT, 0, 82, comm, &status);
            MPI_Recv(&curTaskEnd, 1, MPI_INT, 0, 83, comm, &status);
            MPI_Recv(&curSubDataHeight, 1, MPI_INT, 0, 84, comm, &status);
            MPI_Recv(&curSubTaskHeight, 1, MPI_INT, 0, 85, comm, &status);
            MPI_Recv(&curStartTag, 1, MPI_INT, 0, 86, comm, &status);
            MPI_Recv(&curWidth, 1, MPI_INT, 0, 87, comm, &status);
            MPI_Recv(&curNodata, 1, MPI_INT, 0, 88, comm, &status);
            //重构对象
            curBlock.mnDataStart = curDataStart;
            curBlock.mnDataEnd = curDataEnd;
            curBlock.mnTaskStart = curTaskStart;
            curBlock.mnTaskEnd = curTaskEnd;
            curBlock.mnSubDataHeight = curSubDataHeight;
            curBlock.mnSubTaskHeight = curSubTaskHeight;
            curBlock.mnStartTag = curStartTag;
            curBlock.mnWidth = curWidth;
            curBlock.mnNodata = curNodata;

            int width = curWidth;

            //all node should have this two variable
            dim3 blockDim1 = cudaConfig::getBlock2D();
            dim3 gridDim1 = cudaConfig::getGrid(curWidth, curSubTaskHeight);

            //////////////////////////////////////////////////////////////////////////////////////////////
            //processing...
            //////////////////////////////////////////////////////////////////////////////////////////////
            curBlock.loadBlockData(pread);
            checkMemAlloc(curBlock.mh_SubData);
            LineCCL(curBlock, blockDim1, gridDim1);

            //////////////////////////////////////////////////////////////////////////////////////////////
            //传回结果
            //发送dataBlockAfterCCL（当前分块的结果），并传回主机端，tag 90-98.
            //////////////////////////////////////////////////////////////////////////////////////////////
            MPI_Send(&curBlock.mh_curPatchNum, 1, MPI_INT, 0, 94, comm);
            MPI_Recv(&getValidNum, 1, MPI_INT, 0, 100, comm, &status);
            if (getValidNum)
            {
                MPI_Send(curBlock.mh_holoUp, width, MPI_INT, 0, 90, comm);
                MPI_Send(curBlock.mh_holoDown, width, MPI_INT, 0, 91, comm);
                MPI_Send(curBlock.mh_SubData, width * curSubTaskHeight, MPI_INT, 0, 92, comm);

                MPI_Send(curBlock.mh_LabelVal, width * curSubTaskHeight, MPI_INT, 0, 93, comm);

                MPI_Send(curBlock.mh_compactSrc, curBlock.mh_curPatchNum, MPI_INT, 0, 95, comm);
                MPI_Send(curBlock.mh_compactLabel, curBlock.mh_curPatchNum, MPI_INT, 0, 96, comm);
                MPI_Send(curBlock.mh_compactAreaByPixel, curBlock.mh_curPatchNum, MPI_INT, 0, 97, comm);
                MPI_Send(curBlock.mh_compactPerimeterByPixel, curBlock.mh_curPatchNum, MPI_INT, 0, 98, comm);
            }

        }
        printf("-----------------------------------------------------------------\n\n\n");
    }
}

void getCCLmpi(int argc, char *argv[], const char *filename,
    CuLSM::dataBlock** dataBlockArray, CuLSM::UnionFind* Quf, int *blockNum)
{
    // Initialize GDAL
    GDALAllRegister();
    CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");
    CGDALRead* pread = new CGDALRead;
    if (!pread->loadMetaData(filename))
    {
        cout << "load error!" << endl;
    }
    cout << "rows:" << pread->rows() << endl;
    cout << "cols:" << pread->cols() << endl;
    cout << "bandnum:" << pread->bandnum() << endl;
    cout << "datalength:" << pread->datalength() << endl;
    cout << "invalidValue:" << pread->invalidValue() << endl;
    cout << "datatype:" << GDALGetDataTypeName(pread->datatype()) << endl;
    cout << "projectionRef:" << pread->projectionRef() << endl;
    cout << "perPixelSize:" << pread->perPixelSize() << endl;

    // Initialize MPI state
    MPI_Init(&argc, &argv);

    // Get our MPI node number and node count
    MPI_Comm comm;
    comm = MPI_COMM_WORLD;

    int commSize, commRank;
    MPI_Comm_size(comm, &commSize);
    MPI_Comm_rank(comm, &commRank);
    if (commRank == 0)
    {
        getCCLmpi_manager(commSize,comm, pread, dataBlockArray, Quf, blockNum);
    }
    else
    {
        getCCLmpi_worker(commRank, comm, pread);
    }
}

```
### 参考
* [https://stackoverflow.com/questions/10419990/creating-an-mpi-datatype-for-a-structure-containing-pointers/10421892#10421892](https://stackoverflow.com/questions/10419990/creating-an-mpi-datatype-for-a-structure-containing-pointers/10421892#10421892)
* MPI中如何发送C++对象？[https://www.zhihu.com/question/25088675](https://www.zhihu.com/question/25088675)

