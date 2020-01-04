---
title: 多节点MPI+CUDA的程序中GPU与进程的绑定
date: 2019-04-07 09:56:23
tags: [MPI, CUDA, MPI+CUDA]
---

### 问题描述
GPU是提升性能的强大工具，所以我们希望能够利用多GPU提升程序的效率，这样就可以实现MPI进程+CUDA轻量级线程的两层并行。NVIDIA的SLI(Scalable Link Interface)技术允许一个主机同时控制4个GPU，由于功耗和散热的限制，在一个系统上运行多个任务一直是一个挑战。然而，kernel的启动一次只能针对一个GPU，在多GPU系统中可以使用cudaSetDevice()函数来指定目标设备。如果我们有GPU集群，能否支持超过4个GPU呢？
解决方案是用MPI协调多个主机程序，每个主机程序控制一个或多个GPU。MPI用于在主机缓冲区间进行数据传输。现在对于部分NVIDIA设备已经支持MPI的识别，可以直接将设备指针传到MPI的函数中。但为了使得程序在任何集群上都能运行，在MPI不能识别CUDA的情况下，在调用MPI函数进行数据传输之前，将数据显式的拷贝到主机端，在目的进程接收之后在拷贝到设备端。

<!--more-->

在编写多GPU程序的时候，如何将MPI进程与节点上的GPU进行绑定呢？绑定时要考虑以下情况：有时一个节点上可能有多个GPU但是只有其中一部分是有效的，比如我们的程序需要计算能力大于3.5的GPU，这时有效GPU的编号可能并不连续。
注意：在运行多节点MPI程序时需要在每个节点的相同目录下存放运行程序所需要的数据，例如：scp -r /state/partition1/zhb/ root@compute-0-0:/state/partition1/，在不同节点之间用scp命令将数据拷贝到其他参与计算的节点。程序一般都是放在集群的共享分区中的，不需要拷贝。如果不知道文件的具体位置可以用类似这样的命令find -name cal*.tif来查找cal开头的.tif文件
### 解决方案
对MPI进程进行封装，实现与特定节点特定GPU的绑定。在写machinefile文件之前需要了解集群中每个节点的GPU设备情况。可以用nvidia-smi来查看。
![图片](1.png)
![图片](2.png)
![图片](3.png)
从上面的查询结果来看compute-0-0和hpscil都有2个可用的K40GPU，compute-0-11只有一个K40。跟据各节点的GPU设备情况写machinefile文件，假设下面是集群中的machinefile配置文件。
```
compute-0-0 slots=2
compute-0-11  slots=1
hpscil slots=2
```
接下来实现一个DeviceInfo类实现对GPU信息的查看和编号的获取。程序中设置计算能力大于3.0的为有效的GPU，getId()函数获取有效GPU的编号。
实现DeviceProcessor类。mnDevCount 是一个节点上的GPU的个数。mvValidDeviceId是vector数组，保存了一个节点上有效GPU的ID。
```
//DeviceInfo.h
#ifndef DEVICE_INFO_H
#define DEVICE_INFO_H
#include <string>
#include <vector>
#include <math.h>
#include "mpi.h"
#include <cuda_runtime.h>
#include <iostream>
using namespace std;

class DeviceInfo
{
public:
    DeviceInfo(int id)
    {
        mnId = id;
        cudaDeviceProp prop1;
        cudaGetDeviceProperties(&prop1,mnId);
        mnMultiProcessorCount = prop1.multiProcessorCount;
        mnComputeAblity = prop1.major;
    }
    int getId()
    {
        return mnId;
    }
    int getSMCount()
    {
        return mnMultiProcessorCount;
    }
    bool isValid()
    {
        if(mnMultiProcessorCount > 1 && mnComputeAblity >= 3)
            return true;
        else
            return false;
    }
private:
    int mnId;
    int mnMultiProcessorCount;
    int mnComputeAblity;
};
class DeviceProcessor
{
public:
    DeviceProcessor(string name)
    {
        cudaGetDeviceCount(&mnDevCount);
        mstrDeviceOnName = name;
        for(int i = 0;i < mnDevCount;i++)
        {
            if(DeviceInfo(i).isValid())
                mvValidDeviceId.push_back(i);
        }
    }
    DeviceProcessor()
    {
        cudaGetDeviceCount(&mnDevCount);
        for(int i = 0;i < mnDevCount;i++)
        {
            if(DeviceInfo(i).isValid())
                mvValidDeviceId.push_back(i);
        }
    }
    void setDeviceOn(string strname)
    {
        mstrDeviceOnName = strname;
    }
    vector<int> getAllDevice()
    {
        return mvValidDeviceId;
    }
    bool Valid()
    {
        // allDevice();
        if(mvValidDeviceId.empty())
        {
            cerr << __FILE__ << "  " << __FUNCTION__ << ":" << "there is no valid device" << endl;
            return false;
        }
        return true;
    }
    int setDevice()
    {
        if(mnDeviceId < mnDevCount)
        {
            cudaSetDevice(mnDeviceId);
            return 1;
        }
        else
        {
            cudaSetDevice(0);
            return -1;
        }
    }
    void setDeviceID(int nDevice)
    {
        mnDeviceId = nDevice;
    }

private:
    int mnDeviceId;
    int mnDevCount;
    string mstrDeviceOnName;
    vector<int> mvValidDeviceId;
};
#endif
```

对MPI进程进行封装，在Process类中包含一个DeviceProcessor对象。该类实现了initCUDA函数，用于实现进程与设备的绑定。下面对initCUDA()进行说明。
对于主进程来说，定义两个map，mapAvailableDevId用于存放当前节点可用的设备ID，mapProcId记录当前节点上要运行的进程号。子进程将他们的节点名称、可用设备数量、可用设备ID数组，发送给主进程，主进程在mapProcId中查找该节点，如果没有找到，那么将<节点名称，可用GPU编号数组>插入mapAvailableDevId，并将<节点名，进程编号vec>插入mapProcId。如果找到了，说明该节点可用的GPU编号已经有了，只需要将该进程编号插入mapProcId的进程数组即可。得到每个节点的运行进程数组和可用的设备ID数组后，主进程负责遍历进程数组，给每个进程分配一个设备，将设备编号发送给对应的进程。
对于子进程来说，将节点名称，该节点可用的设备数量和设备ID数组发送给主进程。然后等着接收主进程给分配的设备编号，调用setDeviceID()函数给进程绑定设备，再调用setDevice()函数真正绑定CUDA设备。
```
//Process.h
#include "mpi.h"
#include "DeviceInfo.h"
class Process
{
public:
    Process(MPI_Comm _comm);
    bool initialized() const;
    bool init(int argc = 0, char *argv[] = NULL);
    int getId() const;
    int getProcsNum() const;
    const MPI_Comm& getComm() const;
    const char* getProcessorName() const;
    bool isMaster() const;
    bool initCuda();
    
private:
    MPI_Comm m_mpiCommon;
    int mnPId;
    int mnTotalProcs;
    string mstrProcName;
    DeviceProcessor mDevice;
};
```
Process类的实现
```
//Process.cpp
#include "Process.h"
#include <map>
typedef vector<int> IntVect;

Process::Process(MPI_Comm _comm):m_mpiCommon(_comm), mnPId(-1), mnTotalProcs(-1) {}

bool Process::initialized() const
{
    int mpiStarted;
    MPI_Initialized(&mpiStarted);
    return static_cast<bool>(mpiStarted);   
}

bool Process::init(int argc, char *argv[])
{
    MPI_Comm_rank(m_mpiCommon, &mnPId);
    MPI_Comm_size(m_mpiCommon, &mnTotalProcs);


    char aPrcrName[MPI_MAX_PROCESSOR_NAME];
    int prcrNameLen;
    MPI_Get_processor_name(aPrcrName, &prcrNameLen);
    mstrProcName.assign(aPrcrName, aPrcrName + prcrNameLen);
    this->mDevice.setDeviceOn(mstrProcName);
    return initCuda();
}
int Process::getId() const
{
    return mnPId;
}
int Process::getProcsNum() const
{
    return mnTotalProcs;
}
const MPI_Comm& Process::getComm() const
{
    return m_mpiCommon;
}
const char* Process::getProcessorName() const
{
    return mstrProcName.c_str();
}
bool Process::isMaster() const
{
    return (mnPId == 0);
}
bool Process::initCuda()
{
    MPI_Status status;
    if(!mDevice.Valid() && !isMaster())
    {
        return false;
    }
    else
    {
        if(this->isMaster())
        {
            map<string, IntVect> mapAvailableDevId; //记录当前节点可用的设备ID
            map<string, IntVect> mapProcId;         //记录当前节点的进程号
            map<string, IntVect>::iterator iterProcId;
            map<string, IntVect>::iterator iterAvailableDevId;
            mapProcId.insert(pair<string,IntVect>(mstrProcName, IntVect()));
            mapProcId.begin()->second.push_back(0);
            mapAvailableDevId.insert(pair<string,IntVect>(mstrProcName, mDevice.getAllDevice()));
            for(int i = 1; i < mnTotalProcs; i++)
            {
                char _cName_recv[100];
                int _nDevCount_Recv;

                MPI_Recv(_cName_recv, 100, MPI_CHAR, i, i, m_mpiCommon, &status);
                MPI_Recv(&_nDevCount_Recv, 1, MPI_INT, i, 2*i, m_mpiCommon, &status);
                IntVect _vecAvailableDevId_Recv(_nDevCount_Recv);
                MPI_Recv(&_vecAvailableDevId_Recv[0], _nDevCount_Recv, MPI_INT, i, 3*i, m_mpiCommon, &status);
                string strName = _cName_recv;

                iterProcId = mapProcId.find(strName);
                if(iterProcId == mapProcId.end())
                {
                    IntVect nvecPocesses;
                    nvecPocesses.push_back(i);
                    mapProcId.insert(pair<string,IntVect>(strName, nvecPocesses));
                    nvecPocesses.clear();
                    mapAvailableDevId.insert(pair<string,IntVect>(strName, _vecAvailableDevId_Recv));
                }
                else
                {
                    iterProcId->second.push_back(i);
                }
            }
            iterAvailableDevId = mapAvailableDevId.begin();
            for(iterProcId = mapProcId.begin(); iterProcId!=mapProcId.end(); iterProcId++)
            {
                cout << iterProcId->first << "进程为:" << "\t";
                IntVect vnPoID = iterProcId->second;
                IntVect vnPoID2 = iterAvailableDevId->second;
                for(int i = 0; i < vnPoID.size(); i++)
                {
                    cout << vnPoID[i] << ",";
                    int nSend = i % vnPoID2.size();
                    if(vnPoID[i] != 0)
                    {
                        MPI_Send(&vnPoID2[nSend], 1, MPI_INT, vnPoID[i], 4 * vnPoID[i], m_mpiCommon);
                    }
                    else
                    {
                        mDevice.setDeviceID(vnPoID2[nSend]);
                    }
                }
                cout << endl;
                cout << iterProcId->first << "设备为:" << "\t";
                for(int i = 0; i < vnPoID2.size(); i++)
                {
                    cout << vnPoID2[i] << ",";
                }
                cout << endl;
                iterAvailableDevId++;
            }
        }
        else
        {
            const char *name = mstrProcName.c_str();
            MPI_Send(name, 100, MPI_CHAR, 0, mnPId, m_mpiCommon);
            int nCount = mDevice.getAllDevice().size();
            MPI_Send(&nCount, 1, MPI_INT, 0, 2 * mnPId, m_mpiCommon);
            MPI_Send(&mDevice.getAllDevice()[0], nCount, MPI_INT, 0, 3*mnPId, m_mpiCommon);
            int nRecv;
            MPI_Recv(&nRecv, 1, MPI_INT, 0, 4 * mnPId, m_mpiCommon, &status);
            cout << "进程" << mnPId << "的设备为:" << name << " " << nRecv << "\n"; 
            mDevice.setDeviceID(nRecv);
            mDevice.setDevice();
        }
        return true;
    }   
}
```
最后做个测试，实现数据的接力传递
```
//jieli.cpp
#include <stdio.h>
#include "mpi.h"
#include "Process.h"

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm comm;
    comm = MPI_COMM_WORLD;
    Process proc(comm);
    proc.init(argc, argv);
    int procSize = proc.getProcsNum();
    int pId = proc.getId();
    fprintf(stderr, "%d (%d)\n",pId,procSize);

    int value;
    MPI_Status status;
    char *filename = argv[1];

    if (proc.getId()==0) {
        fprintf(stderr, "\nPlease give new value=");
        scanf("%d",&value);
        fprintf(stderr, "%d read <-<- (%d)\n",pId,value);
        /*必须至少有两个进程的时候 才能进行数据传递*/
        if (procSize>1) {
            MPI_Send(&value, 1, MPI_INT, pId+1, 0, MPI_COMM_WORLD);
            fprintf(stderr, "%d send (%d)->-> %d\n", pId,value,pId+1);
        }
    }
    else {
        MPI_Recv(&value, 1, MPI_INT, pId-1, 0, MPI_COMM_WORLD, &status);
        fprintf(stderr, "%d receive(%d)<-<- %d\n",pId, value, pId-1);
        if (pId<procSize-1) {
            MPI_Send(&value, 1, MPI_INT, pId+1, 0, MPI_COMM_WORLD);
            fprintf(stderr, "%d send (%d)->-> %d\n", pId, value, pId+1);
        }
    }
    MPI_Finalize();
    return 0;
}
```
### 结果
编译命令：
mpic++ jieli.cpp Process.cpp -L /usr/local/cuda-8.0/lib64 -lcudart -I /usr/local/cuda-8.0/include

运行的时候用 mpirun -machinefile machinefile -n 6 ./a.out 就可以实现多节点多GPU的运算，在这个例子中没有真正的用到GPU做计算，只是证明了Process类编写正确。
下面是我的另外一个测试程序中GPU的分配情况，在该程序中开了5个进程，其中进程0为主进程，不参与运算，剩下的4个进程1,2,3,4,分别绑定一个节点上的一个GPU。
![图片](4.png)

