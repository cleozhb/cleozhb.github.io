---
title: MPI的进程组和通信域
date: 2019-03-26 09:55:57
tags: MPI 
---

### 概念
通信域是MPI的重要概念：*MPI的通信在通信域的控制和维护下进行 → 所有MPI通信任务都直接或间接用到通信域这一参数 → 对通信域的重组和划分可以方便实现任务的划分*
（1）**通信域（communicator）是一个综合的通信概念**。其包括上下文（context），进程组（group），虚拟处理器拓扑（topology）。其中进程组是比较重要的概念，表示通信域中所有进程的集合。一个通信域对应一个进程组。
 
（2）**进程（process）与进程组（group）的关系**。每个进程是客观上唯一的（一个进程对应一个pid号）；同一个进程可以属于多个进程组（每个进程在不同进程组中有个各自的rank号）；同一个进程可以属于不同的进程组，因此也可以属于不同的通信域。
 
（3）**通信域产生的方法**。根据看过的资料，大概有三种方法，先简要了解路子即可：
  a. 在已有通信域基础上划分获得：MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm)  
   b. 在已有通信域基础上复制获得：MPI_Comm_dup(MPI_Comm comm, MPI_Comm *newcomm)
     c. 在已有进程组的基础上创建获得：MPI_Comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm *newcomm)

<!--more-->

（4）**进程组产生的方法**。进程组（group）可以当成一个集合的概念，可以通过“子、交、并、补”各种方法。所有进程组产生的方法都可以套到集合的各种运算，用到的时候现看函数就可以了。
 
（5）“**当前进程**”与“**通信域产生函数**”。如果在已有进程组的基础上创建新的通信域（即（3）中c方法），则newcomm有两种结果：如果调用MPI_Comm_create的当前进程***在group中***，则newcomm就是新产生的通信域对象；如果调用MPI_Comm_create的当前进程***不在group中***，则newcomm就是MPI_COMM_NULL。由于MPI是多进程编程，类似“当前进程”与“通信域产生函数”这种情况会比较频繁的出现，在设计思路上要适应并行编程这种改变。
 
（6）**不同通信域间互不干扰**。“互不干扰”严格来说并不完全正确，这里想说的意思是：同一个进程，可以属于不同的通信域；同一个进程可以同时参与不同通信域的通信，互不干扰。
### 重要函数
MPI_Group_incl(MPI_Group group,int n,int *ranks,MPI_Group new_group)
input:
     Group     要被划分的进程组
     n             ranks数组中元素的个数
     ranks      将在新进程组中出现的旧进程组中的编号
output:
      new_group   由ranks定义的序号导出的新的进程组

MPI_Group_excl(MPI_Group group,int n,int *ranks,MPI_Group newgroup)
 input:
     Group     要被划分的进程组
     n             ranks数组中元素的个数
     ranks      将在新进程组中不出现的旧进程组中的编号
output:
      new_group   由ranks定义的序号导出的新的进程组
### 例子
下面通过一个例子来感受一下进程组和通信域在MPI多进程任务划分和处理上的应用。
代码做的事情如下：
（1）共有6个进程，在MPI_COMM_WORLD中的编号分别是{0，1，2，3，4，5}。
（2）将{1，3，5}进程形成一个新的通信域comm1；将编号为{0，2，4}的进程生成一个新的通信域comm2
（3）在comm1中执行MAX归约操作；在comm2中执行MIN归约操作；在MPI_COMM_WORLD中执行SUM归约操作
（4）显示各个通信域中归约操作的结果
具体代码如下：
```
//MPI_GROUP.c
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

#define LEN 5
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);


    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    int n = 3;
    const int ranks[3] = {1,3,5};
    const int ori1[1] = {1};
    const int ori2[1] = {0};
    int root1, root2;

    // 从world_group进程组中构造出来两个进程组
    MPI_Group group1, group2;
    MPI_Group_incl(world_group, n, ranks, &group1);
    MPI_Group_excl(world_group, n, ranks, &group2);
    // 根据group1 group2分别构造两个通信域
    MPI_Comm comm1, comm2;
    MPI_Comm_create(MPI_COMM_WORLD, group1, &comm1);
    MPI_Comm_create(MPI_COMM_WORLD, group2, &comm2);


    // 维护发送缓冲区和接受缓冲区
    int i;
    double *sbuf, *rbuf1, *rbuf2, *rbuf3;
    sbuf = (double *)malloc(LEN*sizeof(double));
    rbuf1 = (double *)malloc(LEN*sizeof(double));
    rbuf2 = (double *)malloc(LEN*sizeof(double));
    rbuf3 = (double *)malloc(LEN*sizeof(double));
    srand(world_rank*100);
    for(i=0; i<LEN; i++) sbuf[i] = (1.0*rand()) / RAND_MAX;
    fprintf(stderr,"rank %d:\t", world_rank);
    for(i=0; i<LEN; i++) fprintf(stderr,"%f\t",sbuf[i]);
    fprintf(stderr,"\n");
    MPI_Group_translate_ranks(world_group, 1, ori1, group1, &root1);
    MPI_Group_translate_ranks(world_group, 1, ori2, group2, &root2);
    // MPI_COMM_WORLD comm1 comm2分别执行不同的归约操作
    if (MPI_COMM_NULL!=comm1) { // comm1
        MPI_Reduce(sbuf, rbuf1, LEN, MPI_DOUBLE, MPI_MAX, root1, comm1);
        int rank_1;
        MPI_Comm_rank(comm1, &rank_1);
        if (root1==rank_1) {
            fprintf(stderr,"MAX:\t");
            for(i=0; i<LEN; i++) fprintf(stderr,"%f\t",rbuf1[i]);
            fprintf(stderr,"\n");
        }
    } 
    else if (MPI_COMM_NULL!=comm2) { // comm2
        MPI_Reduce(sbuf, rbuf2, LEN, MPI_DOUBLE, MPI_MIN, root2, comm2);
        int rank_2;
        MPI_Comm_rank(comm2, &rank_2);
        if (root2==rank_2) {
            fprintf(stderr,"MIN:\t");
            for(i=0; i<LEN; i++) fprintf(stderr,"%f\t",rbuf2[i]);
            fprintf(stderr,"\n");
        }
    }
    MPI_Reduce(sbuf, rbuf3, LEN, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); // MPI_COMM_WORLD 
    if (0==world_rank) {
        fprintf(stderr,"SUM:\t");
        for(i=0; i<LEN; i++) fprintf(stderr,"%f\t",rbuf3[i]);
        fprintf(stderr,"\n");
    }
    // 清理进程组和通信域
    if(MPI_GROUP_NULL!=group1) MPI_Group_free(&group1);
    if(MPI_GROUP_NULL!=group2) MPI_Group_free(&group2);
    if(MPI_COMM_NULL!=comm1) MPI_Comm_free(&comm1);
    if(MPI_COMM_NULL!=comm2) MPI_Comm_free(&comm2);
    MPI_Finalize();
}
```
![图片](1.png)
可以看到：
a. MIN归约操作针对的是{0，2，4}
b. MAX归约操作针对的是{1，3，5}
c. SUM归约操作针对的是{0，1，2，3，4，5}
d. SUM与MIN或MAX归约操作在时间上可能是重叠的，参与归约操作的进程也有重叠，但在结果上没有互相干扰。


### 参考
[https://www.cnblogs.com/xbf9xbf/p/5239094.html](https://www.cnblogs.com/xbf9xbf/p/5239094.html)
