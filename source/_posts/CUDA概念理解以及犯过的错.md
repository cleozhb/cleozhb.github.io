---
title: CUDA概念理解以及犯过的错
date: 2018-09-16 12:35:59
tags: CUDA 
---

首先要记录的就是这次改bug的过程，Program received signal CUDA_EXCEPTION_14, Warp Illegal Address.错误的原因是CUDA访问越界。要记住一个点__syncthreads()函数仅仅能够用于线程块内的线程同步，不能用于全局所有线程块的同步。我这次犯的错就是在一个核函数内部试图构造一个全局数组，然后接下来的操作用到此全局数组的值。

<!--more-->

```
//错误的写法
__global__ void ReLabelEachPixel(int* d_label, int* d_RootPos, int* d_IsRoot, int curPatchNum, int labelStart, int width, int task_height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int gid = x + y * width;//global 1D index;
    
    if (gid < curPatchNum)
    {
        d_IsRoot[d_RootPos[gid]] = 1;
    }   
    __syncthreads();
    //判断哪些节点是根节点，是根节点的pixel不需要重标记
    bool limits = x < width && y < task_height;
    if (limits)
    {
        int center = d_label[gid];
        if(center!= NO_USE_CLASS)
        {
            if(!d_IsRoot[gid])//如果当前pixel不是根，更新其为根节点的值
            {
                d_label[gid] -= labelStart;
                d_label[gid] = d_label[d_label[gid]];
            }
        }
    }
}
```
在上面错误的版本中，在一个核函数内部构造全局数组，然后接下来的操作用到此全局数组的值，这两步中间用了__syncthreads()函数同步。这样的同步并不能保证d_IsRoot中所有线程块负责的像元都初始化完毕，所以会出错。
结论：__syncthreads()函数仅仅能够用于线程块内的线程同步，不能用于全局所有线程块的同步，所有线程块的同步必须用两个核函数来完成。两个核函数是串行执行的，相当于中间有个阻塞。正确的应该将上面的改为如下两个核函数。
```
//正确的版本
__global__ void Set_d_IsRoot(int* d_RootPos, int* d_IsRoot, int curPatchNum, int width)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int gid = x + y * width;//global 1D index;
    //判断哪些节点是根节点，是根节点的pixel不需要重标记
    if (gid < curPatchNum)
    {
        d_IsRoot[d_RootPos[gid]] = 1;
    }   
}
__global__ void ReLabelEachPixel(int* d_label, int* d_IsRoot, int labelStart, int width, int task_height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int gid = x + y * width;//global 1D index;
    //判断哪些节点是根节点，是根节点的pixel不需要重标记

    bool limits = x < width && y < task_height;
    if (limits)
    {
        int center = d_label[gid];
        if(center!= NO_USE_CLASS)
        {
            if(!d_IsRoot[gid])//如果当前pixel不是根，更新其为根节点的值
            {
                d_label[gid] -= labelStart;
                d_label[gid] = d_label[d_label[gid]];
            }
        }
    }
}
```

    并行编程的中心思想是分而治之：将大问题划分为一些小问题，再把这些小问题交给相应的处理单元并行地进行处理。在   *CUDA*  中，这一思想便体现在  *Grid, Block, Thread*   等层次划分上。
### CUDA执行模型
一个*Thread*被执行过程：
    *Grid*在*GPU*上启动；
    *block*被分配到*SM*上；
    *SM*把线程组织为*warp*；
    *SM*调度执行*warp*；
    执行结束后释放资源；
    *block*继续被分配*....*

* sp : streaming processor 最基本的处理单元，最后具体的指令都是在sp上进行处理的，GPU进行并行计算也就是多个sp同时做处理。
* sm : streaming multiprocessor 多个sp加上一些其他的资源组成一个sm。其他的资源就是存储资源，共享内存，寄存器等
* warp:GPU执行程序时的调度单位，目前cuda的warp大小为32,在同一个warp 的线程，以不同的数据执行相同的指令

![图片](1.png)![图片](2.png)
一个sm只会执行一个block中的warp，当一个block中的warp执行完，才会执行其他block中的warp。进行划分时，保证每个block中的warp比较合理，可以让sm交替执行里面的warp。此外，在分配block时，要根据GPU的sm个数，分配出合理的block数，让GPU的sm都利用起来，提高利用率。分配时，也要考虑到同一个线程block的资源问题，不要出现对应的资源不够。
GPU线程以网格（grid）的方式组织，而每个网格中又包含若干个线程块，在G80/GT200系列中，每一个线程块最多可包含512个线程，Fermi架构中每个线程块支持高达1536个线程。同一线程块中的众多线程拥有相同的指令地址，不仅能够并行执行，而且能够通过共享存储器（Shared memory）和栅栏（barrier）实现块内通信。这样，同一网格内的不同块之间存在不需要通信的粗粒度并行，而一个块内的线程之间又形成了允许通信的细粒度并行。这些就是CUDA的关键特性：线程按照粗粒度的线程块和细粒度的线程两个层次进行组织、在细粒度并行的层次通过共享存储器和栅栏同步实现通信，这就是CUDA的双层线程模型。
       在执行时，GPU的任务分配单元（global block scheduler）将网格分配到GPU芯片上。启动CUDA 内核时，需要将网格信息从CPU传输到GPU。任务分配单元根据这些信息将块分配到SM上。任务分配单元使用的是轮询策略：轮询查看SM是否还有足够的资源来执行新的块，如果有则给SM分配一个新的块，如果没有则查看下一个SM。决定能否分配的因素有：每个块使用的共享存储器数量，每个块使用的寄存器数量，以及其它的一些限制条件。任务分配单元在SM的任务分配中保持平衡，但是程序员可以通过更改块内线程数，每个线程使用的寄存器数和共享存储器数来隐式的控制，从而保证SM之间的任务均衡。任务以这种方式划分能够使程序获得了可扩展性：由于每个子问题都能在任意一个SM上运行，CUDA程序在核心数量不同的处理器上都能正常运行，这样就隐藏了硬件差异。
       对于程序员来说，他们需要将任务划分为互不相干的粗粒度子问题(最好是易并行计算)，再将每个子问题划分为能够使用线程处理的问题。同一线程块中的线程开始于相同的指令地址，理论上能够以不同的分支执行。但实际上，在块内的分支因为SM构架的原因被大大限制了。内核函数实质上是以块为单位执行的。同一线程块中的线程需要SM中的共享存储器共享数据，因此它们必须在同一个SM中发射。线程块中的每一个线程被发射到一个SP上。任务分配单元可以为每个SM分配最多8个块。而SM中的线程调度单元又将分配到的块进行细分，将其中的线程组织成更小的结构，称为线程束（warp）。在CUDA中，warp对程序员来说是透明的，它的大小可能会随着硬件的发展发生变化，在当前版本的CUDA中，每个warp是由32个线程组成的。SM中一条指令的延迟最小为4个指令周期。8个SP采用了发射一次指令，执行4次的流水线结构。所以由32个线程组成的Warp是CUDA程序执行的最小单位，并且同一个warp是严格串行的，因此在warp内是无须同步的。在一个SM中可能同时有来自不同块的warp。当一个块中的warp在进行访存或者同步等高延迟操作时，另一个块可以占用SM中的计算资源。这样，在SM内就实现了简单的乱序执行。不同块之间的执行没有顺序，完全并行。无论是在一次只能处理一个线程块的GPU上,还是在一次能处理数十乃至上百个线程块的GPU上，这一模型都能很好的适用。
例如GTX760, 6SM， 192SP/SM，一个SM一次执行一个Block，假设一个Warp含32个Thread,一个Block线程数量应该远远大于192(6warp)，为的是GPU执行长延时操作。（CUDA处理器需要高效地执行长延时操作，如果warp中的线程执行一个条指令需要等待前面启动的长延时操作的结果，那么不会选择执行该warp，而是选择执行另一个不用等待结果的驻留的warp，这样，如果有了多个warp准备执行，则总可以选择不产生延时的线程先执行，达到所谓的延时隐藏。）

![图片](3.png)
       目前，某一时刻只能有一个内核函数正在执行，但是在Fermi架构中，这一限制已被解除。如果在一个内核访问数据时，另一个内核能够进行计算，则可以有效的提高设备的利用率。
       每一个块内线程数应该首先是32的倍数，因为这样的话可以适应每一个warp包含32个线程的要求，每一个warp中串行执行，这就要求每一个线程中不可以有过多的循环或者需要的资源过多。但是每一个块中如果线程数过多，可能由于线程中参数过多带来存储器要求过大，从而使SM处理的效率更低。所以，在函数不是很复杂的情况下，可以适当的增加线程数目，线程中不要加入循环。在函数比较复杂的情况下，每一个块中分配32或是64个线程比较合适。每一个SM同时处理一个块，只有在粗粒度层面上以及细粒度层面上均达到平衡，才能使得GPU的利用到达最大。我用的显卡为GeForce GTX560 Ti，每一个网格中允许的最大块数位65535个，而每个块中的线程数为1024个，所以说粗粒度平衡对于我来说影响比较小，就细粒度来说，每一个块中的线程数以及每一个线程中的循环就变得至关重要了。
### CUDA编程模型
CUDA函数类型
* __device__     执行于Device，仅能从Device调用。限制，不能用&取地址；不支持递归；不支持static variable；不支持可变长度参数
* __global__      void： 执行于Device，仅能从Host调用。此类函数必须返回void
* __host__         执行于Host，仅能从Host调用，是函数的默认类型 

thread,block,grid是CUDA编程中的概念，用来组织GPU线程。在启动CUDA核函数时要指定gridsize和blocksize。假设有如下线程块配置：
  dim3 gridsize(2,2);
  dim3 blocksize(4,4);

grid中的blockidx序号标注情况为
                                                                   ![图片](4.png)
block中的threadidx序号标注情况
                                                      ![图片](5.png)
![图片](6.png)![图片](7.png)
![图片](8.png)
一个一维的block（此处只有x维度上存在16个线程）。所以，內建变量只有一个在起作用，就是threadIdx.x，它的范围是[0,15]。因此，我们在计算线程索引是，只用这个內建变量就行了（其他的为0，写了也不起作用）


![图片](9.png)
这个线程格只有一个一维的线程块，该线程块内的线程是二维的


![图片](10.png)



![图片](11.png)
一个grid和16个block，每个block都是一维



![图片](12.png)![图片](13.png)![图片](14.png)
### CUDA内存模型
每个 thread都有自己的一份 register 和 local memory的空间。一组thread构成一个 block，这些thread则共享有一份shared memory。__syncthreads()可以同步一个Block内的所有线程，不同Block内的Thread不能同步。此外，所有的 thread(包括不同 block 的 thread)都共享一份global memory、constant memory、和 texture memory。不同的 grid则有各自的 global memory、constant memory和 texture memory。cudaMalloc和cudaFree用于内存分配及释放，它们分配的是global memory，cudaMemcpy用于Hose-Device数据交换。
![图片](15.png)
### CUDA调试
![图片](16.png)
### 参考
CUDA学习笔记：[http://luofl1992.is-programmer.com/posts/38830.html](http://luofl1992.is-programmer.com/posts/38830.html)
CUDA概念理解：[https://blog.csdn.net/lg1259156776/article/details/52804840](https://blog.csdn.net/lg1259156776/article/details/52804840)
grid,block,thread的关系及计算：[https://blog.csdn.net/hujingshuang/article/details/53097222](https://blog.csdn.net/hujingshuang/article/details/53097222)
