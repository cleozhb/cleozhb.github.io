---
title: linux集群相关笔记
date: 2017-12-29 15:51:06
tags: [Linux, 集群]
---

HPSCIL集群介绍
    实验室的集群装的是rocks的集群，centos7的系统。需要学习一些集群管理的知识，sge, slurm等。

<!--more-->

CPU信息
参考 [https://blog.csdn.net/dongfang12n/article/details/79968217](https://blog.csdn.net/dongfang12n/article/details/79968217)
    非统一内存访问（NUMA）是一种用于多处理器的电脑记忆体设计，内存访问时间取决于处理器的内存位置。 在NUMA下，处理器访问它自己的本地存储器的速度比非本地存储器（存储器的地方到另一个处理器之间共享的处理器或存储器）快一些。
![图片](1.png)
 
![图片](2.png)
    我们的集群中有两个NUMA结点。每个NUMA结点有一些CPU, 一个内部总线，和自己的内存，甚至可以有自己的IO。每个CPU有离自己最近的内存可以直接访问。所以，使用NUMA架构，系统的性能会更快。在NUMA结构下，我们可以比较方便的增加CPU的数目。而在非NUMA架构下，增加CPU会导致系统总线负载很重，性能提升不明显。 
    每个CPU也可以访问另外NUMA结点上的内存，但是这样的访问，速度会比较慢。我们要尽量避免。应用软件如果没有意识到这种结构，在NUMA机器上，有时候性能会更差，这是因为，他们经常会不自觉的去访问远端内存导致性能下降。
 
**NUMA的几个概念（Socket，Core，Thread , Node）**
man lscpu 查看系统中对这个命令的解释：
   CPU    The logical CPU number of a CPU as used by the Linux kernel.
   CORE   The logical core number.  A core can contain several CPUs.
   SOCKET The logical socket number.  A socket can contain several cores.
   BOOK   The logical book number.  A book can contain several sockets.
·         Socket就是主板上的CPU插槽;
·         Core就是socket里独立的一组程序执行的硬件单元，比如寄存器，计算单元等;
·         Thread：就是超线程hyperthread的概念，逻辑的执行单元，独立的执行上下文，但是共享core内的寄存器和计算单元。
     
    NUMA体系结构中多了Node的概念，这个概念其实是用来解决core的分组的问题，具体参见下图来理解（图中的OS CPU可以理解thread，那么core就没有在图中画出），从图中可以看出每个Socket里有两个node，共有4个socket，每个socket 2个node，每个node中有8个thread，总共4（Socket）× 2（Node）× 8 （4core × 2 Thread） = 64个thread。
![图片](3.png)
我们的集群中就是
    2（Socket）× 2（Node）× 8 （4core × 2 Thread） = 32个thread。
 
 
CPU配置信息
![图片](4.png)![图片](5.png)
 
内存信息
![图片](6.png)
可以看出内存大小为125G。
 
![图片](7.png)
硬盘总共930.4G
 
Rocks 集群基本信息查看
![图片](8.png)
然后我访问 ssh compute-0-5
访问到子节点，然后查看显卡信息
![图片](9.png)

