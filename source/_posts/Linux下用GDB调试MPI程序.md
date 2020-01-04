---
title: Linux下用GDB调试MPI程序
date: 2019-04-27 09:56:37
tags: [GDB, MPI]
---
### GDB常用命令
GDB有三种启动方式
* gdb ./a.out
* gdb ./a.out core        用gdb同时调试一个运行程序和core文件，core是程序非法执行后core dump后产生的文件。
* gdb -p <pid>           如果你的程序是一个服务程序，那么你可以指定这个服务程序运行时的进程ID。gdb会自动attach上去，并调试他。可以通过 ps -ef | grep procName 来查找pId。

<!--more-->

先介绍以下GDB的常用命令

| l          | l命令相当于list，从第一行开始例出原码   | 
| ---------- | ------------------- |
| 直接回车   | 直接回车表示，重复上一次命令   | 
| break <16>   | 设置断点，在源程序第16行处   | 
| break <func>   | 设置断点，在函数func()入口处   | 
| info break   | 查看断点信息   | 
| r   | 运行程序，run命令简写   | 
| n   | 单条语句执行，next命令简写   | 
| c   | 继续运行程序，continue命令简写   | 
| bt   | backtrace 查看函数堆栈   | 
| p <x>   | 打印变量   | 
| finish   | 退出函数   | 
| q   | 退出gdb   | 
|   info f(frame)   | 打印出更为详细的当前栈层的信息，只不过，大多数都是运行时的内内地址。比如：函数地址，调用函数的地址，被调用函数的地址，目前的函数是由什么样的程序语言写成的、函数参数地址及值、局部变量的地址等等   | 
| frame <n>   | frame后加栈中的层编号，表示切换到相应的函数栈。   | 
| up <n>   | 表示向栈的上面移动n层，可以不打n，表示向上移动一层   | 
| down <n>   | 表示向栈的下面移动n层，可以不打n，表示向上移动一层   | 
| info args   | 打印出当前函数的参数名及其值   | 
| info locals   | 打印出当前函数中所有局部变量及其值   | 
| p *array@len   | 查看从array首地址开始的len个元素。@的左边是数组的首地址的值，也就是变量array所指向的内容，右边则是数据的长度，其保存在变量len中   | 
| set var < x=7 >   | 设置变量x的值为7   | 
| whatis <x>   | 查看变量x的类型   | 

### 用gdb调试MPI程序
要用到两个GDB调试器的功能
1. 在终端输入：gdb -p [pId] 该命令用于将已经在运行的进程附加到GDB上
2. 在程序中添加了死循环，将程序暂停，等待gdb调试器连接该进程。这段代码向stdout输出一行，输出正在运行的主机名以及PID，接着进入循环等待调试。
```
int j = 0;
char hostname[256];
gethostname(hostname, sizeof(hostname));
printf("PID %d on %s ready for attach\n", getpid(), hostname);
fflush(stdout);
while (0 == j)
    sleep(5);
```
gdb启动后通过 set var j=9 使程序跳出死循环，得以继续向下运行。然后就可以像调试串行程序一样调试每个进程。
注意：这段代码应该放在足够靠前的位置，比如放在MPI_Init()之后，然后调试的状态由变量j控制。当变量j被改变为非0值后，gdb连接的进程就会进入函数体
3. 使用调试器连接之后，向下运行，直到进入这个代码块（可能在睡眠的时候附加gdb调试器），然后用set var将变量置为非0值，然后就可以在代码块后设置断点并继续运行，直到命中断点。可以像调试串行程序一样调试并行程序。用这种方法一个终端对应于一个进程，可以开启多个终端。

正常运行MPI程序
![图片](1.png)

GDB连接到主进程
![图片](2.png)

GDB连接到从进程
![图片](3.png)

![图片](4.png)

### 参考
该连接第6条一般情况下够用了 [https://www.open-mpi.org/faq/?category=debugging](https://www.open-mpi.org/faq/?category=debugging)
MPI调试方法  [http://www.sci.utah.edu/~tfogal/academic/Fogal-ParallelDebugging.pdf](http://www.sci.utah.edu/~tfogal/academic/Fogal-ParallelDebugging.pdf)
GDB的常用命令  [https://www.xuebuyuan.com/3237533.html](https://www.xuebuyuan.com/3237533.html)
