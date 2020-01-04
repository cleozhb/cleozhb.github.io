---
title: linux性能测试工具
date: 2019-01-19 16:03:19
tags: 性能测试
---

## 安装perf
sudo apt install linux-tools-common
然后键入perf
![图片](1.png)

<!--more-->

按照提示键入
sudo apt-get install linux-tools-4.4.0-139-generic linux-cloud-tools-4.4.0-139-generic linux-tools-generic linux-cloud-tools-generic
安装完成
![图片](2.png)

## gprof
gprof可以显示程序运行的“flat profile”，包括每个函数的调用次数，每个函数消耗的处理器时间。也可以显示“调用图”，包括函数的调用关系，每个函数调用花费了多少时间。还可以显示“注释的源代码”，是程序源代码的一个复本，标记有程序中每行代码的执行次数。常用于Linux中，默认是自带的。
### 基本用法
1. 使用-pg选项编译和链接你的应用程序。
2. 像往常一样运行你的应用程序，使之运行完成后生成供gprof分析的数据文件（默认是gmon.out）。运行会比平时慢一些（大约2倍）
3. 使用gprof程序分析你的应用程序生成的数据。
4. （可选， 见章节4）可用python可视化。
```
 gcc -pg -o test test.c               //程序文件名称 test.c 编译时使用 -pg
```
现在我们可以再次运行生成的结果文件test ，并使用我们前面使用的测试数据。这次我们运行的时候，test运行的分析数据会被搜集并保存在'gmon.out'文件中，我们可以通过运行 ‘gprof  test gmon.out>result’到一个result文件来查看结果。
### 原理分析
profiling中插入了工具代码，确定程序的各个部分需要多少时间，GPROF产生两种形式的信息。首先它能确定程序中每个函数花费了多少CPU时间。其次，它能统计每个函数被调用的次数，包括顶层调用次数和递归调用次数。例如函数是一个递归过程，它扫描一个hash bucket的链表，查找一个特殊的字符串。对于这个函数，递归调用次数 / 顶层调用 大致等于程序每次平均扫描大约多少个元素，这提供了关于遍历这些链表的长度的统计信息。这些信息帮助我们确定这些函数在整个程序运行中的重要性。使得我们能理解程序的动态行为。

在所有的函数内部加入了三个函数：
**.cfi_startproc负责初始化profile环境，分配内存空间**
**_mcount记录每个函数代码的caller和callee的位置**
**.cfi_endproc清除profile环境，保存结果数据为gmon.out，供gprof分析结果**
 
在编译和链接程序的时候（使用 -pg 编译和链接选项），gcc 在你应用程序的每个函数中都加入了一个名为mcount（or“_mcount”, or“__mcount”）的函数，也就是说-pg编译的应用程序里的每一个函数都会调用mcount, 而mcount会在内存中保存一张函数调用图，并通过函数调用堆栈的形式查找子函数和父函数的地址。这张调用图也保存了所有与函数相关的调用时间，调用次数等等的所有信息。程序运行结束后，会在程序退出的路径下生成一个 gmon.out文件。这个文件就是记录并保存下来的监控数据。
```
g++ -pg -o main main.cpp
./main ../data/input.txt  //生成gmon.out

gprof ./main gmon.out > result.txt
gprof2dot -n0 -e0 -w ./result.txt > result.dot

以上2步可以用管道连接写成一步
gprof ./main | gprof2dot -n0 -e0 -w | dot -o result.dot

//如果需要生成png图片，继续执行下面的指令
dot -Tpng -o result.png result.dot  
```

## 安装可视化工具
gprof2dot可以将众多代码效率检测工具生成的结果可视化出来，gprof,valgrind等。
sudo apt-get install python graphviz
sudo apt-get install python3 xdot
sudo pip install gprof2dot

xdot是用来查看dot文件的python脚本工具
然后输入gprof2dot -h
![图片](3.png)
在编译时-pg参数产生供gprof剖析用的可执行文件(会在每个function call 插入一个_mcount的routine 功能)  -g参数产生供gdb调试用的可执行文件
例如：
```
编译时加入-pg
$ gcc -pg -g -o test test.c

接着执行  ./test ，会在当前目录下默认生成 gmon.out，有输入参数的跟在后面就可以
$./test ../data/srcdata.tif

执行以下命令，生成图片output.png
$ gprof ./test | gprof2dot -n0 -e0 | dot -Tpng -o output.png
执行下面命令生成dot文件，可以用xdot查看
$ gprof ./main | gprof2dot -n0 -e0 -s | dot -o output.dot
```

```
也可以用valgrind工具检查运行过程中的时间。它生成的结果非常详细，甚至连函数入口，及库函数调用都标识出来了
valgrind --tool=callgrind ./main ../../data/sanDiego.tif
执行完之后会生成"callgrind.out.XXX"的文件这是分析文件，可以直接利用：callgrind_annotate callgrind.out.XXX 打印结果，也可以使用：
gprof2dot -f callgrind callgrind.out.XXX | dot -o output.dot 来生成dot结果
```
gprof2dot默认是部分函数调用图，对性能影响不大的函数调用都不显示，如果想要显示全部的函数调用，可以 gprof2dot -n0 -e0 ，默认是n0.5即影响小于5%的函数就不显示了。具体可以通过gprof2dot -h查看。
**-s** 表示不显示诸如模板，函数入口参数等等，使得函数名称显示更加精简。

查看图片，可以看到函数调用的百分比和次数，以及热点线路。建议生成.dot文件，用xdot查看，这是矢量文件，当函数比较多，调用情况复杂时放大看不会失真。程序调用过程清晰明了，
![图片](4.png)
+-----------------------------------+
|          function name           |
| total time % ( self time % ) |
|            total calls                 |
+-----------------------------------+
每个节点，方框内部依次显示
* 函数名称
* 函数整体包括内部子函数占用时间%比
* 函数自身，不包括内部子函数占用时间%比
* 函数执行次数

每一条边表示父函数调用子函数所占用的时间百分比，默认低于1%就不显示了，这里将设置-e0。
           total time %
              calls
parent --------------------> children
节点和边的颜色跟据总时间的百分比来变化

![图片](5.png)
## 参考
* gprof2dot:[https://github.com/jrfonseca/gprof2dot](https://github.com/jrfonseca/gprof2dot)
* xdot:[https://github.com/jrfonseca/xdot.py](https://github.com/jrfonseca/xdot.py)
* 深入理解计算机系统 第5章 P388
* C++优化工具gprof [https://blog.csdn.net/jacke121/article/details/56044754](https://blog.csdn.net/jacke121/article/details/56044754)
* linux下性能分析及内存泄漏检测[https://blog.csdn.net/u014717036/article/details/50762252](https://blog.csdn.net/u014717036/article/details/50762252)

