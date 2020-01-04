---
title: linux下编译CUDPP
date: 2018-04-10 15:59:03
tags: [CUDA, CUDPP]
---

今天在linux下编译了一个CUDA库，CUDPP，第一次用CMake这个工具
记录以下编译的过程
参考cudpp库的编译和使用 [https://www.cnblogs.com/liangliangdetianxia/p/4162634.html](https://www.cnblogs.com/liangliangdetianxia/p/4162634.html)这篇文章是在windows中编译的，有些地方有点变化

<!--more-->

### 编译动态库和静态库
我的环境
ubuntu 16.04 64位
项目主页
[http://cudpp.github.io/](http://cudpp.github.io/)
源码地址
[https://github.com/cudpp/cudpp](https://github.com/cudpp/cudpp)
根据这个网址的提示进行
[https://github.com/cudpp/cudpp/wiki/BuildingCUDPPwithCMake](https://github.com/cudpp/cudpp/wiki/BuildingCUDPPwithCMake)

1. 从github上下载源码，下载后的源码目录中有个ext文件夹，是空的
2. 所以在github网页中点开进入这个文件夹。

![图片](1.png)![图片](2.png)
点开这两个连接之后，都打包下载下来一定要点击download，而不是用clone的方式下载源码，clone方式下载的源码编译不通过。我也不知道为啥！
到此，所需文件全了！！！
放在对应的文件夹下。
![图片](3.png)![图片](4.png)

下载CMake用cmake-gui来编译，配置源码目录和要build的目录，点击configure,再点generate
![图片](5.png)
Useful options to set:
* CMAKE_BUILD_TYPE: set to Debug for debug builds, Release (or blank) for release builds. Not needed on Windows.
* BUILD_APPLICATIONS: set to on to build the CUDPP example applications and testrig in addition to CUDPP libraries.（这个选项是同时编译cudpp这个库里面的例子，选上比较好）
* BUILD_SHARED_LIBS: set to on to build dynamic/shared CUDPP libraries, off to build static libraries
* CUDA_VERBOSE_BUILD: Print out commands run while compiling CUDA source files
* CUDA_VERBOSE_PTXAS: Print out the output of --ptxas-options=-verbose, which displays details of shared memory, registers, and constants used by CUDA device kernels.

这样在cudpp-build目录中就生成了Makefile文件
然后控制台进入cudpp-build目录，键入make，然后等一会儿，就编译完了
![图片](6.png)之后在cudpp-build/bin目录下可以看到三个可执行文件，控制台进入目录，执行。可以看到all test passed的提示。
![图片](7.png)
![图片](8.png)

### 在项目中配置动态库
在cudpp-build/lib目录下可以找到生成的.so动态连接库。如果不勾选动态连接选项的话生成的是静态连接库，就是.a文件。然后配合include目录下的头文件就可以用于其他项目了。
![图片](9.png)![图片](10.png)
![图片](11.png)
一开始我将动态库和源码放在同一个目录下，然后用nvcc -o main scan_gold.cpp simpleCUDPP.cu -L ./ -lcudpp命令，可以正常编译，但是运行时报错。报错的原因是共享路径配置不正确;
![图片](12.png)
一个简单易行的办法是将动态库拷贝到/usr/local/lib目录下即可
![图片](13.png)
还有一个办法就是使用静态库，将静态库拷贝到源码目录下nvcc -o main scan_gold.cpp simpleCUDPP.cu -L ./ -lcudpp编译成功。然后./main运行成功。但是静态库生成的可执行文件main非常大，因为它包括了库中所有的数据。比如在这个实验中用静态库生成的可执行文件有40多M，而用动态库生成的可执行文件只有500多K。
动态链接库和静态函数库不同，它里面的函数并不是执行程序本身的一部分，而是根据执行程序需要按需装入，同时其执行代码可在多个执行程序间共享，节省了空间，提高了效率。由于函数库没有被整合进你的程序，而是程序运行时动态的申请并调用，所以程序的运行环境中必须提供相应的库。动态函数库的改变并不影响你的程序，所以动态函数库的升级比较方便。linux系统有几个重要的目录存放相应的函数库，如/lib /usr/lib。
当要使用静态的程序库时，连接器会找出程序所需的函数，然后将它们拷贝到执行文件，由于这种拷贝是完整的，所以一旦连接成功，静态程序库也就不再需要了。然而，对动态库而言，就不是这样。动态库会在执行程序内留下一个标记指明当程序执行时，首先必须载入这个库。由于动态库节省空间，linux下进行连接的缺省操作是首先连接动态库，也就是说，如果同时存在静态和动态库，不特别指定的话，将与动态库相连接。

### 一个cudpp_hash的例子
```
#include <stdio.h>
#include <cuda_runtime_api.h>
#include "cudpp.h"
#include "cudpp_hash.h"

int main() {
  const int N = 10;

  int keys[N] = {1, 6, 4, 9, 0, 3, 7, 2, 5, 8};
  int vals[N] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  int *d_keys, *d_vals;
  cudaMalloc((void**)&d_keys, sizeof(int) * N);
  cudaMemcpy(d_keys, keys, sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_vals, sizeof(int) * N);
  cudaMemcpy(d_vals, vals, sizeof(int) * N, cudaMemcpyHostToDevice);

  int input[N] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  int output[N];

  int *d_input, *d_output;
  cudaMalloc((void**)&d_input, sizeof(int) * N);
  cudaMemcpy(d_input, input, sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_output, sizeof(int) * N);
  cudaMemset(d_output, 0, sizeof(int) * N);

  CUDPPHandle cudpp;
  cudppCreate(&cudpp);

  CUDPPHashTableConfig config;
  config.type = CUDPP_BASIC_HASH_TABLE;
  config.kInputSize = N;
  config.space_usage = 2.0;

  CUDPPHandle hash_table_handle;
  cudppHashTable(cudpp, &hash_table_handle, &config);

  cudppHashInsert(hash_table_handle, d_keys, d_vals, N);

  cudppHashRetrieve(hash_table_handle, d_input, d_output, N);

  cudaMemcpy(output, d_output, sizeof(int) * N, cudaMemcpyDeviceToHost);
  for (int i = 0; i < N; ++i) {
    printf("%d\n", output[i]);
  }

  cudppDestroyHashTable(cudpp, hash_table_handle);
  
  cudppDestroy(cudpp);

  return 0;
}
```
编译命令： nvcc cudpphashtesting.cu -o hashtesting -lcudpp -lcudpp_hash
运行结果
![图片](14.png)

### tips:
1、linux库文件分为静态库和动态库两种。静态库习惯以.a 结尾，而动态库习惯以.so(shared object)结尾。而且必须以lib开头。
2、静态库的原则是“以空间换时间”，增加程序体积，减少运行时间;因为整个函数库的所有数据都会被整合进目标代码中，他的优点就显而易见了，即编译后的执行程序不需要外部的函数库支持，因为所有使用的函数都已经被编译进去了。当然这也会成为他的缺点，因为如果静态函数库改变了，那么你的程序必须重新编译。
生成：在编译时候，先生成目标文件.o，然后用ar文件对目标文件归档，生成静态库文件。
例如：ar -rc libtest.a myalib.o （注意：ar -rc 目标 .o文件名），目标一定要以lib开头。
3、使用时候，在链接时候，加上选项 -l 后接库文件名，注意：必须是文件名去掉后缀和lib，
 如：gcc -o main main.o -ltest 。而且-ltest必须放在main.o的后面，（规则是，越底层的库越要放在后面）。
4、gcc的其他常用的选项，
-c 编译成目标文件 如：gcc -c main.c 就是编译main.c成目标文件main.o
-I 头文件的查找路径，如：gcc -c main.c -I./inc 意思是：头文件的查找路径除了默认的之外，再加上./inc目录下的。
-L 库文件的查找路径，如：gcc -o main main.o -L./lib -ltest 说明：libtest.a 或者 libtest.so 库文件的查找路径除了默认之外，再加上./lib目录。
-MM 导出文件的依赖关系（用#include 中的内容）如：gcc -MM main.c找出main.c的所依赖的头文件
-o 生成最终目标
-D宏定义 相当于在C中些语句#define ... 如：-DPI=3.14 就相当于在文件里面写语句#define PI 3.14
5、动态库
（1）、生成：在链接时，用如下选项：-shared -fpic 如： gcc -fpic -shared -o libtest.so myalib.c
（2）、使用：有隐式使用和显示使用，隐式使用就是共享方式，程序一开始运行就调进去。在链接时候用如下：
 gcc -o main main.o -L./lib -ltest(像静态库的一样)
显示使用就是在程序中用语句把动态库调进来，用系统调用：dlopen、dlsym、dlerror、dlclose函数，那样在编译链接时候，不用加上：-L./lib -ltest了。不过要使用dl*系列函数在编译链接时要加上 -ldl
6、如果同一目录下，既有静态库也有动态库，比如libtest.a libtest.so都存在，那么dl程序（等一下介绍）就把动态库调进去，没有动态的，就找静态的。再没有，就报错。
7、动态库的搜索路径
dl对动态库的搜索路径如下（按顺序如下）
 
a.编译目标代码时指定的动态库搜索路径；（如果要指定程序行时在./lib目录下找库文件libtest.so，命令如下：gcc -o main main.c -L./lib -ltest -Wl,-rpath ./lib) ，其中，-Wl的意思是，后面的选项直接交给ld程序处理,-rpath选项是说更改搜索路径为后面的参数./lib
 
b.环境变量LD_LIBRARY_PATH指定的动态库搜索路径；
 
c.配置文件/etc/ld.so.conf中指定的动态库搜索路径；（修改完文件后，用ldconfig更新）
 
d.默认的动态库搜索路径/lib和/usr/lib；
 
8、一些常用的命令（与库有关的）
 
（1）、ld 是gcc的链接程序。
 
（2）、ldd是查看可执行文件中所依赖的库的程序，比如想查main程序用到了那些动态库，可以直接
 ldd main
 
（3）、ldconfig用来更新文件/etc/ld.so.conf的修改生效。
 
（4）、nm用来查看.so库中的函数名字，标记是T的就是动态库里面生成的名字。如：nm /lib/libc*.so
 有时候当我们的应用程序无法运行时，它会提示我们说它找不到什么样的库，或者哪个库的版本又不合它胃口了等等之类的话。那么应用程序它是怎么知道需要哪些库的呢？我们前面已几个学了个很棒的命令ldd，用就是用来查看一个文件到底依赖了那些so库文件。
![图片](15.png)
Linux系统中动态链接库的配置文件一般在/etc/ld.so.conf文件内，它里面存放的内容是可以被Linux共享的动态联库所在的目录的名字。
9.库的依赖问题
比如我们有一个基础库libbase.a,还有一个依赖libbase.a编译的库，叫做libchild.a；在我们编译程序时，一定要先-lchild再-lbase。 如果使用 -lbase -lchild，在编译时将出现一些函数undefined，而这些函数实际上已经在base中已经定义；
   为什么会有库的依赖问题？    一、静态库解析符号引用：       链接器ld是如何使用静态库来解析引用的。在符号解析阶段，链接器从左至右，依次扫描可重定位目标文件（*.o）和静态库（*.a）。    在这个过程中，链接器将维持三个集合：    集合E：可重定位目标文件(*.o文件)的集合。    集合U：未解析(未定义)的符号集，即符号表中UNDEF的符号。    集合D： 已定义的符号集。    初始情况下，E、U、D均为空。    1、对于每个输入文件f，如果是目标文件(.o)，则将f加入E，并用f中的符号表修改U、D(在文件f中定义实现的符号是D，在f中引用的符号是U)，然后继续下个文件。    2、如果f是一个静态库(.a)，那么链接器将尝试匹配U中未解析符号与静态库成员(静态库的成员就是.o文件)定义的符号。如果静态库中某个成员m(某个.o文件)定义了一个符号来解析U中引用，那么将m加入E中，    同时使用m的符号表，来更新U、D。对静态库中所有成员目标文件反复进行该过程，直至U和D不再发生变化。此时，静态库f中任何不包含在E中的成员目标文件都将丢弃，链接器将继续下一个文件。    3、当所有输入文件完成后，如果U非空，链接器则会报错，否则合并和重定位E中目标文件，构建出可执行文件。  到这里，为什么会有库的依赖问题已经得到解答：  因为libchild.a依赖于libbase.a，但是libbase.a在libchild.a的左边，导致libbase.a中的目标文件(*.o)根本就没有被加载到E中，所以解决方法就是交换两者的顺序。当然也可以使用-lbase -lchild -lbase的方法。

### 参考
linux下添加动态链接库路径的方法 [https://blog.csdn.net/zxh2075/article/details/54629318](https://blog.csdn.net/zxh2075/article/details/54629318)
Linux中的静态库和动态库那点事儿 [https://www.cnblogs.com/sky-heaven/p/5918139.html](https://www.cnblogs.com/sky-heaven/p/5918139.html)
[linux下静态库和动态库一些东西](https://www.cnblogs.com/wainiwann/p/4204248.html) [https://www.cnblogs.com/wainiwann/p/4204248.html](https://www.cnblogs.com/wainiwann/p/4204248.html)


