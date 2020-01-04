---
title: Array_2D_CUDA使用CUDA扩展C++模板库
date: 2019-01-23 11:17:40
tags: [CUDA, 模板]
---
### 场景
考虑一个场景，你正在开发一个c++模板库，用于对自定义的复杂数据类型执行计算密集型处理，并且使用了一些通过CUDA加速一些较慢的功能。但是对于大多数用户来说CUDA的学习成本较高，且用户的机器并不一定支持CUDA的运行。最佳方案是让用户能够像使用CPU函数一样使用GPU函数。从软件工程的角度思考以下，这个问题该怎样处理呢？
一种简单直观的思路是开发两个独立的项目。更好一点的方法是对每个用CPU和GPU实现的函数提供编译时选项。这样用户在使用时需要选择一个版本的实现并一直使用，或者要重新编译才能改变执行文件。此外，对于一些小数据来说，CPU运行的比GPU快，但当数据量大的时候GPU会有很好的加速效果。所以最佳方案是让用户能在在CPU模式和GPU模式下任意切换。
还有一个问题涉及到开发CUDA库，CUDA是一个非常低级的语言，如果我们的库中有复杂的数据结构，就有可能很难管理数据分配、内存传输。在CPU端，C++类可以通过抽象的方式使得开发变得容易。理想情况下，我希望在GPU上能够做同样的事情。通过专业的模板元编程，能够为现有的类创建CUDA-interface，通过抽象的处理cudaMalloc,cudaMemcpy等低级的GPU内存管理操作，从而大大简化GPU的开发。
<!--more-->
### 案例
接下来实现一个简单的二维数组的模板类，并实现将数组中每个元素求平方的简单函数。这个例子只是考虑传达一种软件工程角度的编程思想。在实现中没有引用计数机制的情况下，在构造函数、析够函数中传递/释放了原始指针，这通常是我们不希望看到的。同样也没有重载operator[]，没有错误检查等。
### CPU版本实现
```
//Array2D.h
#ifndef ARRAY2D_H
#define ARRAY2D_H
#include <iostream>

using namespace std;
template <class T>
class Array2D {
public:
    Array2D(T* _data,
            const size_t& _nrows,
            const size_t& _ncols); // constructor
    Array2D(const Array2D<T>& other); // copy constructor
    Array2D<T>& operator=(const Array2D<T>& other);
    T& operator[](int i)
    {
        if(i > size())
        {
            return data[0];
        }
        else
        {
            return data[i];
        }
    }
    ~Array2D(){delete[] this->data;}
    size_t get_nrows() const {return this->nrows;}
    size_t get_ncols() const {return this->ncols;}
    size_t size()      const {return this->N;}
    T* begin(){return data;}
    T* begin()const{return data;}
    T* end(){return data + this->size();}
    T* end()const{return data + this->size();}

private:
    T* data;
    size_t nrows;
    size_t ncols;
    size_t N;
};

template <class T>
Array2D<T>::Array2D(T* _data,
                    const size_t& _nrows,
                    const size_t& _ncols):data(_data), nrows(_nrows), ncols(_ncols){
    this->N = _nrows * _ncols;
};

template <class T>
Array2D<T>::Array2D(const Array2D<T>& other):nrows(other.nrows), ncols(other.ncols), N(other.N){
    data = new T[N];
    auto i = this->begin();
    for (auto& o:other)*i++=o;
};

template <class T>
Array2D<T>& Array2D<T>::operator=(const Array2D<T>& other){
    this->ncols = other.ncols;
    this->ncols = other.nrows;
    this->N     = other.N;

    // here should compare the sizes of the arrays and reallocate if necessary
    delete[] data;
    data = new T[N];
    auto i = this->begin();
    for (auto& o:other)*i++=o;
    return *this;
};
#endif //ARRAY2D_H
```
这个类中存储了一个指针，和二维数组的行数、列数。有一个copy操作的重载。定义了begin()和end()方法，析够函数中释放了指针。一个二维数组对象可以通过new操作返回的指针来构造了。接下来实现ArrayPow2，非常简单。
```
//ArrayPow2.h
#include <algorithm>
#include "Array2D.h"
template <class T>
void ArrayPow2_CPU(Array2D<T>& in, Array2D<T>& result){
    std::cout << "Using the CPU version\n";
    std::transform(in.begin(), in.end(), result.begin(), [](const T& a){return a*a;});
}
```
std::transform的第四个参数中用一个lambda表达式对在第一个参数到第二个参数中的每个元素进行计算，并存储在第三个参数中。
接下来实现一个简单的测试，来验证函数是否正确。
```
//driver1.cpp
#include <iostream>
#include "Array2D.h"
#include "ArrayPow2.h"
using namespace std;
int main() {
    Array2D<float> arr(new float[100], 10, 10);
    int a = 2;
    for (auto& i:arr)i=++a;
    Array2D<float> result(arr);
    ArrayPow2(arr, result);


    cout << "arr[0]   = " << *arr.begin() << endl;
    cout << "arr[0]^2 = " << *result.begin() << endl;
    return 0;
}
```
### GPU-CUDA模板实现
为了使得开发CUDA代码更容易，首先要实现一个CUDA版本的Array2D类。可以通过创建一个特殊的Array2D类。为了区别于CPU版本的Array2D，引入了一个辅助结构体，在结构体中只包含一个值。现在能够创建一个Array2D< Cutype<T>>数组，这与Array2D<T>的对象完全不同。这样就可以通过抽象的方法来调用cudaMalloc,cudaMemcpy等低级函数。在我的实现中所有的类型都会包含float型的数据，可以通过Cutype<float>用CUDA版的array本来存储float型的数据。这样一层数据类型并不会产生额外的开销。在编译阶段已经被float代替掉了，可以通过检查汇编代码来测试。
```
//Array2D_CUDA.h
#ifndef ARRAY2D_CUDA_H
#define ARRAY2D_CUDA_H
#include <iostream>
#include "Array2D.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
using namespace std;

template <class T>
struct Cutype{
    T val;
};

template <class U>
class Array2D< Cutype<U> > {
public:
    Array2D(U* _data,
            const size_t& _nrows,
            const size_t& _ncols);
    Array2D(const Array2D<U>&);
    Array2D< Cutype<U> >& operator=(const Array2D<U>& other);
    U& operator[](int i)
    {
        if(i > size())
        {
            return data[0];
        }
        else
        {
            return data[i];
        }
    }
    ~Array2D();
    size_t get_nrows() const {return *this->nrows;}
    size_t get_ncols() const {return *this->ncols;}
    size_t size()      const {return *this->N;}
    U* begin()const{return data;}
    U* end()const{return data + this->size();}
    U* begin(){return data;}
    U* end(){return data + this->size();}
private:
    U* data;
    size_t* nrows;
    size_t* ncols;
    size_t* N;


};

template <class U>
Array2D< Cutype<U> >::Array2D(U* _data,
                    const size_t& _nrows,
                    const size_t& _ncols):data(_data){
    size_t N_tmp = _nrows * _ncols;


    cudaMalloc((void**)&nrows, sizeof(size_t));
    cudaMalloc((void**)&ncols, sizeof(size_t));
    cudaMalloc((void**)&N    , sizeof(size_t));
    cudaMalloc((void**)&data , sizeof(U) * N_tmp);


    cudaMemcpy(nrows, &_nrows, sizeof(size_t) , cudaMemcpyHostToDevice);
    cudaMemcpy(ncols, &_ncols, sizeof(size_t) , cudaMemcpyHostToDevice);
    cudaMemcpy(N,     &N_tmp , sizeof(size_t) , cudaMemcpyHostToDevice);
    cudaMemcpy(data,  _data  , sizeof(U)*N_tmp, cudaMemcpyHostToDevice);
};


template <class U>
Array2D< Cutype<U> >::Array2D(const Array2D<U>& other){
    size_t N_tmp = other.size();


    cudaMalloc((void**)&nrows, sizeof(size_t));
    cudaMalloc((void**)&ncols, sizeof(size_t));
    cudaMalloc((void**)&N    , sizeof(size_t));
    cudaMalloc((void**)&data , sizeof(U) * N_tmp);


    const size_t other_nrows = other.get_nrows();
    const size_t other_ncols = other.get_ncols();
    const size_t other_N = other.size();
    U *other_data = other.begin();


    cudaMemcpy(nrows, &other_nrows, sizeof(size_t) , cudaMemcpyHostToDevice);
    cudaMemcpy(ncols, &other_ncols, sizeof(size_t) , cudaMemcpyHostToDevice);
    cudaMemcpy(N,     &other_N    , sizeof(size_t) , cudaMemcpyHostToDevice);
    cudaMemcpy(data,  other_data  , sizeof(U)*N_tmp, cudaMemcpyHostToDevice);
}

template <class U>
Array2D< Cutype<U> >& Array2D< Cutype<U> >::operator=(const Array2D<U>& other){
    size_t N_tmp = other.size();


    cudaMalloc((void**)&nrows, sizeof(size_t));
    cudaMalloc((void**)&ncols, sizeof(size_t));
    cudaMalloc((void**)&N    , sizeof(size_t));
    cudaMalloc((void**)&data , sizeof(U) * N_tmp);


    const size_t other_nrows = other.get_nrows();
    const size_t other_ncols = other.get_ncols();
    const size_t other_N = other.size();
    U *other_data = other.begin();


    cudaMemcpy(nrows, &other_nrows, sizeof(size_t) , cudaMemcpyHostToDevice);
    cudaMemcpy(ncols, &other_ncols, sizeof(size_t) , cudaMemcpyHostToDevice);
    cudaMemcpy(N,     &other_N    , sizeof(size_t) , cudaMemcpyHostToDevice);
    cudaMemcpy(data,  other_data , sizeof(U)*N_tmp, cudaMemcpyHostToDevice);


    return *this;
}

template <class U>
Array2D< Cutype<U> >::~Array2D(){
    cudaFree(nrows);
    cudaFree(ncols);
    cudaFree(N);
    cudaFree(data);
}

#endif //ARRAY2D_CUDA_H
```
大多数代码看上去与CPU版本非常相似，在拷贝构造函数是通过传入一个Array2D<T>的数组对象进行的，在代码中定义了如何从一个Array2D<T>数组拷贝到一个Array2D<Cutype<T>>.这样可以通过传入一个主机端的数组，在GPU上构建一个数组对象。new和delete被cudaMalloc和cudaFree取代，数据拷贝由cudaMemcpy处理。构建类的方式变了，但是与类进行交互的方式并没有变。花时间构造这样一个类从长远来看可以节约大量的时间，不必为每个内核手动处理内存复制。
下面是CUDA版本的ArrayPow2
```
//ArrayPow2.cuh
#include "Array2D_CUDA.h"
#include "Array2D.h"
template <class T>
void ArrayPow2_CUDA(Array2D<T>& in, Array2D<T>& result);
```

```
//ArrayPow2.cu
#include "Array2D_CUDA.h"
#include "Array2D.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include <iostream>
#define BLOCK_SIZE 1024
template <class T>
__global__ void pow2(T* in, T* out, size_t N){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if (idx < N)out[idx] = in[idx] * in[idx];
}

template <class T>
void ArrayPow2_CUDA(Array2D<T>& in, Array2D<T>& result) {
    std::cout << "Using the GPU version\n";
    Array2D< Cutype<T> > in_d(in);
    std::cout << "in[0] = " << *in.begin() << std::endl;
    size_t N = in.size();
    std::cout << "N = " << N << std::endl;
    pow2 <<< (N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >>> (in_d.begin(), in_d.begin(), in.size());
    cudaDeviceSynchronize();
    cudaMemcpy(result.begin(), in_d.begin(), sizeof(T) * N, cudaMemcpyDeviceToHost);
}

template void ArrayPow2_CUDA(Array2D<float>&, Array2D<float>&);
template __global__ void pow2(float*, float*, size_t);
```
### CUDA模板trick
在上面的实现中，包装函数的名字与原始的CPU实现相同，调用内核也使用了.begin()。更重要也让人很痛苦的一点是最后两行必须加上。
经常使用模板的同学一定碰到过当你想把定义和实现分开在不同的文件中时，会报错undefined reference...原因是模板仅仅用于编译器用一个给定的类型构造类。而如果模板代码在编译单元中存在，那么它在知道实际需要实例化哪些类之前就会被编译。当连接器试图与它需要的任何类型模板连接时，通常会报错undefined reference...这意味着它在寻找一个未定义的对象。在C++中，通常最简单的解决方案是将完整的模板实现放在头文件中。这样保证了需要实例化的类在编译时可见。但是在CUDA代码中，这种方案不起作用，原因是NVCCC对CUDA和C++代码的编译在本质上是分离的，因此必须有多个文件。解决方案是强制实例化要使用的模板类型，最后两行就是实例化的过程。
### 整合CPU版本和CUDA版本
现在已经实现了CUDA模板和内核，接下来要做的就是将这些与之前的CPU版本的库整合在一起，并且当用户只能使用CPU或者不想用GPU时不填加任何CUDA的东西，提供一种方式在CPU和GPU之间选择。选择时不需要程序员在代码中修改很多地方的函数名称。方法如下：
1. 创建一个函数指针，指针指向的函数与纯C++版本和CUDA版本的一样，在你改变函数指针之前对其进行重命名，命名成任何方便理解的名字。
2. 添加一个编译指示，ENABLE_GPU,如果在编译时没有定义ENABLE_GPU，意味着不会使用CUDA的函数，将函数指针指向CPU版本。 如果定义了ENABLE_GPU，在命令行输入引入运行时检查，并适当的将函数指针设置为CPU或者CUDA版本。
```
//main.cpp
#include <iostream>
#include <cstring>
#include "Array2D.h"
#include "ArrayPow2_CPU.h"

#ifdef ENABLE_GPU
#include "Array2D_CUDA.h"
#include "ArrayPow2_CUDA.cuh"
#endif //ENABLE_GPU

template <class T>
using ArrayPow2_F = void(*)(Array2D<T>&, Array2D<T>&);
ArrayPow2_F<float> ArrayPow2;
using namespace std;

int main(int argc, char** argv) {
#ifdef ENABLE_GPU
    if (argc>2 && !strcmp(argv[1],"gpu")){
        if (!strcmp(argv[2],"1")){
            ArrayPow2 = ArrayPow2_CUDA;
        } else{
            ArrayPow2 = ArrayPow2_CPU;
    }
    } else
    {
        ArrayPow2 = ArrayPow2_CUDA;
    }
#else
    ArrayPow2 = ArrayPow2_CPU;
#endif //ENABLE_GPU

    Array2D<float> arr(new float[12], 6, 2);
    int a = 2;
    for (auto& i:arr)
    {
        i=++a;
    }
    
    Array2D<float> result(arr);
    ArrayPow2(arr, result);
    for (int i = 0; i < 12; ++i)
    {
        cout << *(arr.begin()+i) << "\t"<< *(result.begin()+i) << endl;
    }
    return 0;
}
```

```
all: cpu
clean:
    -rm demo
cpu:
    g++ -std=c++11 main.cpp -o demo
gpu:
    nvcc -std=c++11 main.cpp -D ENABLE_GPU ArrayPow2_CUDA.cu -o demo
```
通过这样的设置现在可以有3种方式来编译运行程序：
1. 直接make
2. 编译是使用GPU，make gpu,现在默认就是用GPU运行
3. 用make gpu编译，在运行时的命令行上通过添加gpu 0，这样就不会使用GPU运算了

![CPUtest](CPUtest.png)![GPUtest](GPUtest.png)

通过这种编程模式，不想使用GPU的用户不会受到任何影响。对于想要使用GPU的用户来说，用函数指针来代替加速函数是代价最小的一种方式。一个项目中可能会调用很多次ArrayPow2，只要这个指针指向GPU实现的函数，整个项目立即就会使用GPU版本，无需进一步更改。考虑以下几种场景
* 根据数组的大小和优化的效果选择使用GPU版本还是CPU版本
* 当程序在大型集群上运行时，有些节点上有GPU有些节点没有，此时运行一个简单的查询来判断是否有可用的GPU，如果没有，则退回CPU版本执行

总之良好的可扩展性是软件工程首要考虑的问题之一。
### 参考
[http://alanpryorjr.com/2017-02-11-Flexible-CUDA/](http://alanpryorjr.com/2017-02-11-Flexible-CUDA/)
