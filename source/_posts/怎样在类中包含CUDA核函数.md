---
title: 怎样在类中包含CUDA核函数
date: 2018-11-27 12:18:13
tags: CUDA
---

因为CUDA中__global__函数不能作为类的成员函数，所以该如何将CUDA和C++类更好的结合起来使用呢？
方法：在类的外边定义核函数，在类的成员函数中调用核函数，将类的成员变量作为参数传入。

<!--more-->

```
// How to "wrap" a CUDA kernel with a C++ class; the kernel must be defined outside of
// the class and launched from within a class instance's method.
// 怎样将一个CUDA核函数包装在一个C++类中？
// 核函数必须在类的外面定义，并且从类实例的方法中启用
#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
 
class MyClass;
__global__ void kernel(int *a, unsigned int N);

class MyClass {
public:
  MyClass(int len) {
    length = len;
    cudaMalloc((void **)&d_data, sizeof(int)*length);
    cudaMemset((void *)d_data, 0, sizeof(int)*length);
  };
  
  ~MyClass() {
    cudaFree((void *)d_data);
    printf("%s\n","cudafree" );
  };
  
  void run(dim3 grid,dim3 block) {
    kernel<<<grid, block>>>(d_data, length);
  };


  void set(int* h_data)
  {
  cudaMemcpy(d_data,h_data,sizeof(int)*length,cudaMemcpyHostToDevice);
  }
  
  int* getData(void) {
    return d_data;
  };
  int getLength(void)
  {
    return length;
  }
  void show(void)
  {
    int h_data[length];
    cudaMemcpy(h_data, getData(), sizeof(int)*length, cudaMemcpyDeviceToHost);
    for (int i=0; i<length; i++) {
      std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;
  }
private:
  int *d_data;
  int length;
};

__global__ void kernel(int *a, unsigned int N) {
  const unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i<N) {
    a[i] += i;
  }
}
  
int main(void) {
  int arraySize = 20;

  int* testArr = new int[arraySize];
  for (int i = 0; i < arraySize; ++i)
  {
      testArr[i] = i;
  }
  // MyClass c(arraySize);  //直接声明的对象是定义在栈上的，会被自动释放  
  // c.run();
  // c.show();

  dim3 grid(1);
  dim3 block(arraySize);
  MyClass *c = new MyClass(arraySize);
  c->set(testArr);
  c->run(grid,block);
  c->show();
  delete c; //用指针指向new出来的对象是存放在堆上的，必须要手动delete对象，否则对象不会被释放掉。
}
```
更近一步，可以将参数提取出来放在一个结构体中。比如上面的例子可以改为下面这样。本质上就是将上面那个版本的kernel函数的参数封装在一个结构体中，然后通过__global__函数的方式调用。在这里MyClassStruct就相当于上面的kernel函数。
```
#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
class MyClassStruct
{
public:
  int *d_data;
  int length;
  __device__ void kernel()
  {
    const unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i<length) {
      d_data[i] += i;
    }
  };
};

__global__ void g_run(MyClassStruct obj)
{
  obj.kernel();
}

class MyClass {
public:
  MyClass(int len) {
    obj.length = len;
    cudaMalloc((void **)&obj.d_data, sizeof(int)*obj.length);
    cudaMemset((void *)obj.d_data, 0, sizeof(int)*obj.length);
  };
  
  ~MyClass() {
    cudaFree((void *)obj.d_data);
    printf("%s\n","cudafree" );
  };
  
  void run(dim3 grid,dim3 block) {
    g_run<<<grid,block>>>(obj);
  };
  
  void set(int* h_data)  {
 cudaMemcpy(obj.d_data,h_data,sizeof(int)*obj.length,cudaMemcpyHostToDevice);
  };

  int* getData(void) {
    return obj.d_data;
  };
  int getLength(void)
  {
    return obj.length;
  };
  void show(void)
  {
    int h_data[obj.length];
    cudaMemcpy(h_data, getData(), sizeof(int)*obj.length, cudaMemcpyDeviceToHost);
    for (int i=0; i<obj.length; i++) {
      std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;
  };
private:
  MyClassStruct obj;
};
int main(void) {
  int arraySize = 20;
  int* testArr = new int[arraySize];
  for (int i = 0; i < arraySize; ++i)
  {
      testArr[i] = i;
  }
  dim3 grid(1);
  dim3 block(arraySize);
  MyClass *c = new MyClass(arraySize);
  c->set(testArr);
  c->run(grid,block);
  c->show();
  delete c; //用指针指向new出来的对象是存放在堆上的，必须要手动delete对象，否则对象不会被释放掉。
}
```

### 参考
[https://gist.github.com/lebedov/bca3c70e664f54cdf8c3cd0c28c11a0f](https://gist.github.com/lebedov/bca3c70e664f54cdf8c3cd0c28c11a0f)
[https://devtalk.nvidia.com/default/topic/802257/working-with-cuda-and-class-methods/](https://devtalk.nvidia.com/default/topic/802257/working-with-cuda-and-class-methods/)
