---
title: MPI+CUDA混合编程 Makefile文件写法
date: 2018-10-22 16:06:07
tags: [Makefile, MPI, CUDA] 
---

### CUDA代码与c++代码分开时Makefile文件的写法
用网上找的一个例子作为参考，主要记录Makefile文件的写法
总的来说就是要用nvcc编译.cu文件，生成.o文件;
然后用mpic++编译.cpp文件，生成.o文件;
最后用mpic++将这两个.o文件连接起来，生成可执行文件。
在控制台中依次键入下面的命令，可以生成可执行文件main。
```
# nvcc -c test_cuda.cu
# mpic++ -c test.cpp
# mpic++ -o main test.o test_cuda.o  -L /usr/local/cuda-8.0/lib64 -lcudart
```

<!--more-->

完整的Makefile文件写法如下：
几个要注意的点：
1. 弄清楚CUDA和MPI的环境变量。如果不知道，可以用which 命令查看。
2. 最后一行，生成可执行文件的那一行，依赖的库放在最后，将目标文件写在中间，否则在有些机器上会报错。
3. 学到了一个函数，fseek， 可以将数组写入到文件指定的位置;

```
CUDA_INSTALL_PATH = /usr/local/cuda-8.0
MPI_INSTALL_PATH = /home/mpi/mpich-3.2.1

NVCC = $(CUDA_INSTALL_PATH)/bin/nvcc
MPICC = $(MPI_INSTALL_PATH)/bin/mpic++

LDFLAGS = -L $(CUDA_INSTALL_PATH)/lib64
LIB = -lcudart

CFILES = test.cpp
CUFILES = test_cuda.cu
OBJECTS = test_cuda.o test.o 
EXECNAME = test

all:
    $(NVCC) -c $(CUFILES)
    $(MPICC) -c $(CFILES)
    $(MPICC) -o $(EXECNAME) $(OBJECTS) $(LDFLAGS) $(LIB) 

clean:
    rm -f *.o $(EXECNAME)
```
完整代码
```
//test.cpp
#include<stdio.h>
#include<malloc.h>
#include<math.h>
#include<stdlib.h>
#include "mpi.h"


extern "C" void cudaFun (int is , FILE  *fp ,  int  nx , int nz );
int main( int argc, char  *argv[ ] )
{
        int myid , numprocs , count , is , nx , nz ;
        float * vp;


        nx = 1000 ; nz = 1000 ;


        MPI_Init(&argc,&argv);
        MPI_Comm_rank(MPI_COMM_WORLD,&myid);
        MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
        MPI_Barrier(MPI_COMM_WORLD);


        FILE *fp;
        fp=fopen( "test.dat" , "wb" );
        for ( is = myid ; is < 10 ; is = is    +    numprocs )
        {
                printf( " is== %d  \n "  , is ) ;
                cudaFun( is , fp , nx , nz );
        }
        MPI_Finalize( );
    return 0;
}
```

```
//test_cuda.cu
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <cuda_runtime_api.h>

__device__  volatile  int  vint = 0;
//a#########################
__global__ void fun ( float  * vp_device , int n, int nz, int  is )
{
        int it = threadIdx.x + blockDim.x * blockIdx.x;
        if  ( it < n ) {
                vp_device[it]=2000;
                if ( ( it > nz * 40 && it < 40 && it % nz < 60 ) ) 
                        vp_device [ it ] = 2500 * is * 100 ;
        }
}
//a########################
extern "C" void cudaFun ( int is , FILE  *fp ,  int  nx , int nz )
{
        int i ;
        float  * vp_device , * vp_host;


        cudaMalloc(&vp_device, nx*nz*sizeof(float));  
        cudaMemset(vp_device, 0, nx*nz*sizeof(float));


        vp_host=(float*)malloc(nx*nz*sizeof(float));


        float mstimer;


        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);


        fun <<<(  nx * nz   +   511 ) / 512, 512>>> ( vp_device ,  nx*nz , nz , is ) ;


        cudaMemcpy(vp_host, vp_device, nx*nz*sizeof(float),cudaMemcpyDeviceToHost); 


        fseek(fp,is*nx*nz*sizeof(float),0);
        for (  i  =  0  ;  i  <  nx  *  nz   ;  i   ++   )
                fwrite( &vp_host[i] , sizeof(float) , 1 , fp);


        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&mstimer, start, stop);
        printf( "CUDA : is = %d, time = %g (s)\\n " ,is, mstimer/1000);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);


        cudaFree(vp_device);
        free(vp_host);
}
```
### CUDA和MPI混合编译方法
在上面的例子中，所有与CUDA有关的代码都在test_cuda.cu中，而有时候我们需要从外部传入线程块的配置参数，就是说要在.cpp文件中调用cuda函数，此时就需要配置连接库的位置，和包含CUDA头文件的位置。在Makefile文件中分别用 -L 和 -I 指定库的位置，和.h文件的位置。
比如下面这个例子，将CUDA实现写在了kernel.cu中，将其中的函数在cuda.cuh中申明。然后在test.cpp中包含cuda.cuh头文件。在控制台中分别键入下面的命令：
```
nvcc -c kernel.cu

mpic++ -c test.cpp -L /usr/local/cuda-8.0/lib64 -lcudart -I /usr/local/cuda-8.0/include

mpic++ -o main test.o kernel.o -L /usr/local/cuda-8.0/lib64 -lcudart -I /usr/local/cuda-8.0/include
```

```
#上面的命令对应的Makefile写法
CUDA_INSTALL_PATH = /usr/local/cuda-8.0
MPI_INSTALL_PATH = /home/mpi/mpich-3.2.1

NVCC = $(CUDA_INSTALL_PATH)/bin/nvcc
MPICC = $(MPI_INSTALL_PATH)/bin/mpic++
LDFLAGS = -L $(CUDA_INSTALL_PATH)/lib64
LIB = -lcudart
CUDA_INCLUDE = -I /usr/local/cuda-8.0/include

CFILES = test.cpp
CUFILES = kernel.cu
OBJECTS = kernel.o test.o 
EXECNAME = test

all:
    $(NVCC) -c $(CUFILES)
    $(MPICC) -c $(CFILES) $(LDFLAGS) $(LIB) $(CUDA_INCLUDE)
    $(MPICC) -o $(EXECNAME) $(OBJECTS) $(LDFLAGS) $(LIB) $(CUDA_INCLUDE)

clean:
    rm -f *.o $(EXECNAME)
```
最后键入
mpirun -n 2 ./main运行代码，注意这里，机器上有几个GPU就用几个进程，否则会报错。
![图片](1.png)


源代码
```
//cuda.cuh
#pragma once
#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void GPU_add(float *d_loc_matA,float * d_loc_matB,float * d_loc_matC,int nx,int ny,int loc_nz,dim3 dimBlock,dim3 dimGrid);
```

```
//kernel.cu
#include "cuda.cuh"
__global__ void matadd_kernel(float *matA, float *matB, float *matC, int nx, int ny, int nz)
{
   int ix = blockDim.x * blockIdx.x + threadIdx.x;
   int iy = blockDim.y * blockIdx.y + threadIdx.y;


   for (int iz = 0; iz < nz; iz ++)
   {
     if (ix < nx && iy < ny)
        matC[iz * ny * nx + iy * nx + ix] = matA[iz * ny * nx + iy * nx + ix] + matB[iz * ny * nx + iy * nx + ix];
   }
}


 void GPU_add(float *d_loc_matA,float * d_loc_matB,float * d_loc_matC,int nx,int ny,int loc_nz,dim3 dimBlock,dim3 dimGrid)
 {
       matadd_kernel<<<dimGrid, dimBlock>>>(d_loc_matA, d_loc_matB, d_loc_matC, nx, ny, loc_nz);
 }
```


```
//test.cpp
#include "cuda.cuh"
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <stdio.h>
#include "mpi.h"

using namespace std;
#define BLOCK_DIMX 16
#define BLOCK_DIMZ 16
void CPUtest(int settimes, int row, int col, float **GA, float **GB)
{
 for (int i = 0; i < settimes; i++)
 {
  for(int i=0;i<row+1;i++){
   for(int j=0;j<col+1;j++){
    GA[i][j]=GB[i][j];
   }
  }
 }
}

int InitCUDA(int myid)
{
  int devCount = 0;
  int dev = 0;

  cudaGetDeviceCount(&devCount);
  if (devCount == 0) 
  {
    fprintf(stderr, "There is no device supporting CUDA.\n");
    return false;
  }

  for (dev = 0; dev < devCount; ++dev)
  {
    cudaDeviceProp prop;

    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
    {
       if (prop.major >= 1) break;
    }
  }

  if (dev == devCount)
  {
     fprintf(stderr, "There is no device supporting CUDA.\n");
     return false;
  }

  cudaSetDevice(myid);
  cudaDeviceProp prop1;
  cudaGetDeviceProperties(&prop1,myid);
  fprintf(stdout, "multiProcessorCount %d : %d\n", myid, prop1.multiProcessorCount);
  return true;
}

void matadd(float *matA, float *matB, float *matC, int nx, int ny, int nz, int myid, int size)
{
  int ista, iend;
  //给主进程分配最后一个分块
  if (myid != 0) 
  {
    ista = (myid - 1) * ( nz / size); 
    iend = ista + nz / size - 1;
  } 
  else 
  {
    ista = (size - 1) * (nz / size);
    iend = nz - 1;
  }


  int loc_nz = iend - ista + 1;
  float *loc_matA = (float *) malloc( loc_nz * ny * nx * sizeof(float));
  float *loc_matB = (float *) malloc( loc_nz * ny * nx * sizeof(float));
  float *loc_matC = (float *) malloc( loc_nz * ny * nx * sizeof(float));


  MPI_Status status;
  int *count=new int[size];//记录每个进程处理的个数多少


  if (myid != 0)
  {
    MPI_Send(&loc_nz, 1, MPI_INT, 0, myid, MPI_COMM_WORLD);
  } 
  else 
  {
    count[0] = loc_nz;
    for (int i = 1; i < size; i ++) 
    {
      MPI_Recv(&count[i], 1, MPI_INT, i, i, MPI_COMM_WORLD, &status);
    } 
  }


  if (myid == 0)
  {
    for (int ix = 0; ix < count[0] * ny * nx; ix ++)
    {
     loc_matA[count[0] * ny * nx - 1 - ix] = matA[nz * ny * nx - 1 - ix];
     loc_matB[count[0] * ny * nx - 1 - ix] = matB[nz * ny * nx - 1 - ix];
    }
    for (int isz = 1; isz < size; isz ++) 
    {
      int idx = 0;
      if (isz == 1) 
      {
        idx = 0;
      }
      else 
      {
        idx += count[isz - 1];
      }
      MPI_Send(matA + idx * ny * nz, count[isz] * ny * nx, MPI_FLOAT, isz, isz, MPI_COMM_WORLD);
      MPI_Send(matB + idx * ny * nz, count[isz] * ny * nx, MPI_FLOAT, isz, isz, MPI_COMM_WORLD);
    }
  } 
  else 
  {
    MPI_Recv(loc_matA, loc_nz * ny * nx, MPI_FLOAT, 0, myid, MPI_COMM_WORLD, &status);
    MPI_Recv(loc_matB, loc_nz * ny * nx, MPI_FLOAT, 0, myid, MPI_COMM_WORLD, &status);
  }


  float *d_loc_matA;
  cudaMalloc((void **) &d_loc_matA, loc_nz * ny * nx * sizeof(float));
  cudaMemcpy(d_loc_matA, loc_matA, loc_nz * ny * nx * sizeof(float), cudaMemcpyHostToDevice);


  float *d_loc_matB;
  cudaMalloc((void **) &d_loc_matB, loc_nz * ny * nx * sizeof(float));
  cudaMemcpy(d_loc_matB, loc_matB, loc_nz * ny * nx * sizeof(float), cudaMemcpyHostToDevice);


  float *d_loc_matC;
  cudaMalloc((void **) &d_loc_matC, loc_nz * ny * nx * sizeof(float));


  dim3 dimBlock(BLOCK_DIMX, BLOCK_DIMZ);
  dim3 dimGrid(nx / BLOCK_DIMX, ny / BLOCK_DIMZ);


  GPU_add(d_loc_matA, d_loc_matB, d_loc_matC, nx, ny, loc_nz,dimBlock,dimGrid);


  cudaMemcpy(loc_matC, d_loc_matC, loc_nz * ny * nx * sizeof(float), cudaMemcpyDeviceToHost);


  if (myid !=0 )
  {
    MPI_Send(loc_matC, loc_nz * ny * nx, MPI_FLOAT, 0, myid, MPI_COMM_WORLD);
  } 
  else 
  {
    for (int ix = 0; ix < count[0] * ny * nx; ix ++)
      matC[nz * ny * nx - 1 - ix] = loc_matC[loc_nz * ny * nx - 1 - ix];
    for (int isz = 1; isz < size; isz ++)
    {
      int idx = 0;
      if (isz == 1) 
        idx = 0;
      else 
        idx += count[isz - 1];
      MPI_Recv(matC + idx * ny * nz, count[isz] * ny * nx, MPI_FLOAT, isz, isz, MPI_COMM_WORLD, &status);
    }
  }

  cudaFree(d_loc_matA);
  cudaFree(d_loc_matB);
  cudaFree(d_loc_matC);

  free(loc_matA);
  free(loc_matB);
  free(loc_matC);
  return;
}


////////////////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>


int InitCUDA(int myid);
void matadd(float *matA, float *matB, float *matC, int nx, int ny, int nz, int myid, int size);
int main(int argc, char *argv[])
{
  int myid, numprocs;
  int namelen;
  MPI_Status status;


  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);


  InitCUDA(myid);


  int nx = 96, ny = 96, nz = 500;
  float *a = (float *)malloc(nx * ny * nz * sizeof(float));
  float *b = (float *)malloc(nx * ny * nz * sizeof(float));
  float *c = (float *)malloc(nx * ny * nz * sizeof(float));


  for (int iz = 0; iz < nz; iz ++)
    for (int iy = 0; iy < ny; iy ++)
      for (int ix = 0; ix < nx; ix++)
      {
        a[iz * ny * nx + iy * nx + ix] = 1.0f;
        b[iz * ny * nx + iy * nx + ix] = 2.0f;
        c[iz * ny * nx + iy * nx + ix] = 0.0f;
      }


  clock_t tstart = clock();
  matadd(a, b, c, nx, ny, nz, myid, numprocs);
  clock_t tend = clock();


  if (myid == 0)
    printf("time for matrix addition is %.5f\n", (double)(tend - tstart)/CLOCKS_PER_SEC);


  if (myid == 0)
  {
    printf("c = %f\n", c[nx * ny * nz - 1]);


    for (int iz = 0; iz < nz; iz ++)
     for (int iy = 0; iy < ny; iy ++)
       for (int ix = 0; ix < nx; ix++)
         if ((c[iz * ny * nx + iy * nx + ix] - 3.0) >1.0e-2)
         {
              fprintf(stderr, "Error occurs\n");
              return EXIT_FAILURE;
         }
  }


  free(a);
  free(b);
  free(c);


  MPI_Finalize();
  return EXIT_SUCCESS;
}
```

