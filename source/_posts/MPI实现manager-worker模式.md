---
title: MPI实现manager-worker模式
date: 2018-10-11 16:10:11
tags: MPI
---

### 并行计算中的任务分发
在并行计算中，任务划分与求解问题的规模和处理机的结构密切相关，任务划分的方式极大的影响并行计算机能否有效发挥其性能，减少求解的时间。任务分配的几种常用方法有：按块分配、卷帘分配等;
传统的块分配中，如果有m个处理器，就将任务分为m块，然后每个处理器或者进程组负责完成其中的一块任务，这种方法属于静态负载平衡。但是有的时候任务计算时间的多少是不固定的，比如当算法中出现迭代的情况时，静态负载平衡的方法可能效果较差。
还有一种情况就是当任务之间存在相互依赖时。一种解决方案就是将这些任务在处理器中利用求解连通分量的方法划分为s个任务组，再将这些任务组分配给各个从处理器。
当任务之间不存在依赖关系时，可以在主进程上将任务进行分组，形成任务列表，各个从进程向主进程发出任务请求，如果任务列表中还有没有完成的任务，那么将任务发送给从进程，否则发送一个终止信号给从进程，说明没有任务可以执行了，然后从进程发送一个本进程要终止的消息，主进程那边终止进程的计数+1,当所有从进程都终止之后程序完成，退出。

<!--more-->

### 用MPI模拟文件读取以及任务分发的操作
程序思路：
将0号进程设置为Manager，执行manager(int p)函数，而其他进程为worker,执行worker(int id)函数。Manager进程负责读取文件需要处理的文件列表，然后等待请求; 而worker进程则发出请求，在文件存在的条件下，每接收一个worker的请求，就读出一个文件发送给worker进程处理，当文件分配完毕的时候将发送给woeker的filename设置为'\n'，worker进行判断，如果接收到的filename为'\n'，则说明没有任务可以执行了，将power[0]设置为1,给manager发送本进程要终止的消息，然后Manager的终止进程计数+1,当所有进程都发送终止之后，程序完成，退出;
实现任务在主进程和从进程之间传递，以及主进程控制终止的条件就是这个int *power变量，power是一个由两个int型整数组成的数组。
power[0]=0表示从进程获得的任务是正常的任务，
power[0] = 1,表示从进程没有受到要处理的任务，可以终止了。
power[1] 表示worker_id

如果程序报错说少不到io.h头文件，可以用find /usr/include -name "io.h"，找到文件位置，在我的机器上发现io.h不在 /usr/include目录下，而在/usr/include/x86_64-linux-gnu/sys目录下，所以用sudo cp /usr/include/x86_64-linux-gnu/sys/io.h /usr/include 将头文件拷贝到/usr/include目录下即可。
```
#include <stdio.h>
#include "mpi.h"
#include <malloc.h>
#include <string.h>
#include <sys/stat.h>
#include <io.h>

void manager(int p);
void worker(int id);
int read_file_name();
int main(int argc , char* argv[])
{
    int id ;
    int p ;
    MPI_Init(&argc,&argv) ;
    MPI_Comm_rank(MPI_COMM_WORLD,&id);
    MPI_Comm_size(MPI_COMM_WORLD,&p);


    if (!id)
    {
        manager(p);
    }else{
        worker(id);
    }


    MPI_Finalize();
    return 0 ;
}


void manager(int p)
{
    int *power ;
    MPI_Status status ;
    char * filename ;


    char c ;
    FILE* fp ;
    int i ;
    int file_size ;
    int terminate ;
    terminate = 0 ;


    file_size = read_file_name() ;   // 读取文件个数
    filename = (char*)malloc(sizeof(char)*1024) ;
    power = (int*)malloc(sizeof(int)*2) ;
    memset(filename,0,sizeof(char)*1024) ;
    memset(power,0,sizeof(int)*2) ;


    fp = fopen("./files.txt","r");
    if (fp==NULL)
    {
        printf("Read file list error...") ;
    }


    i = 0 ;
    do 
    {
        printf("------------------------ I am manager ------------------------\n") ;
        MPI_Recv(power,2,MPI_INT,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status) ;   // Manager 接收到Worker的请求信息
        printf("I received the request from process :%d\n",power[1]) ;


        /************ 接收到信息后开始读文件***************/
//fgetc从文件指针stream指向的文件中读取一个字符，读取一个字节后，光标位置后移一个字节
        c= fgetc(fp);
        if ((power[0]==0)&&(c!=EOF))
        {
            while((c!='\n')&&(c!=EOF)){
                filename[i] = c ;
                i++;
                c = fgetc(fp);
            }


            MPI_Send(filename,1024,MPI_CHAR,power[1],0,MPI_COMM_WORLD) ;
            i = 0 ;
            printf("I sent the file : %s to the process :%d\n",filename,power[1]) ;
        }


        /**************** 如果文件读完则发送读完的消息给worker **********************/
        if (c==EOF)
        {
            printf("Now no files need process , send terminate sign to process:%d\n",power[1]) ;
            filename[0] = '\n';
            MPI_Send(filename,1024,MPI_CHAR,power[1],0,MPI_COMM_WORLD) ;


        }


        /******************** worker处理完发送处理完的信号**********************/
        if (power[0]==1)
        {
            terminate++;
        }


        printf("------------------------------------------------\n\n\n");


    } while (terminate<(p-1));


    printf("------------------------ The all work have done ---------------------------\n") ;


    fclose(fp);


}


void worker(int id)
{
     int *power ;
     char *filename ;
     MPI_Status status ;
     power = (int*)malloc(sizeof(int)*2) ;
     filename = (char*)malloc(sizeof(char)*1024) ;
     memset(power,0,sizeof(int)*2) ;
     memset(filename,0,sizeof(char)*1024) ; 


     power[0] = 0 ;
     power[1] = id ;


     for (;;)
     {
         printf("-----------------------I am worker : %d -------------------------\n",power[1]) ;
         MPI_Send(power,2,MPI_INT,0,id,MPI_COMM_WORLD) ;     // 给Manager发送请求信息
         MPI_Recv(filename,1024,MPI_CHAR,0,0,MPI_COMM_WORLD,&status) ;
         if(filename[0]!='\n')
         {
             printf("I received the file : %s from Manager.\n",filename) ;


         }else{
             printf("I have not received from Manager , shoud be terminated .\n") ;
             power[0] = 1 ;
             MPI_Send(power,2,MPI_INT,0,id,MPI_COMM_WORLD) ; 
             break ;
         }


         printf("-----------------------------------------------------------------\n\n\n") ;
     }


}


int read_file_name()
{
    return 10 ;
}
```
### 参考：
[https://blog.csdn.net/houqd2012/article/details/8315321#commentBox](https://blog.csdn.net/houqd2012/article/details/8315321#commentBox)
