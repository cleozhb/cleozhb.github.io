---
title: c++类的内存分布
date: 2018-02-16 12:19:05
tags: c++ 
---
1.C++中，类函数是放在代码段的，用static修饰的类成员变量和函数是放在静态数据区的，这些都不放入类的内存中。
2.要是有虚函数（无论多少个虚函数）的话，编译器会自动给类创建一个虚表指针，指向虚表，这个虚表指针占8个字节。
3.类的内存计算，linux64位默认为8个字节对齐

### 例1

<!--more-->                

```
#pragma pack(push)
#pragma pack(4)  //要是这里的改成1字节对齐，则这个类是13个字节，而不是16个字>节
#include <iostream>
using namespace std;

class A               //总字节数：4+1+8=13，由于4字节对齐，所以为16
{
public:
        int a;                                  //4个字节
        char b;                                 //一个字节
        virtual void fun1(){ //虚函数，代表有虚表，因此会创建一个8个字节的虚表指针
        }
        virtual void fun2(){

        }        
        void f();                               //在代码段，不占字节。
        static int s_data ;                     //在静态区，不占字节
};

int main()
{
        int a;
        a = sizeof(A);
        cout<<a<<endl;

        A obj;
        cout<<sizeof(obj)<<endl;
        return 0;
}
```


存在虚函数的类都有一个一维的虚函数表叫虚表，虚表里存放的就是虚函数的地址了，因此，虚表是属于类的。这样的类对象的前8个字节是一个指向虚表的指针，类内部必须得保存这个虚表的起始指针。在32位的系统分配给虚表指针的大小为4个字节，我的电脑是64位，分配8个字节给指针，所以最后得到类A的大小为16．不管类里面有多少个虚函数，类内部只要保存虚表的起始地址即可，虚函数地址都可以通过偏移等算法获得。类的静态数据成员被编译器放在程序的一个global data members中，它是类的一个数据成员,但是它不影响类的大小,不管这个类实际产生了多少实例还是派生了多少新的类，静态成员数据在类中永远只有一个实体存在，而类的非静态数据成员只有被实例化的时候，他们才存在．但是类的静态数据成员一旦被声明，无论类是否被实例化，它都已存在．可以这么说，类的静态数据成员是一种特殊的全局变量.子类的大小是本身成员的大小再加上父类成员的大小.如果父类还有父类，也加上父类的父类，这样一直递归下去。
### 例2
```
#include<iostream>

using namespace std;

class A {};  //每个实例在内存中都有一个独一无二的地址，为了达到这个目的，编译器往往会给一个空类隐含的加一个字节。这样空类在实例化后在内存得到了独一无二的地址．所以sizeof( A )的大小为1．

class B    //类B的非虚成员函数是不计算在内的,不管它是否静态。
{
public:
  B() {}
  ~B() {}
  void MemberFuncTest( int para ) { }
  static void StaticMemFuncTest( int para ){  }
};


//类C有一个虚函数，存在虚函数的类都有一个一维的虚函数表叫虚表，虚表里存放的就是虚函数的地址了，因此，虚表是属于类的。这样的类对象的前四个字节是一个指向虚表的指针，类内部必须得保存这个虚表的起始指针。在64位的系统分配给虚表指针的大小为8个字节，所以最后得到类C的大小为8．
class C 
{
 C(){}
 virtual ~C() {}
};

class D{    //原理同类C，不管类里面有多少个虚函数，类内部只要保存虚表的起始地址即可，虚函数地址都可以通过偏移等算法获得。
 D(){}
 virtual ~D() {}
 virtual int VirtualMemFuncTest1()=0;
 virtual int VirtualMemFuncTest2()=0;
 virtual int VirtualMemFuncTest3()=0;
};


// 32或64位的操作系统int都占4个字节，char占一个字节，加上内存对齐的3字节，为8字节
class E  
{
 int  m_Int;
 char m_Char;
};

//类F的静态数据成员被编译器放在程序的一个global data members中，它是类的一个数据成员,但是它不影响类的大小,不管这个类实际产生了多少实例还是派生了多少新的类，静态成员数据在类中永远只有一个实体存在，而类的非静态数据成员只有被实例化的时候，他们才存在．但是类的静态数据成员一旦被声明，无论类是否被实例化，它都已存在．可以这么说，类的静态数据成员是一种特殊的全局变量.
class F : public E
{
 static int s_data ;
};
int F::s_data=100;

class G : public E
{
 virtual int VirtualMemFuncTest1(int para)=0;
 int m_Int;
};
class H : public G
{
 int m_Int;
};
//可以看出子类的大小是本身成员的大小再加上父类成员的大小.如果父类还有父类，也加上父类的父类，这样一直递归下去。

class I : public D  //父类子类工享一个虚函数指针，虚函数指针保留一个即可。
 virtual int VirtualMemFuncTest1()=0;
 virtual int VirtualMemFuncTest2()=0;
};

int main( int argc, char **argv )
{
 cout<<"sizeof( A ) = "<<sizeof( A )<<endl;
 cout<<"sizeof( B ) = "<<sizeof( B )<<endl;
 cout<<"sizeof( C ) = "<<sizeof( C )<<endl;
 cout<<"sizeof( D ) = "<<sizeof( D )<<endl;
 cout<<"sizeof( E ) = "<<sizeof( E )<<endl;
 cout<<"sizeof( F ) = "<<sizeof( F )<<endl;
 cout<<"sizeof( G ) = "<<sizeof( G )<<endl;
 cout<<"sizeof( H ) = "<<sizeof( H )<<endl;
 cout<<"sizeof( I ) = "<<sizeof( I )<<endl;

#if defined( _WIN32 )
 system("pause");
#endif
 return 0;
}
```

![图片](1.png)

C明空的类也是会占用内存空间的，而且大小是1，原因是C++要求每个实例在内存中都有独一无二的地址。
 （一）类内部的成员变量：
 普通的变量：是要占用内存的，但是要注意内存对齐（这点和struct类型很相似）。
 static修饰的静态变量：不占用内存，原因是编译器将其放在全局变量区。
 从父类继承的变量：计算进子类中
 （二）类内部的成员函数：
 非虚函数(构造函数、静态函数、成员函数等)：不占用内存。
 虚函数：要占用4个字节(32位的操作系统)8个字节（64位系统），用来指定虚拟函数表的入口地址。跟虚函数的个数没有关系。父类子类工享一个虚函数指针。

内存分布：
如果没有虚函数，对象的第一个成员变量的地址就是整个对象的地址，在内存中对象成员变量是按照类中声明的顺序排列的。如果有虚函数，为了让对象能调用虚函数，在每个对象最开始的内存位置添加一个虚函数表的指针_vfptr，其后才是对象成员变量内存数据。如果某个类是派生类，那么它的对象内存中最开始的地方实际上是基类对象的拷贝，包括基类虚函数表指针和成员变量，其后才是派生类自己的成员变量数据。
### 参考blog
c++多态详解 [https://www.cnblogs.com/dormant/p/5223215.html](https://www.cnblogs.com/dormant/p/5223215.html)
类内存分布 [https://blog.csdn.net/u014453898/article/details/53818725](https://blog.csdn.net/u014453898/article/details/53818725)
c++类所占内存大小计算 [https://blog.csdn.net/chenchong08/article/details/7620984](https://blog.csdn.net/chenchong08/article/details/7620984)
