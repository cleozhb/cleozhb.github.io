---
title: c++的单例模式与全局变量
date: 2018-07-07 12:12:42
tags: [单例模式, c++]
---
在之前的code review中因为使用了大量的全局变量被老师批的一无是处，于是痛定思痛，从网上查了全局变量的缺点以及如何减少甚至不用全局变量。单例模式就是全局变量的一种很好的代替方法。
简而言之就是创建一个单例类，保证这个类只有一个对象实例（将构造函数私有化），并提供一个访问的方法，然后用它的静态成员函数得到类的唯一对象实例。
考虑多线程，单例模式还要做很多调整，具体的在第二篇参考blog 中写的非常好;虽然是用java描述，但是转为c++也是相同的原理。
<!--more-->
### 哪里会用到单例模式
在程序开发过程中基本都会用到数据库，要操作数据库就需要建立数据库连接，建立一个数据库连接对程序的运行时间影响很大，其实一个数据库连接建立完成之后如果只是提供给一个数据库操作使用，等下一个操作出现时又创建一个新的连接就会很耗时。因此，假如一个数据库连接建立完成之后可以提供给后面所有的数据库操作使用，而不需要建立新的连接就会节省很多时间，这里就需要用到单例模式，保证程序中只有一个数据库连接实例。
随着程序开发越来越复杂，业务数据越来越多，在系统运行时一个数据库连接的实例显然就不够用 ，此时，数据库连接池的就应运而生，系统运行时可以存在多个数据库连接实例，把这些实例放在一个池子里，使用时从池子里拿出来，用完了再放回去，池子在系统运行时只有一个，因此数据库连接池也要采用单例模式，保证在系统运行时只有一个实例存在。
在程序开发中还会用到日志、还有包括java中的runtime类（与当前运行时环境有关，在当前JVM中应该只有一个这样的实例在运行）……这些都是单例模式的典型应用。
总结：当程序运行时，需要保证一个对象只有一个实例存在时，就应该要用到单例模式
![图片](1.png)
![图片](2.png)
### 一个简单的例子
```
//singleton.h

#ifndef __SINGLETON_HPP_
#define __SINGLETON_HPP_
#include <stddef.h>  // defines NULL
#include <iostream>
#include <cassert>

template <class T>
class Singleton
{
public:
//static修饰的函数是属于类的，所以没有this指针，
//所以static类成员函数不能访问非static的类成员，只能访问static修饰的类成员
  static T& Instance() {
      if(!m_pInstance){
        m_pInstance = new T;
        std::cout << "New Instance created" << std::endl;
      } 
      assert(m_pInstance != NULL);
      return *m_pInstance;
  }
protected:
  Singleton();
  ~Singleton();
private:
  //将复制构造函数和“=”操作符也设为私有，防止被复制
  Singleton(Singleton const&);
  Singleton& operator=(Singleton const&);
  static T* m_pInstance;
};

//类的静态成员，独立于一切类的对象存在，必须先在类外进行初始化。static修饰的变量先与类对象存在，所以必须要在类外先进行初始化。
//static修饰的变量在静态存储区生成
template <class T> T* Singleton<T>::m_pInstance=NULL;

#endif
```

```
//myclass.h
#include "singleton.h"
class MyClass 
{
public:
    MyClass() 
    {
        myArr = NULL;
        std::cout << "MyClass::MyClass()" << std::endl;
    }
    void setId(int Id)
    {
        myId = Id;
    }
    int getId()
    {
        return myId;
    }
    int* getArr()
    {
        return myArr;
    }
    void setArr(int* pp)
    {
        myArr = pp;
    }

    ~MyClass() 
    {
        std::cout << "MyClass::~MyClass()" << std::endl;
    }
private:
    int myId;
    int* myArr;
};
```
### 
```
//main.cpp
#include "myclass.h"
#include <iostream>

typedef Singleton<MyClass> MyClassSingleton;   // Global declaration

int main(){
    
    MyClass & obj = MyClassSingleton::Instance();
    obj.setId(1);
    int* p = obj.getArr();

    MyClass & obj1 = MyClassSingleton::Instance();
    obj1.setId(2);
    int* p1 = obj1.getArr();

    int i = 0;
    int*pp = new int[5];
    for(i=0;i<5;i++)
        pp[i] = i;
    obj1.setArr(pp);
    
    p = obj.getArr();
    p1 = obj1.getArr();

    return 0;
}
```
在代码中定义了两个MyClass类的实例，事实证明这两个实例其实是一个实例的不同名字的引用而已。所指向的地址一样，改变其中一个另一个会随之改变;
![图片](3.png)![图片](4.png)
### 参考blog
单例模式跟全局变量相比的好处[https://blog.csdn.net/ozdazm/article/details/8538014](https://blog.csdn.net/ozdazm/article/details/8538014)
java深入浅出单实例Singleton设计模式[https://blog.csdn.net/haoel/article/details/4028232](https://blog.csdn.net/haoel/article/details/4028232)
c++设计模式——单例模式[https://www.cnblogs.com/tianzeng/p/9062008.html](https://www.cnblogs.com/tianzeng/p/9062008.html)
单例模式C++实现 [https://www.cnblogs.com/cxjchen/p/3148582.html](https://www.cnblogs.com/cxjchen/p/3148582.html)
