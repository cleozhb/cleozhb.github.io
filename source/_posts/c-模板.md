---
title: c++模板
date: 2018-08-26 12:25:24
tags: [c++, 模板]
---
### 概念
* 模板：模板不是函数或者类，并不能直接拿来使用生成实实在在的对象，也就是没有实体。模板跟据你所给它的类型参数帮你生成一个可以拿来使用的实例对象。
* 模板实例化（instantiation）：给模板类型参数之后，编译器由模板自动创建实例类或函数的过程叫做模板的实例化。实例化的结果是产生一个具体的类或者函数。这时候虽然没有创建类的对象，编译器将生成类声明。在一些编译器中，模板只有在被实例化时编译器才能检查其语法正确性，如果写了一个模板但没有用到，那么编译器是不会报告这个模板中的语法错误的。

<!--more-->

* 模板特化（也叫具体化specialization）：针对某些特定的具体类型，对模板进行修改，使其对于该类型的实例对象的行为不同。比如定义一个swap函数，用于交换两个对象。现在对于某个类型的对象交换时只交换其中的一个属性，就不能直接使用原先的模板函数。原先的模板函数会交换整个对象，这时候就需要对模板进行特化，实现对该类型实例对象不同的行为。模板具体化的结果是从类模板产生一个具体的类（类似于声明）。
```
template <typename T1, typename T2>
class Pair
{
  ...
};

Pair <string, string > strPair; //隐式实例化对象 implicit instantiation

template class Pair<int,double>;  //显式实例化模板 explicit instantiation

//显式具体化模板 explicit specialization
template <> class Pair<const char* , int> 
{
  ...
};

//部分具体化模板 partial specialization
template  <typename T1> class Pair<typename T1, int>
{
  ...
};
```
### 例子
```
template<typename T>
void func(){
    vector<int>::iterator it_int;
    vector<T>::iterator it;
}

编译后提示:[Error] need ‘typename’ before ‘std::vector::iterator’ because ‘std::vector’ is a dependent scope
```
func函数的第一行，vector<int>::iterator it_int;是可以编译通过的，而第二行，用模板参数T申明的却不能通过。它们的区别在于vector<int>是实例化之后的一个类型，而vector<T>还是一个模板类型，如何实例化还取决于外部实际传进来的参数类型T。
正如编译器编译的提示dependent scope，指出了这个类型还需要别的依赖条件。但是为什么编译器不能直接跟据模板类型的定义来实现这个声明呢？原因是编译器无法识别std::vector<T>::iterator这个名称是一个成员变量还是一个类型。加入这个类被特化了，例如vector的特化类定义如下：
```
template <>
class vector<char>{
    int iterator;
    // ..detail omitted
}
```
### 那么在std::vector<char>::iterator这个名称就是一个成员变量了，不能当作类型使用。因此为了防止出现这样的歧义，正如编译器的提示，需要在std::vector<T>::iterator前面加一个关键字typename来显式的说明这是一个类型而不是一个成员变量。

模板类中可以使用虚函数吗？
在模板类中使用虚函数与在非模板类中使用虚函数完全一样。
```
template class<T>
class A
{
public:
  virtual void f1(){cout<<"A is called"<<endl;}  //虚函数
  virtual void f2()=0{cout<<"A=0"<<endl;}  //纯虚函数
};

template class<T>
class B: public A<T>
{
public:
  void f1(){cout<<"B is called"<<endl;}
  void f2(){cout<<"B!=0"<<endl;}
};

void main()
{
  A<int>* p=new B<int>;
  p->f1();                   //输出B is called,虚函数成功
}
```
需要注意的是A<int>和A<char>是两个完全不同的类。不能写A<int>* p = new B<char>;因为A<int>不是B<char>的基类。所以得出结论：模板不影响类的多态。
### 模板成员函数可以是虚函数吗？
模板成员函数不可以是虚函数！
原因：
编译器期望在处理类的定义的时候就能确定这个类的虚函数表的大小，如果允许有类的虚成员模板函数，那么就必须要求编译器提前知道程序中所有对该类的虚成员模板函数的调用。在实例化模板类时，需要创建虚函数表 virtual table， 在模板类实例化完成之前不能确定函数模板会被实例化多少个（包括虚函数模板）。普通成员函数模板无所谓，什么时候需要什么时候编译器就实例化。编译器不用知道到底需要实例化多少个，虚函数的个数必须知道，否则这个类无法被实例化（因为要创建 virtual table）。因此目前不支持虚函数模板继承。

### 模板嵌套
一个模板可以采用本身就是模板名称的参数
例如
```
template <typename T, template <typename> class Cont>
class Stack;
```
这个栈模板的第二个参数Cont是模板的模板参数，Cont是一个具有单个类型参数的类模板的名字。这里没有给Cont的类型名称参数命名。也可以像下面这样写：
```
template <typename T, template <typename ElementType> class Cont>
class Stack;
ElementType这样的名称可以被省略，但是为了提高程序的可读性，应该写上。Stack模板使用类型名参数来实例化其模板参数的模板。生成的容器类型用于实现堆栈。就像下面这样：
template <typename T, template <typename> class Cont>
class Stack {
    //...
  private:
    Cont<T> s_;
};
```
这种方法允许通过Stack本身的实现来协调元素与容器。保证了元素类型与保存元素的容器之间的一致性。用这种方法来处理一组模板参数与要用这一组参数实例化的模板的好方法。

### 参考
* 模板的编译错误need 'typename' before *** because *** is a dependent scope [https://blog.csdn.net/pb1995/article/details/49532285](https://blog.csdn.net/pb1995/article/details/49532285)
* 详谈模板实例化和具体化 [https://blog.csdn.net/gettogetto/article/details/79439577](https://blog.csdn.net/gettogetto/article/details/79439577)
* 模板类中可以使用虚函数吗？模板成员函数可以是虚函数吗？[https://blog.csdn.net/zzuchengming/article/details/51763563](https://blog.csdn.net/zzuchengming/article/details/51763563)
* C++虚函数表分析[https://www.cnblogs.com/hushpa/p/5707475.html](https://www.cnblogs.com/hushpa/p/5707475.html)
* C++：模板参数的模板 [http://www.informit.com/articles/article.aspx?p=376878](http://www.informit.com/articles/article.aspx?p=376878)


