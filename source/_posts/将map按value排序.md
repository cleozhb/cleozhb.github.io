---
title: 将map按value排序
date: 2019-03-02 15:17:04
tags: [map, STL]
---
我们知道map的底层是红黑树，可以保证map内部的数据都是有序的。红黑树是一种二叉查找树，但在每个节点上增加了一个位表示颜色，通过对任何一条从跟到叶子的路径上各个节点的着色方式的现在，红黑树确保没有一条路径会比其它路径长出两倍，所以是接近平衡的，也就保证了红黑树的查找、插入、删除的时间复杂度最坏为O(logn)。map的排序默认按照key从小到大排序。有两个常用的功能需要注意：
1. key是一个结构体，按照key从大到小排序
2. 想按value排序

map是STL中的一个模板类
```
template <class Key, class Value, class Compare = less<Key>,
       class Allocator = allocator<pair<const Key, Value>>> class map;
```
这个类有4个模板参数，Key和Value是我们比较熟悉的，最后一个是allocator分配器，用来定义存储分配。Compare这个参数也是一个class类型的，提供默认值less<Key>。这个参数决定了map中元素的排序。接下来来解决刚刚说到的两个问题

<!--more-->

1. key是一个结构体，按照key从大到小排序

对于内置类型,其内部实现了<操作符重载。想要从大到小排序只需要写
```
map<string, int, greater<string>> mapStudent;
```
key是结构体的，如果没有重载<号，就会导致insert函数在编译时无法编译成功。下面实现一个将学生按iID排序，如果iID相等的话，按strName排序。mapStudent的key是StudentInfo类型的。要重载StudentInfo的<号才能正常的插入。
```
#include <map>
#include <string>
#include <iostream>
using namespace std;
typedef struct tagStudentInfo  
{  
    int iID;  
    string  strName;  
    bool operator < (tagStudentInfo const& r) const {  
        //这个函数指定排序策略，按iID排序，如果iID相等的话，按strName排序  
        if(iID < r.iID)  return true;  
        if(iID == r.iID) return strName.compare(r.strName) < 0;  
        return false;
    }  
}StudentInfo;//学生信息 
int main(){
    /*用学生信息映射分数*/  
    map<StudentInfo, int>mapStudent;  
    StudentInfo studentInfo;  
    studentInfo.iID = 1;  
    studentInfo.strName = "student_one";  
    mapStudent[studentInfo]=90;
    studentInfo.iID = 2;  
    studentInfo.strName = "student_two"; 
    mapStudent[studentInfo]=80;
    map<StudentInfo, int>::iterator iter=mapStudent.begin();
    for(;iter!=mapStudent.end();iter++){
        cout<<iter->first.iID<<" "<<iter->first.strName<<" "<<iter->second<<endl;
    }
    return 0;
}
```
![图片](1.png)
2. 按value排序

将map按value排序，第一反应是利用STL中的sort算法实现，但是sort只能对序列容器（vector, deque, list）进行排序。map是个集合容器，里面存储的元素是pair，底层的红黑树不是线性存储，所以不能用sort直接和map结合进行排序。但是可以间接进行，先将map中的元素放入vector中，然后再对这些元素进行排序。这个想法看似可行，sort排序的一个基本要求就是元素是可比较的，也就是实现了 < 操作的。map中的元素类型是pair，具体定义如下：
```
template <class T1, class T2> struct pair
{
    typedef T1 first_type;
    typedef T2 second_type;


    T1 first;
    T2 second;
    
    pair():first(T1()), second(T2()) {}
    
    pair(const T1& x, const T2& y) : first(x), second(y){}
    
    template<class U, class V>
    pair(const pair<U,V>& p) : first(p.first), second(p.second){}
}
```
pair 也是一个模板类，这样就实现了良好的通用性。它仅仅有两个数据成员 first和second，在<utility>文件中为pair重载了<运算符.具体实现如下
```
template <class _T1, class _T2>
inline bool operator < (const pair<_T1, _T2>& __x, const pair<_T1, _T2>& __y)
{
    return (__x.first <__y.first)||
           (!(__x.first <__y.first) && __x.second <__y.second);
}
```
这个实现中有一处非常巧妙，(__x.first <__y.first)  || !(__x.first <__y.first)  这一句等价于 __x.first == __y.first， 可是为什么不直接用__x.first ==__y.first呢？这样写看似费解，其实是有道理的，前面讲过，作为map的key必须实现 <操作符的重载，但是并不保证 == 操作符也被重载了，如果key 没有提供 == ，那么__x.first ==__y.first的写法就不对。
从上面这个pair 的实现中可以看到，它是按照先对key进行比较，key相等的时候才对value进行比较。显然不能满足按value进行排序的要求。而且，既然已经对pair重载了 < 运算符，也不能修改其实现，不能在外部重复实现重载 < 运算符。那么要怎样实现对pair按照value进行比较呢？可以写一个比较函数或一个仿函数来实现。
```
#include <map>
#include <vector>
#include <string>
#include <iostream>
using namespace std;
typedef pair<string, int> PAIR;   
bool cmp_by_value(const PAIR& lhs, const PAIR& rhs) {  
  return lhs.second < rhs.second;  
}  
struct CmpByValue {  
  bool operator()(const PAIR& lhs, const PAIR& rhs) {  
    return lhs.second < rhs.second;  
  }  
};
int main(){  
  map<string, int> name_score_map;  
  name_score_map["LiMin"] = 90;  
  name_score_map["ZiLinMi"] = 79;  
  name_score_map["BoB"] = 92;  
  name_score_map.insert(make_pair("Bing",99));  
  name_score_map.insert(make_pair("Albert",86));  
  /*把map中元素转存到vector中*/   
  vector<PAIR> name_score_vec(name_score_map.begin(), name_score_map.end());  
  sort(name_score_vec.begin(), name_score_vec.end(), CmpByValue());  
  /*sort(name_score_vec.begin(), name_score_vec.end(), cmp_by_value);也是可以的*/ 
  for (int i = 0; i != name_score_vec.size(); ++i) {  
    cout<<name_score_vec[i].first<<" "<<name_score_vec[i].second<<endl;  
  }  
  return 0;  
}
```
![图片](2.png)
要对map中的元素按照value进行排序，先将map的元素按照pair形式插入到vector中，然后对vector写个信的比较函数，这样就可以实现按照map的value排序了.
