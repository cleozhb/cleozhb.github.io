---
title: STL 条件查找——find_if用法
date: 2019-03-07 09:53:16
tags: STL 
---

find_if()函数，它接收一个**函数对象的参数**作为参数， 并使用它来做更复杂的评价对象是否和给出的查找条件相符。以下三个例子分别举例说明在map,vector,list中的用法，其实都是一样的，STL最大的好处就是用迭代器实现了容器和算法的分离，我们只需要在自己实现的类中实现仿函数，因为只有这个类本身知道该怎么样判定相等。构造一个函数对象参数传入find_if()函数。仿函数对象内部定义了要查找的条件，且返回类型必须为bool，客观反应在find_if()函数查找过程中的是否匹配。知道了这个，以后就不用再用遍历来查找元素了。
<!--more-->

### map跟据value查找的例子
```
#include <iostream>
#include <map>
#include <string>
#include <algorithm>
using namespace std;

class map_finder
{
public:
    map_finder( string cmp_string ) : m_string(cmp_string) {}
    bool operator () (const map<int,string>::value_type pair)
    {
        return pair.second == m_string;
    }
private:
    string m_string;
};

int main()
{
    map<int ,string> my_map;
    my_map.insert( make_pair(10,"china"));
    my_map.insert( make_pair(20,"usa"));
    my_map.insert( make_pair(30,"english"));
    my_map.insert( make_pair(40,"hongkong"));

    map<int,string>::iterator it = find_if(my_map.begin(),my_map.end(),map_finder("english"));
    if( it == my_map.end() )
        cout<<"not found!"<<endl;
    else
        cout<<"found key:"<<(*it).first<<", value:"<<(*it).second<<endl;
    return 0;
}
```

### vector查找的例子
这种方法也可以用于我们自己定义的结构体
```
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
using namespace std;

struct value_t
{
    int a;
    int b;
};

class vector_finder
{
public:
    vector_finder( const int a, const int b ) :m_v_a(a),m_v_b(b){}
    bool operator ()( vector<struct value_t>::value_type &value)
    {
        return (value.a==m_v_a)&&(value.b = m_v_b);
    }
private:
    int m_v_a;
    int m_v_b;
};

int main()
{
    vector<value_t> my_vector;
    value_t my_value;

    my_value.a = 11; my_value.b = 1001;
    my_vector.push_back(my_value);

    my_value.a = 12; my_value.b = 1002;
    my_vector.push_back(my_value);

    my_value.a = 13; my_value.b = 1003;
    my_vector.push_back(my_value);

    my_value.a = 14; my_value.b = 1004;
    my_vector.push_back(my_value);

    vector<value_t>::iterator it = find_if( my_vector.begin(), my_vector.end(), vector_finder(13,1003));
    if( it == my_vector.end() )
        cout<<"not found!"<<endl;
    else
        cout<<"found value a:"<<(*it).a <<", b:"<<(*it).b<<endl;
    return 0;
}
```

### list查找的例子
同样可以用于线性表结构的查询。假设我们的list中有一些按年代排列的包含了事件和日期的记录。我们希望找出发生在1997年的事件。
```
#include <iostream>
#include <string>
#include <list>
#include <algorithm>
using namespace std;

class EventIsIn1997 {
public: 
    bool operator () (string& EventRecord) {
        // year field is at position 12 for 4 characters in EventRecord
        return EventRecord.substr(11,4)=="1997";
        //return this->substr(11,4)=="1997"
    }
};

int main (void) {
    list<string> Events;

    // string positions 0123456789012345678901234567890123456789012345
    Events.push_back("07 January 1995 Draft plan of house prepared");
    Events.push_back("07 February 1996 Detailed plan of house prepared");
    Events.push_back("10 January 1997 Client agrees to job");
    Events.push_back("15 January 1997 Builder starts work on bedroom");
    Events.push_back("30 April 1997 Builder finishes work");

    list<string>::iterator EventIterator = find_if (Events.begin(), Events.end(), EventIsIn1997());

    // find_if completes the first time EventIsIn1997()() returns true
    // for any object. It returns an iterator to that object which we
    // can dereference to get the object, or if EventIsIn1997()() never
    // returned true, find_if returns end()
    if (EventIterator==Events.end()) {
        cout << "Event not found in list" << endl;
    }
    else {
        cout << *EventIterator << endl;
    }
}
```

### 参考
[https://blog.csdn.net/hj490134273/article/details/6051080](https://blog.csdn.net/hj490134273/article/details/6051080)
