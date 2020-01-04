---
title: STL sort函数实现详解
date: 2019-05-06 15:39:25
tags: [算法, STL]
---
## 问题
STL中的sort是怎么实现的为什么比我们自己写的sort要快那么多，接下来从源码的角度来看看STL中sort的实现。
## 分析
### 函数声明
sort函数有两个重载，一个使用默认的 < 操作符，另一个使用自定义的比较仿函数。
```
#include <algorithm>
 
template< class RandomIt >
void sort( RandomIt first, RandomIt last );
 
template< class RandomIt, class Compare >
void sort( RandomIt first, RandomIt last, Compare comp );
```
### 实现原理
原来，STL中的sort并非只是普通的快速排序，除了对普通的快速排序进行优化，它还结合了插入排序和堆排序。根据不同的数量级别以及不同情况，能自动选用合适的排序方法。当数据量较大时采用快速排序，分段递归。一旦分段后的数据量小于某个阀值，为避免递归调用带来过大的额外负荷，便会改用插入排序。而如果递归层次过深，有出现最坏情况的倾向，还会改用堆排序。

<!--more-->

### 普通快排

普通快速排序算法可以叙述如下，假设S代表需要被排序的数据序列：

1. 如果S中的元素只有0个或1个，结束。
2. 取S中的任何一个元素作为枢轴pivot。
3. 将S分割为L、R两端，使L内的元素都小于等于pivot，R内的元素都大于等于pivot。
4. 对L、R递归执行上述过程。

快速排序最关键的地方在于枢轴的选择，最坏的情况发生在分割时产生了一个空的区间，这样就完全没有达到分割的效果。STL采用的做法称为median-of-three，即取整个序列的首、尾、中央三个地方的元素，以其中值作为枢轴。
分割的方法通常采用两个迭代器head和tail，head从头端往尾端移动，tail从尾端往头端移动，当head遇到大于等于pivot的元素就停下来，tail遇到小于等于pivot的元素也停下来，若head迭代器仍然小于tail迭代器，即两者没有交叉，则互换元素，然后继续进行相同的动作，向中间逼近，直到两个迭代器交叉，结束一次分割。
```
template <class _RandomAccessIter, class _Tp>
_RandomAccessIter __unguarded_partition(_RandomAccessIter __first, 
                                        _RandomAccessIter __last, 
                                        _Tp __pivot) 
{
    while (true) {
        while (*__first < __pivot)
            ++__first;
        --__last;
        while (__pivot < *__last)
            --__last;
        if (!(__first < __last))
            return __first;
        iter_swap(__first, __last);
        ++__first;
    }
}
```

### 内省式排序IntroSort
不当的枢轴选择，导致不当的分割，会使快速排序恶化为 O(n2)。David R.Musser于1996年提出一种混合式排序算法：Introspective Sorting（内省式排序），简称IntroSort，其行为大部分与上面所说的median-of-three Quick Sort完全相同，但是当分割行为有恶化为二次方的倾向时，能够自我侦测，转而改用堆排序，使效率维持在堆排序的 O(nlgn)，又比一开始就使用堆排序来得好。

### 代码分析
下面是完整的SGI STL sort()源码（使用默认<操作符版）
```
template<typename _RandomAccessIterator, typename _Compare>
inline void
__sort(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp)
{
    if (__first != __last)
    {
        // __introsort_loop先进行一遍IntroSort，但是不是严格意义上的IntroSort。
        //因为执行完之后区间并不是完全有序的，而是基本有序的。
        //__introsort_loop和IntroSort不同的地方是，__introsort_loop会在一开始会判断区间的大小，当区间小于16的时候，就直接返回。
        std::__introsort_loop(__first, __last,
                std::__lg(__last - __first) * 2,
                __comp); 
        // 在区间基本有序的基础上再做一遍插入排序，使区间完全有序
        std::__final_insertion_sort(__first, __last, __comp);
    }
}
```
其中，__introsort_loop便是上面介绍的内省式排序，其第三个参数中所调用的函数__lg()便是用来控制分割恶化情况，代码如下：
```
template <class Size>
inline Size __lg(Size n) {
    Size k;
    for (k = 0; n > 1; n >>= 1) ++k;
    return k;
}
```
即求lg(n)（取下整），意味着快速排序的递归调用最多 2*lg(n) 层。
内省式排序算法如下：
```
enum { _S_threshold = 16 };

template<typename _RandomAccessIterator, typename _Size, typename _Compare>
void
__introsort_loop(_RandomAccessIterator __first,
         _RandomAccessIterator __last,
         _Size __depth_limit, _Compare __comp)
{
    while (__last - __first > int(_S_threshold))// 若区间大小<=16就不再排序
    {
        if (__depth_limit == 0)// 若递归次数达到限制，就改用堆排序
        {
            std::__partial_sort(__first, __last, __last, __comp);
            return;
        }
        --__depth_limit;
        _RandomAccessIterator __cut =
        std::__unguarded_partition_pivot(__first, __last, __comp); // 分割
        std::__introsort_loop(__cut, __last, __depth_limit, __comp); // 右半区间递归
        __last = __cut;
        // 回到while循环，对左半区间进行排序，这么做能显著减少__introsort_loop的调用的次数
    }
}
```
__final_insertion_sort代码：
```
template<typename _RandomAccessIterator, typename _Compare>
void
__final_insertion_sort(_RandomAccessIterator __first,
           _RandomAccessIterator __last, _Compare __comp)
{
    if (__last - __first > int(_S_threshold)) // 区间长度大于16
    {
        // 插入排序
        std::__insertion_sort(__first, __first + int(_S_threshold), __comp); 
        // 也是插入排序，只是在插入排序的内循环时，不再判断边界条件，因为已经保证了区间前面肯定有比待插入元素更小的元素
        std::__unguarded_insertion_sort(__first + int(_S_threshold), __last, 
                      __comp);
    }
    else // 区间长度小于等于16的话
        std::__insertion_sort(__first, __last, __comp); // 插入排序
}
```
1. 首先判断元素规模是否大于阀值__stl_threshold，__stl_threshold是一个常整形的全局变量，值为16，表示若元素规模小于等于16，则结束内省式排序算法，返回sort函数，改用插入排序。
2. 若元素规模大于__stl_threshold，则判断递归调用深度是否超过限制。若已经到达最大限制层次的递归调用，则改用堆排序。代码中的partial_sort即用堆排序实现。
3. 若没有超过递归调用深度，则调用函数__unguarded_partition()对当前元素做一趟快速排序，并返回枢轴位置。__unguarded_partition()函数采用的便是上面所讲的使用普通快排中两个迭代器交换的方法。
4. 经过一趟快速排序后，再递归对右半部分调用内省式排序算法。然后回到while循环，对左半部分进行排序。

递归上述过程，直到元素规模小于__stl_threshold，然后返回sort函数，对整个元素序列调用一次插入排序__final_insertion_sort，此时序列中的元素已基本有序，所以插入排序也很快。至此，整个sort函数运行结束。
## 参考
侯捷 STL源码剖析
[https://www.cnblogs.com/fengcc/p/5256337.html](https://www.cnblogs.com/fengcc/p/5256337.html)
[https://www.jianshu.com/p/50af00263200](https://www.jianshu.com/p/50af00263200)
