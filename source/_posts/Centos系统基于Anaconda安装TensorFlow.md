---
title: Centos系统基于Anaconda安装TensorFlow
date: 2017-07-15 23:54:40
categories: 环境配置 
tags: [linux,tensorflow,Centos]
---
Ubuntu下远程访问linux集群服务器（系统版本：centos6）
这几天一直在折腾安装tensorflow，先在Ubuntu系统上装了一次，本以为在Centos系统上也是一样，没有想到遇到了更多的问题，安装过程记录如下。
``` bash
ssh zhb@192.168.122.1
su root
```

<!--more-->

切换用户，在自己的用户下操作  命令su - [username]选择用户，例如：su - zhb
![图1](http://oqadn1oza.bkt.clouddn.com/1%E8%BF%9C%E7%A8%8B%E8%BF%9E%E6%8E%A5centos.jpg)
![图2](http://oqadn1oza.bkt.clouddn.com/2%E8%BF%9C%E7%A8%8B%E8%BF%9E%E6%8E%A5centos%E6%8E%88%E6%9D%83root.jpg)
 
## 首先安装Anaconda
下载anaconda安装包，默认安装位置是/home/zhb/anaconda2
输入 
``` bash
bash Anaconda2-4.2.0-Linux-x86_64.sh
```
按照提示完成安装，然后输入python测试，发现安装了anaconda后，在终端输入python依然是Linux自带的版本，这是因为.bashrc的更新没有生效
命令行中输入：
``` bash
source ~/.bashrc
```
然后再输入python，验证是否安装成功。

## 安装cuda 
查看系统版本信息，包括位数、版本号、CPU具体型号等
``` bash
uname -a (Linux查看版本当前操作系统内核信息)
cat /proc/version (Linux查看当前操作系统版本信息)
```
在官网上下载对应的cuda版本，上传到服务器并运行以下命令
``` bash
sudo sh cuda_8.0.61_375.26_linux.run
```
第一个是否安装驱动选择no，其他的选择yes
![图3](http://oqadn1oza.bkt.clouddn.com/3cuda%E5%AE%89%E8%A3%85%E9%80%89%E9%A1%B9.jpg)
安装完成后显示：
![图4](http://oqadn1oza.bkt.clouddn.com/4cuda%E5%AE%89%E8%A3%85%E7%BB%93%E6%9E%9C.jpg)
安装完毕后，再声明一下环境变量，并将其写入到 ~/.bashrc 的尾部:
``` bash
export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
检查驱动版本,这里显示驱动版本高于最低要求361.00，满足运行要求。
``` bash
lspci | grep -i vga
```
![pic5](http://oqadn1oza.bkt.clouddn.com/5%E6%9F%A5%E7%9C%8B%E9%A9%B1%E5%8A%A8%E7%89%88%E6%9C%AC.jpg)
测试cuda例子
``` bash
cd /root/NVIDIA_CUDA-8.0_Samples/1_Utilities/deviceQuery
make
```
![pic6](http://oqadn1oza.bkt.clouddn.com/6%E6%B5%8B%E8%AF%95cuda-1.jpg)
![pic7](http://oqadn1oza.bkt.clouddn.com/7%E6%B5%8B%E8%AF%95cuda-2.jpg)
![pic8](http://oqadn1oza.bkt.clouddn.com/8%E6%B5%8B%E8%AF%95cuda-3.jpg)
测试通过
## 安装cudnn
``` bash
tar xvzf cudnn-8.0-linux-x64-v5.1.tgz 
sudo cp cuda/include/cudnn.h /usr/local/cuda/include 
sudo cp cuda/lib64/libcudnn.so* /usr/local/cuda/lib64 
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn.so*
```
上面第2、3行就是把Cudnn的头文件和库文件复制到Cuda路径下的include和lib目录。
![pic9](http://oqadn1oza.bkt.clouddn.com/9%E5%AE%89%E8%A3%85cudnn.jpg)
Vi模式修改vi ~/.bash_profile文件，添加环境变量
``` bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64" export CUDA_HOME=/usr/local/cuda
source ~/.bash_profile #使更改的环境变量生效
```
## 使用anaconda安装tensorflow
首先新建一个conda 环境，命名为tensorflow
``` bash
conda create -n tensorflow python=2.7
```
激活环境，并在该环境下安装tensorflow
``` bash
source activate tensorflow
```
在电脑上下载tensorflow的安装文件，打开新的终端用以下命令将文件传到服务器
``` bash
scp /home/bokebi/下载/tensorflow-0.11.0rc1-cp27-none-linux_x86_64.whl zhb@192.168.122.1:/export/home/zhb
```
![pic10](http://oqadn1oza.bkt.clouddn.com/10%E8%BF%9C%E7%A8%8B%E4%BC%A0%E8%BE%93%E6%96%87%E4%BB%B6%E7%A4%BA%E4%BE%8B.jpg)
也可以选择在线安装的方式：
``` bash
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0rc1-cp27-none-linux_x86_64.whl 
pip install –ignore-installed –upgrade $TF_BINARY_URL
```
报错如下：
![pic11](http://oqadn1oza.bkt.clouddn.com/11%E5%AE%89%E8%A3%85TensorFlow%E6%8A%A5%E9%94%99.jpg)
于是改为
``` bash
pip install $TF_BINARY_URL
```
![pic12](http://oqadn1oza.bkt.clouddn.com/12%E5%AE%89%E8%A3%85TensorFlow.jpg)
配置tensorflow
``` bash
cd ~/tensorflow #切换到tensorflow文件夹
./configure     #执行configure文件 
```
安装完成但是import tensorflow时报错
ImportError: /lib64/libc.so.6: version `GLIBC_2.17' not found (required by /export/App/anaconda3/envs/ml2/lib/python2.7/site-packages/tensorflow/python/_pywrap_tensorflow.so)
上面的问题主要是glibc的版本太低，和tensorflow编译使用的glibc环境不一样，升级glibc是一个危险的动作，可能会造成系统无法运行。参考文章6，7解决
查看glibc版本：$strings /lib64/libc.so.6 |grep GLIBC_ 
在更新前可以看到只支持到2.12,采用如下操作：

[下载glibc地址](http://ftp.gnu.org/pub/gnu/glibc/)我们这就只安装符合的版本就行了。
采用直接升级解决安装glibc2.17的问题：
``` bash
wget http://ftp.gnu.org/pub/gnu/glibc/glibc-2.17.tar.xz
xz -d glibc-2.17.tar.xz
tar -xvf glibc-2.17.tar
cd glibc-2.17
mkdir build
cd build
```
运行configure配置
``` bash
../configure --prefix=/usr --disable-profile --enable-add-ons --with-headers=/usr/include --with-binutils=/usr/bin
```
然后编译安装
验证是否更新成功，如果运行程序有相似的报错，采用同样的方法进行更新。
![pic13](http://oqadn1oza.bkt.clouddn.com/13%E5%8D%87%E7%BA%A7GLIBC.jpg)
报错
![pic14](http://oqadn1oza.bkt.clouddn.com/14%E6%B5%8B%E8%AF%95python%E7%A8%8B%E5%BA%8F%E6%8A%A5%E9%94%99.jpg)
添加环境变量
``` bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda
```
简单测试tensorflow是否安装成功
``` python
import tensorflow as tf
import os
import json
import requests
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print sess.run(hello)
a = tf.constant(10)
b = tf.constant(32)
print sess.run(a+b)
```
检查cudnn是否安装正确，以及tensorflow在计算时是否真的将计算分配到GPU上啦。注意在上面的hello world中，尽管cudnn并未安装正确，程序只会报一个libcudnn并未找到的警告，程序还会继续正常去行。要检查cudnn是否正确安装，需要使用用到cudnn的库，可以用下面的代码来检查：
``` python
import tensorflow as tf
import os
import json
import requests
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
input = tf.Variable(tf.random_normal([100, 28, 28, 1]))
filter = tf.Variable(tf.random_normal([5, 5, 1, 6]))
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.initialize_all_variables())
op = tf.nn.conv2d(input, filter, strides = [1, 1, 1, 1], padding = 'VALID')
out = sess.run(op)
```
## 参考文章：
1、[http://www.linuxidc.com/Linux/2017-01/139319.htm](http://www.linuxidc.com/Linux/2017-01/139319.htm)
2、[https://baijiahao.baidu.com/s?id=1570349328882915&wfr=spider&for=pc](https://baijiahao.baidu.com/s?id=1570349328882915&wfr=spider&for=pc)
3、[http://blog.csdn.net/zhaoyu106/article/details/52793183/](http://blog.csdn.net/zhaoyu106/article/details/52793183/)
4、[https://zhuanlan.zhihu.com/p/22410507](https://zhuanlan.zhihu.com/p/22410507)
5、[http://www.linuxidc.com/Linux/2017-03/142291.htm](http://www.linuxidc.com/Linux/2017-03/142291.htm)
6、[http://blog.csdn.net/zhangweijiqn/article/details/53199553](http://blog.csdn.net/zhangweijiqn/article/details/53199553)
7、[http://jingyan.baidu.com/article/b7001fe1b59dd40e7282ddb7.html](http://jingyan.baidu.com/article/b7001fe1b59dd40e7282ddb7.html)
8、[http://blog.csdn.net/jteng/article/details/52975247](http://blog.csdn.net/jteng/article/details/52975247)