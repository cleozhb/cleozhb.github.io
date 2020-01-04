---
title: 怎样在Linux系统中快速传输大文件
date: 2017-07-10 00:36:22
categories: 技术 
tags: Linux
---
这篇文章已经没有用了，为了纪念我折腾的过程留下来。我发现移动硬盘在插入Ubuntu系统的USB接口后没有反应，是因为移动硬盘在Windows中设置了密码，用BitLocker加密过。因为是Windows的加密程序，Linux没有图形界面让用户输入密码，故无法解密。在终端输入命令可以检测到硬盘，但是并没有读写权限。硬盘厂商可能没有考虑到Windows和Linux的衔接问题，毕竟Linux是少量用户。所以，先在Windows中给移动硬盘解密，然后就可以在Linux中使用了。（吐血，因为开始完全没考虑到这个问题，还在想各种别的办法）
由于此次要传输的数据量很大，有1.2T，才开始学习Linux,受到windows中的影响开始的时候我尝试了用2T 的移动硬盘来拷贝数据。然而通过尝试发现，与windows不同，windows对于移动硬盘与U盘都是自动完成挂载的，在ubuntu 系统中并没有这么智能，U盘还好，移动硬盘想要挂载成功读数据已经很不容易，若还要向里面写数据几乎不可能。
<!--more-->
于是通过查阅资料，可以通过网络传输的方式来进行数据的拷贝。采用scp在服务器之间拷贝文件，首先要做的是将被访问的电脑变成服务器，安装**SSH(Secure Shell)**服务以提供远程管理服务,并允许root权限远程登录，需要开启ssh服务支持scp远程登录。SSH分为客户端和服务端，需要安装SSH server 

**一、检查是否开启SSH服务**
因为Ubuntu默认是不安装SSH服务的，所以在安装之前可以查看目前系统是否安装，通过以下命令：

``` bash
ps -e|grep ssh
```

输出结果sshd表示 ssh-server 启动。
**二、安装SSH服务**
``` bash
sudo apt-get install ssh
```

或者
``` bash
sudo apt-get install openssh-client 客户端
sudo apt-get install openssh-server 服务端
```

**三、启动SSH服务**
``` bash
sudo /etc/init.d/ssh start
```
**四、修改SSH配置**
包括修改端口、是否允许root登录等设置，这一步必须要做，如果不做，会报错 SSH服务器拒绝了密码，请再试一次。应该是应该是sshd的设置不允许root用户用密码远程登录修改`/etc/ssh/sshd_config`文件，注意，安装了`openssh`才会有这个文件，如果文件不存在请检查是否安装了`openssh`。
配置文件的位置：`/etc/ssh/sshd_config`
使用 gedit 修改配置文件： 
``` bash
sudo gedit /etc/ssh/sshd_config
```
找到 ***PermitRootLogin prohibit-password***
改成 ***PermitRootLogin yes***
**五、重启SSH服务**
``` bash
sudo /etc/init.d/ssh restart
```
**六、改写要读取的文件夹的权限**
``` bash
chmod 776 /home/yuyin 
```
**七、链接服务器，传输文件**
通过`ifconfig -a`来查看服务器的ip地址
远程服务器`ip:192.168.2.110`
*在客户端远程链接服务器*
``` bash
ssh -l root 192.168.2.110
```
*打开新的终端,从服务器上下载整个目录*
``` bash
scp -r root@192.168.2.110:/home/yuyin/work /home/bokebi/yuyin
```
*上传目录到服务器*
``` bash
scp -r /home/bokebi/mytest root@192.168.2.110:/home/yuyin/work
```
*从服务器上下载*
``` bash
scp root@192.168.2.110:/home/yuyin/work /home/bokebi
```
*将文件上传到服务器:将test.txt上传到192.168.2.110服务器上的work目录下*
``` bash
scp /home/bokebi/test.txt root@192.168.2.110:/home/yuyin/work
```
使用scp命令可以用来通过安全、加密打连接在机器之间传输文件，与rcp相似
**参考文章**
慕课网：linux达人养成计划1
[ubuntu SSH 连接、远程上传下载文件](http://blog.csdn.net/arnoldlu/article/details/17394237)
[Ubuntu下SSH远程连接、文件传输](http://blog.csdn.net/tfc_l/article/details/51722128)