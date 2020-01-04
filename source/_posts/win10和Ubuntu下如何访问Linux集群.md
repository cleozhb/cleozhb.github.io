---
title: win10和Ubuntu下如何访问Linux集群
date: 2017-07-13 16:22:57
tags: Linux
---
linux系统中用ssh访问：
ssh远程登录集群 ssh zhb@192.168.122.1
文件传输使用scp /home/下载/aaa zhb@192.168.122.1:/export/zhb

Windows系统中可以利用现有的工具，十分方便。
下载安装Xshell和xftp，用来完成连接服务器和文件传输