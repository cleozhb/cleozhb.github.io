---
title: socket-python
date: 2018-07-09 16:31:25
tags: socket
---
![图片](1.png)

<!--more-->

![图片](2.png)
客户端发送SYN的报文给服务器端同时设置随机数序号X

服务器端接收到客户端的报文之后，经过处理，给客户端返回SYN+ACK的报文，同时设置随机数序号Y，此时返回的报文确认为ACK=X+1

接收到报文的客户端会在处理之后再发送一个报文给服务器端此时确认为ACK = Y+1

服务器接收到客户端发送的报文之后在服务器端与客户端之间形成一个通路。


![图片](3.png)
服务器端通过socket过程去绑定监听的IP以及监听的端口
客户端通过socket程序发出连接请求给服务器端
服务器端接收客户端发出的连接请求会返回回应报文给客户端
客户端受到服务器端发来的回应报文经过处理发送一个ACK报文给服务器端
服务器端接收到客户端发来的报文在服务器端与客户端之间建立通路
数据在通路中进行传输

![图片](4.png)
![图片](5.png)

socket_server.py
```
# coding:utf-8
import socket
# 服务器端socket
# 创建实例
sk = socket.socket()

# 定义绑定的IP和端口port
ip_port = ("127.0.0.1",8888)

# 绑定监听
sk.bind(ip_port)

# 最大连接数,socket进行监听后的传入连接，
# socket是阻塞的，也就是说同一时刻只有1个程序在进行处理
# 这里的5是说socket可以挂起的最大连接数是5，同时有5个请求来可以同时进行回应
# 正在处理的程序只有一个，其他的4个程序只能等着
# 如果多于5个，服务器会直接拒绝第六个请求，
# 最大连接数最小设置为1,大多数应用程序设置为5
sk.listen(5)

#不断循环，不断接收数据
while True:
    # 接收数据 ，传回两个参数，一个是连接对象，一个是传输地址
    # 接收数据是阻塞状态
    print "waiting....."
    conn,address = sk.accept()

    # 定义信息
    msg = "success!"

    # 返回信息
    # python3.x以上，网络数据的发送和接收都是byte类型
    # 如果发送的数据是str型的则需要进行编码
    conn.send(msg)
    # 此时数据已经被发送给了客户端

    # 主动关闭连接
    conn.close()
```


socket_client.py

```
# coding:utf-8
import socket

# 实例初始化
client = socket.socket()

# 访问的服务器端的IP和端口
ip_port = ("127.0.0.1",8888)

# 连接主机
client.connect(ip_port)

# 接收主机信息,每次接收1024个字节
data = client.recv(1024)

print data
```

上面为服务器端和客户端的代码，只是做到了客户端发送请求，服务器端返回数据，要如何做到客户端和服务器端真正的实时交互呢

下面的代码加上了while true循环，来实现客户端与服务器端的实时通信。但是这里的连接是TCP连接，所以不能同时开启两个client端，服务器会直接拒绝连接，说明这个通信过程是阻塞的，当将第一个客户端连接关闭之后，刚刚阻塞的第二个客户端连接会直接连接上服务器。如果使用的是UDP连接，则没有三次握手建立通道的过程，所以可以同时开启多个客户端

**socket_server.py TCP协议实时通信**

```
# coding:utf-8
import random
# from random import randint
import socket
# 服务器端socket
# 创建实例
sk = socket.socket()

# 定义绑定的IP和端口port
ip_port = ("127.0.0.1",8889)

# 绑定监听
sk.bind(ip_port)

# 最大连接数,socket进行监听后的传入连接，
# socket是阻塞的，也就是说同一时刻只有1个程序在进行处理
# 这里的5是说socket可以挂起的最大连接数是5，同时有5个请求来可以同时进行回应
# 正在处理的程序只有一个，其他的4个程序只能等着
# 如果多于5个，服务器会直接拒绝第六个请求，
# 最大连接数最小设置为1,大多数应用程序设置为5
sk.listen(5)

#不断循环，不断接收数据
while True:
    # 接收数据 ，传回两个参数，一个是连接对象，一个是传输地址
    # 接收数据是阻塞状态
    print "waiting....."
    conn,address = sk.accept()

    # 定义信息
    msg = "success!"

    # 返回信息
    # python3.x以上，网络数据的发送和接收都是byte类型
    # 如果发送的数据是str型的则需要进行编码
    conn.send(msg)
    # 此时数据已经被发送给了客户端

    # 进入一个while true循环，不断的接收客户端发来的消息
    while True:
        # 接收客户端消息
        data = conn.recv(1024)
        print data
        # 接收到退出的指令
        if data == 'exit':
            break
        # 处理客户端的数据
        conn.send(data)
        # 发送一个随机数
        conn.send(str(random.randint(1,1000)))
    # 主动关闭连接
    conn.close()
```


**socket_client.py TCP协议实时通信**

```
# coding:utf-8
import socket

# 实例初始化
client = socket.socket()

# 访问的服务器端的IP和端口
ip_port = ("127.0.0.1",8889)

# 连接主机
client.connect(ip_port)

# 定义一个循环，不断发送消息
while True:
    # 接收主机信息,每次接收1024个字节
    data = client.recv(1024) 
    print data

    # 定义发送的信息
    msg_input = raw_input("input:")
    # 消息发送
    client.send(msg_input)
    if msg_input == "exit":
        break
    data = client.recv(1024)
    print data
```


**socket_serverUDP.py UDP协议实时通信**

```
# coding:utf-8
import random
# from random import randint
import socket
# 服务器端socket
# 创建实例
sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 定义绑定的IP和端口port
ip_port = ("127.0.0.1",8889)
sk.bind(ip_port)
while True:
    data = sk.recv(1024)
    print data
```

**socket_clientUDP.py UDP协议实时通信**
```    
# coding:utf-8
import random
# from random import randint
import socket
# 服务器端socket

sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 定义绑定的IP和端口port
ip_port = ("127.0.0.1",8889)

while True:
    msg_input = raw_input("input:")
    if msg_input == "exit":
        break
    sk.sendto(msg_input,ip_port)

sk.close()
```

## socket的非阻塞
在TCP模式下只能有有限个连接，但是我们平时从来没有看到有阻塞的情况。
![图片](6.png)

在编写服务器端程序的时候，使用SocketServer包，打开SocketServer的源码可以看到它是通过多线程的方式来进行多个客户端通过TCP连接服务器端的 。

socket_serverTCP.py

```
# coding:utf-8
import SocketServer
import random
from random import randint
# 定义一个类

class MyServer(SocketServer.BaseRequestHandler):
    # 如果handle出现报错，会跳过
    # setup 和finish无论如何都会执行
    # 首先执行setup
    # 然后执行handle
    # 最后执行finish
    def setup(self):
        pass

    def handle(self):
        conn = self.request
        msg = "hello"
        conn.send(msg)
        while True:
            data = conn.recv(1024)
            print data
            if data == "exit":
                break
            conn.send(data)
            conn.send(str(random.randint(1,1000)))
        conn.close()
    def finish(self):
        pass

if __name__ == "__main__":
    # 创建多线程实例
    server = SocketServer.ThreadingTCPServer(("127.0.0.1",8888),MyServer)
    # 开启socketserver的异步多线程，等待客户端的连接
    server.serve_forever()
```

## **用socket实现文件的上传和接收：**
客户端上传文件socketSendFile.py，服务器端接收文件socketRecvFile.py
socketSendFile.py

```
# coding:utf-8
import socket
sk = socket.socket()
ip_port = ('127.0.0.1',9999)
sk.connect(ip_port)

with open('socket_serverUDP.py','rb') as f:
    for i in f:
        sk.send(i)
        data = sk.recv(1024)
        if data != 'success':
            break
# 给服务器端发送结束信号
sk.send('quit')
```


socketRecvFile.py

```
# coding:utf-8
import socket
sk = socket.socket()
ip_port = ('127.0.0.1',9999)

sk.bind(ip_port)
sk.listen(5)
# 进入循环接收数据
while True:
    # 等待客户端连接
    conn,address = sk.accept()
    # 一直使用当前连接进行数据接收，知道结束标志出现
    while True:
        # 打开文件，等待数据写入
        with open("file","ab") as f:
            data = conn.recv(1024)
            if data == 'quit':
                break
            f.write(data)
        conn.send('success')
    print "receive finish!"
sk.close()
```





























