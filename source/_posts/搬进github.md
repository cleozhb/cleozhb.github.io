---
title: 搬进github
date: 2018-04-15 20:09:25
tags: github
---

![图片](1.png)

集中式：需要一个中心服务器，如果不联网就没有办法提交，没有办法记录，很多操作都没有办法做，而分布式就不同，每个拥有版本库的人都可以在不联网的情况下快速的完成文件的提交，查看记录，删除等操作，比集中式高效很多。

<!--more-->

commit 制作一个版本
将新建的项目clone到本地，添加了一些新的文件，用git status查看状态，可以看到git提醒位于master分支，提交为空，存在尚未跟踪的文件（用git add 建立跟踪）
![图片](2.png)
然后我用git add命令建立文件跟踪，再次执行git status查看工作区状态，可以看到git提示我们可以用reset命令撤销提交，现在不需要。
![图片](3.png)
执行git commit执行提交变更 ，会跳到一个填写对于此次提交的说明。便于查看此次修改的目的，定位到做了什么修改。然后再次查看git状态，这次的提醒是没有文件要提交了，是一个干净的工作区。可以用git push来发布本地提交。
![图片](4.png)
执行git push输入github的用户名和密码，git提示我们成功push了
![图片](5.png)
用git log命令可以查看过去的提交操作，回到过去的版本时可以从中查看commit ID
![图片](6.png)
回到过去时要复制过去某个时间点的commit ID,使用
git reset --hard 1b3d0a55fbee598fb0c89b1b8d6e8127622be5e1
将hard修改到这个commit上。hard是当前环境中版本的一个指针，如果将这个指针做修改也就将整个环境的代码退回到这个时间点的一个状况。这样就退回到过去的时间点了。

git reflog会列出当前版本之前的版本号都是什么
![图片](7.png)

[https://www.linuxidc.com/Linux/2017-06/145132.htm](https://www.linuxidc.com/Linux/2017-06/145132.htm)

