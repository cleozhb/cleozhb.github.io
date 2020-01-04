---
title: win10+VS2013与Ubuntu16.04配置GDAL
date: 2018-04-10 14:05:16
categories: 经验
tags: [GDAL]
---
## GDAL在不同操作系统下的安装
> 主要内容

* GDAL简介
* Ubuntu配置GDAL
* [win10+VS2013配置GDAL](https://blog.csdn.net/u011574296/article/details/76565703)
<!--more-->

### GDAL简介
GDAL(Geospatial Data Abstraction Library)是开源栅格空间数据转换库，它用抽象的数据模型来表达所支持的各种文件格式，有一系列的命令行工具来进行数据转换和处理。OGR是GDAL项目的一个分支，提供对矢量数据的支持。

#### 支持格式
GDAL提供对多种栅格数据的支持，包括Arc/Info ASCII Grid(asc)，GeoTiff (tiff)，Erdas Imagine Images(img)，ASCII DEM(dem) 等格式。
```
GDAL使用抽象数据模型(abstract data model)来解析它所支持的数据格式
抽象数据模型包括数据集(dataset)
坐标系统
仿射地理坐标转换(Affine Geo Transform)
大地控制点(GCPs)
元数据(Metadata)
栅格波段(Raster Band)
颜色表(Color Table)
子数据集域(Subdatasets Domain)
图像结构域(Image_Structure Domain)
XML域(XML:Domains)
GDALMajorObject类：带有元数据的对象。
GDALDdataset类：通常是从一个栅格文件中提取的相关联的栅格波段集合和这些波段的元数据;
GDALDdataset也负责所有栅格波段的地理坐标转换(georeferencing transform)和坐标系定义。
GDALDriver类：文件格式驱动类，GDAL会为每一个所支持的文件格式创建一个该类的实体，来管理该文件格式。
GDALDriverManager类：文件格式驱动管理类，用来管理GDALDriver类。
```

###Ubuntu配置GDAL
在Ubuntu中配置GDAL时，原本只需要按照博客中说的步骤：

1. 下载[URL: http://download.osgeo.org/gdal/](URL: http://download.osgeo.org/gdal/)
2. 安装GDAL

```
% cd gdal
% ./configure
% make
% su
Password: ********
% make install
% exit

安装成功后，会在/usr/local/include文件夹中产生一系列.h文件(cpl_config.h、gdal.h……)，在/usr/local/lib文件夹中产生5个库文件，其中libgdal.a和libgdal.la为静态链接库，libgdal.so, libgdal.so.1, libgdal.so.1.13.2为动态链接库。
```

3. 将/usr/local/lib添加到环境变量中，命令如下：
```
cd /etc/ld.so.conf.d
touch local.conf
vi local.conf
i
/usr/local/lib
Esc
Wq
ldconfig -v
```

4.  将/usr/local/include文件夹copy到当前工程（test）目录下。test.cpp里面加载必要的头文件就可以使用GDAL和PROJ里面的函数：
```
#include "./include/gdal.h"
#include "./include/gdal_alg.h"
#include "./include/cpl_conv.h"
#include "./include/cpl_port.h"
#include "./include/cpl_multiproc.h"
#include "./include/ogr_srs_api.h"
```

5.  编译g++ test.cpp –lgdal –o TEST
6.  运行./TEST

但是由于Ubuntu中对包的依赖，之前安装anaconda2时安装了一些包，包的等级比较低，不满足GDAL的需求，但Ubuntu对已安装的包进行保护，故不能重新安装高版本，所以无奈之下卸载了anaconda2。以后要用再重装吧。。。

### win10+VS2013配置GDAL

1. 下载源码，从参考文献3的链接中可以找到各种历史版本，我这里下载的是最新的gdal2.2.4
2. GDAL_HOME = “C:\warmerda\bld”，编译后的生成的头文件、静态库、动态库将会存储到这个路径。我在D盘新建了文件夹gdal2.2.4，然后修成了D:\gdal2.2.4。
3. 并将ODBC_SUPPORTED=1注释掉(因为后面会报错。。。)
4. 还有'#WIN64=YES'，在编译64位GDAL时，要删除前面的#。
![](./win10+VS2013与Ubuntu16.04配置GDAL/GDAL安装路径.png)
![](./win10+VS2013与Ubuntu16.04配置GDAL/去掉ODBC支持.png)
5. 进入VS2013 X64 本机工具命令提示符，在菜单栏的Visual Studio 2013文件夹下
6. 使用命令行，进入到源代码目录,依次输入以下命令

```
e:
cd gdal-2.2.4
nmake -f makefile.vc 
nmake /f makefile.vc install 
nmake /f makefile.vc devinstal

第一个命令是编译GDAL 
第二个、第三个命令是将生成的头文件、静态库、动态库复制到GDAL_HOME目录。 
如果需要编译debug模式，就将第一个命令改成nmake -f makefile.vc DEBUG=1
```

7. 配置环境变量
```
计算机->属性->高级系统设置->环境变量->编辑 path 
添加：D:\gdal2.2.4\bin
```
8. 在VS2013中配置GDAL时，按照上面连接中的博客配置完成(亲测有效，为博主点赞)。
生成时出现连接错误，错误原因是我编译的是x64的GDAL，而我的VS默认建工程是x86的，所以需要更改编译平台为x64并重新设置附加依赖项、包含目录、库目录。一定要注意你编译的GDAL是多少位的，你建立的工程是多少位的。需要将GDAL的lib link到你的工程中，而且它的编译版本要跟你的开发环境的VS一致。
![](./win10+VS2013与Ubuntu16.04配置GDAL/编译平台及附加依赖项.png)
![](./win10+VS2013与Ubuntu16.04配置GDAL/包含目录及库目录.png)
9. 小例子
```
#include "stdafx.h"
#include "gdal_priv.h"
#include <iostream>
using namespace std;
int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        return 0;
    }
    GDALDataset *poDataset;
    GDALAllRegister();
    poDataset = (GDALDataset *)GDALOpen(argv[1], GA_ReadOnly);
    if (poDataset != NULL)
    {
        cout << "RasterXSize:" << poDataset->GetRasterXSize() << endl;
        cout << "RasterYSize:" << poDataset->GetRasterYSize() << endl;
        cout << "RasterCount:" << poDataset->GetRasterCount() << endl;
    }
    return 0;
}
```

点击生成，在debug文件夹下进入命令行模式，假设在e:有个名叫nd_dem.tif的图片，输入
```
.\gdal_installtest.exe e:\nd_dem.tif
```
![](./win10+VS2013与Ubuntu16.04配置GDAL/GDAL小例子.png)


## 参考文献
1. [GDAL 的安装介绍及使用](https://www.cnblogs.com/bigbigtree/archive/2011/11/19/2255495.html)
2. [win10+VS2015 编译64位的gdal，并配置环境](https://blog.csdn.net/u011574296/article/details/76565703)
3. [Downloading GDAL/OGR Source](http://trac.osgeo.org/gdal/wiki/DownloadSource)
4. [新手使用GDAL详细教程](https://blog.csdn.net/u012505618/article/details/52724060)