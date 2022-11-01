---
title: DataFrame操作
date: 2022-07-20 11:34:00 +0800
categories: [随笔]
tags: [分类]
pin: true
author: 刘德智

toc: true
comments: true
typora-root-url: ../../liudezhiya.github.io
math: false
mermaid: true

image:
  src: /assets/blog_res/2021-03-30-hello-world.assets/huoshan.jpg!
  alt: 签约成功

---

###  

[TOC]



### pandas常用的数据查看方法有 

| 方法       | 操作                  | 结果              |
| ---------- | --------------------- | ----------------- |
| head(n)    | 查看数据集对象的前n行 | Series或DataFrame |
| tail(n)    | 查看数据集的最后n行   | Series或DataFrame |
| sample(n)  | 随机查看n个样本       | Series或DataFrame |
| describe() | 数据集的统计摘要      | Series            |

```python
print(data.iloc[:,0:54])数据
print(data.iloc[:,54:55])标签
x=np.array(data.iloc[:,0:54])
y=np.array(data.iloc[:,54:55])
```

| 操作           | 语法          | 返回结果  |
| -------------- | ------------- | --------- |
| 选择列         | df[col]       | Series    |
| 按索引选择行   | df.loc[label] | Series    |
| 按位置选择行   | df.iloc[loc]  | Series    |
| 使用切片选择行 | df[2:5]       | DataFrame |
| 用表达式筛选行 | df[bool]      | DataFrame |

```
df[0:3] #通过切片检索行数据
df[0:3][['staff_id','staff_name','staff_gender']] #通过列标签索引列表检索列数据

df.loc[1] #标量标签，返回该行标签的Series数据
df.loc[[1,3]] #标签列表，返回标签列表的行DataFrame数据
df.loc[0:3] #切片对象，返回切片的行DataFrame数据
df.loc[0:3,'staff_id':'staff_salary']  #根据行切片，列切片检索数据
df.loc[[0,1,2,3],['staff_id','staff_name','staff_age']] #根据行标签列表，列标签列表检索数据

df.iloc[1]  #整数标量选择，数据从0开始，为1的就是第二行的数据，返回的是Series

df.iloc[[1,3]] #整数列表选择，选择位置为1和3的数据，返回的是DataFrame
df.iloc[1:3] #切片选择，选择位置1至2的数据，不包含边界结束值，也就是不包含3的位置

df.iloc[1:3,1:4] #切片选择位置为1至3的行和1至4的列不含位置为3的行和位置为4的列

df.at['A','staff_name']  #检索第“A”行的列标签为"staff_name"的数据
df.iat[0,1]  #检索第1行第2列的数据


df[(df.staff_salary>10000)&(df.staff_age<40)]  #检索staff_age小于40且staff_salary>10000的数据
df.query('staff_salary>10000 & staff_age<40') #通过函数检索staff_age小于40且staff_salary>10000的数据


```

###  访问 DataFrame 中的单个元素时 

 ```
并且列标签在前，格式为 `dataframe[column][row]
 ```



### 读取文件夹下所有文件

```
alldata_filenames=glob.glob(alldata_directory+"\*.csv")
for alldata_filename in alldata_filenames:
#for alldata_filename in os.listdir(alldata_directory):#直接文件名没有带路径
    print(alldata_filename)#带路径
```



### 以多个划分符分割

```
 import re
 awardnamelinelist=re.split(r'[;, ]]',awardname)
```

### 将中文姓名转为英文姓名

```python
def nametopingying(str):
    result = pypinyin.pinyin(str, style=pypinyin.NORMAL)
    result = [i[0] for i in result]
    result = ''.join(result[1:]).capitalize() + ' ' + result[0].capitalize()
    return result
```

### python对csv去除重复行

```
frame=pd.read_csv('E:/bdbk.csv',engine='python')

data = frame.drop_duplicates(subset=['名称'], keep='first', inplace=False)

data.to_csv('E:/baike.csv', encoding='utf8')
```





通常会分为两种情况，一种是去除完全重复的行数据，另一种是去除某几列重复的行数据，就这两种情况可用下面的代码进行处理。

```


1. 去除完全重复的行数据

data.drop_duplicates(inplace=True)

2. 去除某几列重复的行数据

data.drop_duplicates(subset=['A','B'],keep='first',inplace=True)

subset： 列名，可选，默认为None

keep： {‘first’, ‘last’, False}, 默认值 ‘first’

first： 保留第一次出现的重复行，删除后面的重复行。

last： 删除重复项，除了最后一次出现。

False： 删除所有重复项。

inplace：布尔值，默认为False，是否直接在原数据上删除重复项或删除重复项后返回副本。

( inplace=True表示直接在原来的DataFrame上删除重复项，而默认值False表示生成一个副本。)

DataFrame.drop_duplicates(subset = None, keep = 'first')

DataFrame.drop_duplicates()中的参数完全实现。

其中subset这个参数默认‘None’是指选择所有列，即所有列的值都相同我才认为这两行是重复的，

也可以自定义为其中一部分列变量名，比如subset=['name','sex','age']。

keep参数中'first'和‘last’会根据index的前后产生不同的效果。参数False会去除所有重复行。
```



举个栗子：

name       sex            age

0    coco          female        7

1      lily           female        7

2     joe           male          15

3     coco        female        7

DataFrame.drop_duplicates(subset = None, keep = 'first')，产生的结果如下：

name       sex         age

0        coco    female         7

1      lily         female        7

2     joe         male          15

若使用代码DataFrame.drop_duplicates(subset = None, keep = 'last')，结果如下：

name       sex         age

1      lily         female        7

2     joe         male          15

3      coco    female        7

发现不考虑index以及行的顺序，效果与参数first相同。

若使用代码DataFrame.drop_duplicates(subset = None, keep = False), 则把相同的行全部删除，结果如下：

name       sex         age

1      lily         female        7

2     joe         male          15

所有重复的行都被删除，没有保留。

若使用代码DataFrame.drop_duplicates(subset = ['sex' , 'age'] , keep = False), 结果如下：

name       sex         age

2     joe         male          15
————————————————

### 统计重复值，并去除相同的元素

统计

```python
value_counts(normalize=False, sort=True, ascending=False, bins=None, dropna=True)
参数:
1.normalize : boolean, default False　默认false，如为true，则以百分比的形式显示
2.sort : boolean, default True　默认为true,会对结果进行排序
3.ascending : boolean, default False　默认降序排序
4.bins : integer, 格式(bins=1),意义不是执行计算，而是把它们分成半开放的数据集合，只适用于数字数据
5.dropna : boolean, default True　默认删除na值
```



```python
data = df[['source', 'target']].value_counts()

df = pd.read_csv('edge.csv')
df = df[['source', 'target']].value_counts().reset_index()
df.columns=['source','target','weight']#为统计值加列名
```

去重

```python
#1. 去除完全重复的行数据
data.drop_duplicates(subset=['A','B'],keep='first',inplace=True)
#2. 去除某几列重复的行数据
data.drop_duplicates(subset=['A','B'],keep='first',inplace=True)

subset： 列名，可选，默认为None

keep： {‘first’, ‘last’, False}, 默认值 ‘first’

    first： 保留第一次出现的重复行，删除后面的重复行。
    last： 删除重复项，除了最后一次出现。
    False： 删除所有重复项。

inplace：布尔值，默认为False，是否直接在原数据上删除重复项或删除重复项后返回副本。（inplace=True表示直接在原来的DataFrame上删除重复项，而默认值False表示生成一个副本。）
```





更换列名

```
# 更换列名，同时显示出来
df.rename(columns={'a':'A'}, inplace=Ture)
# 更换列名，不显示出来 inplace = False(默认)

读入不要索引
df = pd.read_csv(alldata_filename,index_col=0)
写入不要列名
df_author.to_csv('author.csv', index=False)
```

导出加时间

```
import time
timestr = time.strftime("%Y%m%d-%H%M%S")  # 20220715-085113
```

内存溢出

```
方法一：扩充虚拟内存
1、打开 控制面板；
2、找到 系统 这一项；
3、找到 高级系统设置 这一项；
4、点击 性能 模块的 设置 按钮；
5、选择 高级面板，在 虚拟内存 模块点击更改；
6、记得 不要 选中“自动管理所有驱动器的分页文件大小”，然后选择一个驱动器，也就是一个盘，选中自定义大小，手动输入初始大小和最大值，当然，最好不要太大，更改之后能在查看盘的使用情况，不要丢掉太多空间。
7、都设置好之后，记得点击 “设置”， 然后再确定，否则无效，最后 重启电脑 就可以了。

方法二：修改pycharm的运行内存

Help->Find Action->(type “VM Options”)->(Click)“Edit Custom VM Options” 打开pycharm64.exe.vmoptions进行编辑
修改-Xmx750m 为 -Xmx4096m 分配4G内存，视情况而定。保存并重启pycharm

自己的是-Xmx1773m改为-Xmx4096m

```





csv文件连接

```python
outfile = pd.merge(df1, df2,  left_on='df1_id', right_on='df2_id')
#文件合并 left_on左侧DataFrame中的列或索引级别用作键。right_on 右侧

参数how有四个选项，分别是：inner、outer、left、right
pd.merge(dataframe_1,dataframe_2,how="inner")

内连接
inner是merge函数的默认参数，意思是将dataframe_1和dataframe_2两表中主键一致的行保留下来，然后合并列。
外连接
outer是相对于inner来说的，outer不会仅仅保留主键一致的行，还会将不一致的部分填充Nan然后保留下来。
left、right 左连接，右连接
```





###  pycharm提示内存不足的解决方法

```
点击 Pycharm 菜单栏的 Help->Edit Custom VM Options进入 pycharm64.exe.vmoptions 文件
或者直接用记事本打开 C:\Users\Administrator\.PyCharm2017.2\config\pycharm64.exe.vmoptions

-Xmx 是运行时的可用的内存大小，默认是 -Xmx750m，可根据实际需要调大一些，
建议改成 -Xmx1024m
```

## 解决报错DtypeWarning: Columns (2) have mixed types

 添加low_memory参数 

    df_question = pd.read_csv("D:/data/final/question20181201.csv",usecols=[2,4,15],low_memory=False)

注：其中的usecols参数是读取csv第几列的意思，只需要读取有用字段就OK啦。


二、问题解决

按照警告的提示，解决方法有两种：关闭 low_memory 模式或者指定列的数据类型。

1.关闭 low_memory

```
data = pd.read_csv(f, low_memory=False)
```

2.指定类型（推荐）

例如我这里把这些列都让 Pandas 看作是 str：

```
data = pd.read_csv(f, dtype={"site": str, "aqi": str})
```

三、low_memory 是什么

Pandas 在读取 csv 文件时时按块读取的，并不会一次性读取，并且对于数据的类型“都靠猜”，所以就可能出现了 Pandas 在不同块对同一列的数据“猜”出了不同的数据类型，也就造成了上述的警告。

而如果关闭 low_memory 功能时，Pandas 就会一次性读取 csv 中德所有数据，自然对列的数据类型也只会猜测一次，就不会造成这种警告了。但是关闭 low_memory 时，一旦 csv 文件过大，就会内存溢出，所以建议采取指定类型的方式。





 **DataFrame转list** 

```python
lable=data[108].tolist()
tr = data.iloc[train_index, 0:108]
te = data.iloc[test_index, 0:108]
lable_te = data.iloc[test_index,108].tolist()
```

```python
data_w_2 = data_w.iloc[[3,5],[0,1,2,5]]#取下标为3、5的行，0、1、2、5的列

print(type(data_w_2.iloc[:1,1:]))#类型为DataFrame
dm1=np.array(data_w_2.iloc[:1,1:])#用np.array将DataFrame转成ndarray
print(type(dm1))
print(dm1)
dm1=np.array(data_w_2.iloc[:1,1:]).tolist()#再用.tolist()将ndarray转成list
print(type(dm1))
print(dm1)
dm1=np.array(data_w_2.iloc[:1,1:]).tolist()[0]#由于上面的列表结果仍多包含一个列表，故采用此种方式取到单个列表
print(type(dm1))#至于为什么包裹了多个列表没有去细想
print(dm1)


#返回结果
<class 'pandas.core.frame.DataFrame'>
<class 'numpy.ndarray'>
[[ 0.45   0.199 14.3  ]]
<class 'list'>
[[0.45, 0.199, 14.3]]
<class 'list'>
[0.45, 0.199, 14.3]
```

# 某一列中每一行拆分成多行

- 将需要拆分的数据使用split拆分工具拆分，并使用expand功能拆分成多列
- 将拆分后的多列数据进行列转行操作(stack)，合并成一列
- 将生成的复合索引重新进行reset保留原始的索引,并命名
- 将上面处理后的DataFrame和原始DataFrame进行join操作，默认使用的是索引进行连接

```python
year='2018'
alldata_award='G://BaiduNetdiskDownload//135数据//18-22文件合并//'+year+'//alldata_award_'+year+'.csv'
info = pd.read_csv(alldata_award,encoding='utf-8')
info_name=data['authors'].str.split(';',expand=True)
info_name=info_name.stack()
info_name=info_name.reset_index(level=1,drop=True)
info_new=data.drop(['authors'],axis=1).join(info_name)
整合
info_new=info.drop(['authors'], axis=1).join(info['authors'].str.split(';', expand=True).stack().reset_index(level=1, drop=True).rename('author'))
info_new.to_csv('G://data//findnodeoutput//alldata_award_'+year+'.csv')
```

 如果原数据中已经是list了，可以将`info[‘city’].str.split(’ ', expand=True)`这部分替换成`info[‘city’].apply(lambda x: pd.Series(x))`

### 文件读写

加列名

```
 f2 = open('G://data//findnodeoutput//NamePaperProject_safe.csv', 'w', encoding='utf-8', newline='')
    writer = csv.writer(f2)
    # 指定文件列名
    writer.writerow(
        ['p_index', 'alldata_uuid', 'proJName', 'projId', 'typeCode', 'projType', 'admin', 'supportNum', 'timeScope',
         'approveYear', 'finishYear', 'dependUnit', 'keyWords', 'abstractCh', 'abstractEn', 'conclusionAbstract',
         'alldatafilename', 'alldatafileid', 'index', 'award_uuid', 'award_name', 'award_type', 'authors', 'unknown',
         'awardfilename'])
```

不要index索引,title列名

```
index=None,header=None
写入文件间隔
newline=''
```

### 文本筛选行contains和isin

筛选proJName列含有安全的

```
1
year=str(2018)
alldata_award = 'G://BaiduNetdiskDownload//135数据//18-22文件合并//' + year + '//alldata_award_' + year + '.csv'
df = pd.read_csv(alldata_award, encoding='utf-8')
data=df[df['proJName'].str.contains('安全')]
print(data['proJName'])#7621                    组合构型及其在信息安全中的应用

2
import pandas as pd
df = {'地址':['北京','上海','长沙','北京省会','广州市区'],'table':['user','student','course','sc','book']}
df = pd.DataFrame(df)
print(df)
print('================')
citys = ['北京', '天津', '上海']
address = '|'.join(citys)
df_new = df[df['地址'].str.contains(address)]
print(df_new)#输出含有address城市的行
```



### 重置列索引

```python
temp.reset_index(inplace=True,drop=True)
```

安装年份求平均值

```python
temp.groupby('年份').mean()
```



### 归一化

```
# 将整型变为float
dataset = dataset.astype('float32')
#对数据集合进行标准化
scaler = MinMaxScaler(feature_range=(0, 1))
```

# python中逐行遍历Dataframe

​         **1、iterrows()方法**      

  逐行迭代，将DataFrame的每一行迭代成(index, Series)对，可以通过row[name]访问。 

```php
for index, row in df.iterrows():
    print row["c1"], row["c2"]
```

​         **2、itertuples()方法**      

  逐行迭代，将DataFrame的每一行作为一个元组进行迭代，可以通过row[name]访问元素，比iterrows()效率更高。 

```php
for row in df.itertuples(index=True, name='Pandas'):
    print getattr(row, "c1"), getattr(row, "c2")
```

​         **3、iteritems()方法**       


  按列遍历，将DataFrame的每一列迭代成（列名，系列）对，可以通过row[index]访问。 

```php
for date, row in df.iteritems():
    print(date)
for date, row in df.iteritems():
    print(row)
for date, row in df.iteritems():
    print(row[0], row[1], row[2])
```

### 将字典转为dataframe

```python
df = pd.DataFrame(pd.Series(node_degree), columns=['degree'])
df = df.reset_index().rename(columns={'index':'name'})
```

### pandas dataframe 两列转字典

```
d = dict(zip(df['A'],df['B']))
import pandas as pd
import numpy as np

test_dict = {'id':[1,2,3,4,5,6],'name':['Alice','Bob','Cindy','Eric','Helen','Grace '],'gender':[0,1,0,1,0,0],
             'math':[90,89,99,78,97,93]}
df = pd.DataFrame.from_dict(test_dict)

print(df)
'''
   id    name  gender  math
0   1   Alice       0    90
1   2     Bob       1    89
2   3   Cindy       0    99
3   4    Eric       1    78
4   5   Helen       0    97
5   6  Grace        0    93
'''

dict(zip(df['id'],df['math']))
# {1: 90, 2: 89, 3: 99, 4: 78, 5: 97, 6: 93}

```

 方法二：将A设为索引后，转字典 

```
d = df.set_index('A')['B'].to_dict()
d = df.set_index('name')['degree'].to_dict()
#{'张俊伟': [92, 18, 16, 16], '马建峰': [68, 34, 56, 56]
```

 实际问题中，常需要将原始表df，按某种方式聚合得到df2，需要得到df2的两列字典。可以直接联合使用 聚合groupby，agg和to_dict函数。 

```
d = df.groupby('A')['B'].mean().to_dict()

d = df.groupby('gender')['math'].mean().to_dict()
# {0: 94.75, 1: 83.5}
```

eval

```python
	a	       b	        c	        d
0	0.352100	0.660768	0.259112	0.190435
1	0.438345	0.147769	0.702476	0.503706
2	0.214064	0.440153	0.700988	0.029637
3	0.646761	0.539095	0.980113	0.921489
4	0.747330	0.260352	0.191178	0.002823
5	0.969599	0.163768	0.018234	0.458367


df['e'] = df['a']*df['b']+df['c']*df['d'];
df.eval('e=a*c+b*d', inplace=True)
print(df)

[Out]：
	a	        b        	c	        d      	    e
0	0.352100	0.660768	0.259112	0.190435	0.217067
1	0.438345	0.147769	0.702476	0.503706	0.382359
2	0.214064	0.440153	0.700988	0.029637	0.163101
3	0.646761	0.539095	0.980113	0.921489	1.130668
4	0.747330	0.260352	0.191178	0.002823	0.143608
5	0.969599	0.163768	0.018234	0.458367	0.092746
```

### python读取.mat文件处理

直接读取  

```python
import scipy.io as scio
#注意带路劲
data_path='D:\workspace\MachineLearning_HW_CQUT\HW3 SVM\data1.mat'
data= scio.loadmat(data_path)
print(data.keys())
data_train_data=data.get('X')#取出字典里的data  <class 'numpy.ndarray'>
data_train_label=data.get('y')#取出字典里的label  <class 'numpy.ndarray'>
print(type(data_train_label))


[0.9044   3.0198  ]
 [0.76615  2.5899  ]
 [0.086405 4.1045  ]]
[[1]
 [1]
 [1]
 [1]
```

将.mat转为dataframe

```python
from scipy.io import loadmat
import pandas as pd
data_path='D:\workspace\MachineLearning_HW_CQUT\HW3 SVM\data1.mat'
data = loadmat(data_path)
dfdata = pd.DataFrame(data=data['X'][:],columns=['d1', 'd2']).astype(str)
dfdata['d1'] = dfdata['d1'].map(lambda x: x.replace('[', '').replace(']', ''))
dfdata["d2"] = dfdata['d2'].map(lambda x: x.replace('[', '').replace(']', ''))
print(dfdata.values)

[['1.9643' '4.5957']
 ['2.2753' '3.8589']
 ['2.9781' '4.5651']
```



转为数字

```python
train_data["放牧小区（plot）"] = train_data["放牧小（plot）"].astype('category').cat.codes

train_data["放牧强度（intensity）"] = train_data["（intensity）"].astype('category').cat.codes
```

## python生成随机数

## 一、python自带的random模块

python标准库中的`random`函数，可以生成随机浮点数、整数、字符串，甚至帮助你随机选择列表序列中的一个元素，打乱一组数据。

```python
random.randint(n,m) #生成一个n到m之间的随机数
random.random()  #生成一个0到1之间的浮点数
random.sample(range(0, 20), 20)#生成不重复的 生成0~20之间的20个随机整数：
random.uniform(n,m) #生成一个n到m之间的浮点数
random.choice([])  #从列表之间随机选取一个数
random.gauss(5, 1) #生成一个正态分布的随机数，均值为 5， 标准差为 1
random.expovariate(0.2) #生成一个指数分布的随机数，均值为 5
```

## 二、numpy模块生成随机数

```python
    np.random.rand（）#产生N维的均匀分布的随机数
    np.random.randn（）#产生n维的正态分布的随机数
    np.random.randint(n,m,k)#产生n--m之间的k个整数
    np.random.random()#产生n个0--1之间的随机数
    np.random.uniform(1, 10, [2,2])#生成 [1, 10] 内的均匀分布随机数， 2 行 2 列
    np.random.normal(5, 1, [2,2]) #生成一个正态分布的随机数，均值为 5， 标准差为 1， 2 行 2 列
    np.random.poisson(5, [2,2]) #生成一个泊松分布的随机数，均值为 5， 2 行 2 列
    np.random.exponential(5, [2,2])生成一个指数分布的随机数，均值为 5， 2 行 2 列
```

