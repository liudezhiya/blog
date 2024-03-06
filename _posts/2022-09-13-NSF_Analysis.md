---
title: NSF_Analysis
date: 2022-09-15 20:34:00 +0800
categories: [随笔]
tags: [生活]
pin: true
author: 刘德智

toc: true
comments: true
typora-root-url: ../../tomstillcoding.github.io
math: false
mermaid: true
---

[TOC]

### 1.数据集

#### ①计算机类数据集

​		所选数据集为完成时间在2008年到2019年的自然科学基金项目及其所属论文成果，为了对一类项目进行分析和便于读取文件，通过选择项目类别为计算机类（第一位F）和论文关联合并处理产生总的数据集name2paper2project.csv后面的操作都是基于此数据集，涉及到的主要字段如下：

|         计算机类别数据集字段      | 备注 |
| ------ | -------------------- |
| 作者姓名 | 节点成分一 |
| 项目名称 |  |
| 项目类别 | 筛选计算机类 |
| 论文id | 区分论文，统计成员 |
| 论文发表机构 | 分析发表机构 |
| 学校 | 节点成分二 |
| 论文关键字 |  |

具体如下表

![](/../blog/assets/blog_res/2021-09-13-NSF_Analysis/n2p2p.png)

#### ②节点映射字典

​		为了把作者和对应学校映射为数字，将一行数据姓名列和学校进行合并然后去重产生每个节点id，去重是保证每个字典的key值唯一，然后为每个节点一次分配一个值，这样就产生{作者 学校：index}格式字典name2index.txt，对齐反向遍历产生index2name.txt

![](/../blog/assets/blog_res/2021-09-13-NSF_Analysis/name2index.png)

![](/../blog/assets/blog_res/2021-09-13-NSF_Analysis/indexname.png)

这两个字典文件的作用主要有，name2index.txt将作者和对应学校映射为数字从而减少存储量，便于计算，index2name.txt是为后面节点检测服务，通过节点的索引找到对应作者和学校。

### 2.建立“论文和节点映射”字典

方法：为了对更好区分每一年的论文，将数据集处理成多时间片，同一年的论文年产生一个文件，为统计每一年节点个数做准备。产生规则：将每一篇论文id作为字典的key值，每一行数据的name+school组成节点,字典的values由这一篇论文下的所有节点组成的节点列表

输入：name2index.txt，name2paper2project.csv

输出：Paper2node+time+.txt

![](/../blog/assets/blog_res/2021-09-13-NSF_Analysis/paper2node.png)

**代码**

```python
from numpy import *
import csv
# ========================第一步================================
##处理成动态，多时间片
# # 建立“论文：id1,id2,id3”的字典
with open('name2index.txt', 'r', encoding='utf-8') as f1:
    data = f1.read()
    nameDict = eval(data)#string类型的data转换为dict

# csv_reader = csv.reader(open("name2paper2project.csv",encoding='utf-8'))
for time in range(2008,2020):
    print(time)
    projectDict = {}
    csv_reader = csv.reader(open("name2paper2project.csv",encoding='utf-8'))
    with open('Paper2node'+str(time)+'.txt', 'w', encoding='utf-8') as f1:#生成文件
        for row in csv_reader:
            if str(time-1)>row[18][:4] or row[18][:4]>str(time+1):#其他情况
                continue
            id = nameDict[row[1] + " " + row[13]]#id = name+schoole对应的值id
            if row[11] not in projectDict:
                projectDict[row[11]] = [id]
            else:
                projectDict[row[11]].append(id)
        print(projectDict)
        f1.write(str(projectDict))
```

### 3.构建论文作者合作关系网络

**方法**：对每一篇论文的论文节点进行遍历，以节点A作为字典的key值，与他所有相连的边的列表集合作为values值，如果遇到节点B与A相同的，则将与B相连的边全部加入到A的值列表中，然后对A的值列表进行去重，得到论文作者合作关系有向边网络

**输入**：Paper2node'+time+.txt

**输出**：PaperEdge+time)+.txt

![](/../blog/assets/blog_res/2021-09-13-NSF_Analysis/paperedge.png)

**代码**

```python
allauthornum=[]
alledgenum=[]
for time in range(2008,2020):
    print(time)
    paperEdge={}
    author_cu=0
    edge_cu=0
    with open('Paper2node'+str(time)+'.txt', 'r', encoding='utf-8') as f1:
        data = f1.read()
        authorList = eval(data)
    for value in authorList.values():
        print(value)
        for i in value:#[10, 11, 12, 8, 9, 3]
            # print(i)
            tmp1 = list(value)
            tmp1.remove(i)
            if i not in paperEdge.keys():
                author_cu+=1
                paperEdge[i] = tmp1#10：[11, 12, 8, 9, 3]
            else:
                tmp2 = paperEdge[i]
                print(paperEdge[i])
                paperEdge[i] = list(set(tmp2+tmp1))
    print(paperEdge)
    for i in paperEdge:
        author_cu+=1
        edge_cu+=len(paperEdge[i])
    allauthornum.append(author_cu)
    alledgenum.append(edge_cu)

    with open('PaperEdge'+str(time)+'.txt', 'w', encoding='utf-8') as f2:
        f2.write(str(paperEdge))
print(alledgenum)
print(allauthornum)
```

### 4:计算节点的异常度

#### ①计算节点论文的count值,格式“id : [count1,count2,...,count7]”

**方法**：为了使后期比较好计算，统计每一年每个节点发表论文的个数，由长度为12的列表来表示2008年到2019时间片，第一位代表2008年第二位代表2009年...依次到2019年，每一位代表不同年份每个节点发表论文的个数

**输入**：name2paper2project.csv，name2index.txt

**输出**：PaperCountDict.txt

![](/../blog/assets/blog_res/2021-09-13-NSF_Analysis/papercountdict.png)

**代码**

```python
def yearcount(year):
    if year == "2008":
        return 0
    if year == "2009":
        return 1
    if year == "2010":
        return 2
    if year == "2011":
        return 3
    if year == "2012":
        return 4
    if year == "2013":
        return 5
    if year == "2014":
        return 6
    if year == "2015":
        return 7
    if year == "2016":
        return 8
    if year == "2017":
        return 9
    if year == "2018":
        return 10
    if year == "2019":
        return 11


csv_reader = csv.reader(open("name2paper2project.csv", encoding='utf-8'))
countDict = {}
with open('name2index.txt', 'r', encoding='utf-8') as f1:
    data = f1.read()
    nameDict = eval(data)
    f1.close()
# print(nameDict)
with open('PaperCountDict.txt', 'w', encoding='utf-8') as f2:
    for row in csv_reader:
        id = nameDict[row[1] + " " + row[13]]
        year = row[18][:4]
        numlocal = yearcount(year)
        if id not in countDict:
            countDict[id] = [0,0,0,0,0,0,0,0,0,0,0,0]
        countDict[id][numlocal] += 1
    print(countDict)
    f2.write(str(countDict))
```

#### ②以两年为限划定6个时间片,或者08-18年为界

**方法**：将由长度为12的列表来表示2008年到2019时间片，将一个节点连续两年发表的论文求和组成长度为12的列表来表示2008年到2019时间片，这样可以减少数据的稀疏性。可以根据实验要求选择对应年份本次实验选择2008年到2018年节点来计算节点异常值，去掉2019年即去掉2008年到2019时间片列表最后一位即可

**输入**：PaperCountDict.txt

**输出**：PaperCountDict_2year.txt，PaperCountDict_08_18.txt

![](/../blog/assets/blog_res/2021-09-13-NSF_Analysis/papaer2-08-18.png)

**代码**

```python
with open('PaperCountDict.txt', 'r', encoding='utf-8') as f3:
    data = f3.read()
    paperCoutAll= eval(data)
    f3.close()
# print(nameDict)
dict2 = {}
dict3 = {}
with open('PaperCountDict_2year.txt', 'w', encoding='utf-8') as f4:
    for key,values in paperCoutAll.items():
        dict2[key] = [0, 0, 0, 0, 0, 0]
        for i in range(12):
            dict2[key][int(i/2)] += values[i]
    f4.write(str(dict2))
with open('PaperCountDict_08_18.txt', 'w', encoding='utf-8') as f5:
    for key,values in paperCoutAll.items():
        dict3[key] = values[:11]
    f5.write(str(dict3))
```

#### ③为每个网络节点计算p-值

**公式**：**网络节点p值 = (p_count-1)/len(value)**    

p_count：所求年份及之前每一年节点论文总篇数 小于等于 所在时间片内平均值的个数

len(value)：统计的时间区间长度

![](/../blog/assets/blog_res/2021-09-13-NSF_Analysis/pvalues.png)

**算法流程**：

​	①求在2008年时间片上，2008年39号节点发表的论文是不是小于平均值 是的话p_count就加一

​	②求在2008-2009年时间片上，2009年39号节点发表的论文是不是小于平均值 是的话 p_count就加一

​	③求在2008-2009年时间片上，2010年39号节点发表的论文是不是小于平均值 是的话p_count就加一

​	④求在2008-2010年时间片上，2011年39号节点发表的论文是不是小于平均值 是的话 p_count就加一

​	⑤求在2008-2012年时间片上，2012年39号节点发表的论文是不是小于平均值 是的话 p_count就加一

由于第一步恒成立没有比较意义，所以p_count需要减一

如果一个人每年发表的论文篇数呈现单调递增，p_count的数值就会越小，p值 就会越小

**计算经验p值，给p值设定一个范围就可以求出满足p值的节点，这些有潜力的节点被认为是异常点就是优秀人才**

**输入****：PaperCountDict_08_18.txt

**输出**：paper_pvalue_+year+.txt

![](/../blog/assets/blog_res/2021-09-13-NSF_Analysis/p_value.png)

代码

```python
with open('PaperCountDict_08_18.txt', 'r', encoding='utf-8') as f8:
    data = f8.read()
    paperCount = eval(data)
for year in range(2012,2019):
    paperPValueDict = {}
    learn_len=year-2008+1
    for key,value in paperCount.items():
        p_count = 0
        for i in range(learn_len):
            if value[i] <= mean(value[:i+1]):
                p_count += 1
        pvalue = (p_count-1)/len(value)
        paperPValueDict[key] = pvalue
    num = 0
    # 统计了异常的人数
    for key,value in paperPValueDict.items():
        if value <= 0.15:
            num += 1
    print(year)
    print(num)
    with open('paper_pvalue_'+str(year)+'.txt', 'w', encoding='utf-8') as f9:
        f9.write(str(paperPValueDict))
```

### 4.生成异常机构和异常人员名单

通过计算p值将每一年满足p值<= 0.15的节点筛选出来

**输入**：paper_pvalue_+year+.txt，index2name.txt

**输出**：peojectYiChangMinDan+year+.txt，paper_YiChang_school+year+'.txt，allschool.txt

![](/../blog/assets/blog_res/2021-09-13-NSF_Analysis/mindan.png)

![](/../blog/assets/blog_res/2021-09-13-NSF_Analysis/school.png)

![](/../blog/assets/blog_res/2021-09-13-NSF_Analysis/allschool.png)

代码

```python
all_school=[]
for year in range(2012,2019):#2012-2018
    with open('paper_pvalue_'+str(year)+'.txt', 'r') as pvalue_file:
        p_value = eval(pvalue_file.read())
        pvalue_file.close()
    with open('index2name.txt', 'r',encoding='utf-8') as id_file:
        id2name = eval(id_file.read())
        id_file.close()
    num=0
    nameDict={}
    schooldict = {}
    count = 0
    # 统计网络中的异常点{人名+机构名：[id,p值]}
    for key,value in p_value.items():
        if value <= 0.15:
            nameschool = id2name[key]
            if nameschool not in nameDict:
                num += 1
                currentnume=nameschool
                nameDict[nameschool] = [key, value]
                school = nameschool.split(" ")[-1]
                all_school.append(school)
                if school not in schooldict:
                # if school not in schooldict:
                    schooldict[school] = 1
                else:
                    schooldict[school] += 1
    print("一共 "+str(num)+" 的优秀人才")
    print(schooldict)
    with open('peojectYiChangMinDan'+str(year)+'.txt', 'w', encoding='utf-8') as f5:
        f5.write(str(nameDict))
    with open('paper_YiChang_school'+str(year)+'.txt', 'w', encoding='utf-8') as f5:
        f5.write(str(schooldict))
all_school=list(set(all_school))
with open('allschool.txt', 'w', encoding='utf-8') as f5:
    f5.write(str(all_school))
```

### 5.结果绘制

#### ①绘制p值分布excel表格生成

异常节点在在不同年份分布情况

**输入**：allschool.txt，paper_YiChang_school+year)+.txt

**输出**：数据表.xls

![](/../blog/assets/blog_res/2021-09-13-NSF_Analysis/shujubiao.png)

代码

```python
import xlwt
import re
def writeinexcel1():
    f = open('paper_YiChang_school2018.txt', 'r', encoding='utf-8')  # 打开数据文本文档，注意编码格式的影响
    wb = xlwt.Workbook(encoding='utf-8')  # 新建一个excel文件
    ws1 = wb.add_sheet('first')  # 添加一个新表，名字为first
    ws1.write(0, 0, 'school')
    ws1.write(0, 1, 2015)
    ws1.write(0, 2, 2016)
    ws1.write(0, 3, 2017)
    ws1.write(0, 4, 2018)
    row = 1  # 写入的起始行
    col = 0  # 写入的起始列
    # 通过row和col的变化实现指向单元格位置的变化
    k = 1
    allschool={}
    with open('allschool.txt', 'r', encoding='UTF-8-sig') as pvalue_file:
        lines = pvalue_file.read()
        a = lines.split(',')
        for i in range(len(a)):
            current_a=re.sub("'", "", a[i])
            current_a = re.sub(" ", "", current_a)
            print(current_a)
            yearn = dict()
            for year in range(2015,2019):
                with open('paper_YiChang_school'+str(year)+'.txt', 'r', encoding='UTF-8-sig') as pvalue_file2:
                    p_value = eval(pvalue_file2.read())
                    pvalue_file2.close()
                    print(p_value)
                    if current_a in p_value:
                        # print("ll")
                        yearn[year]= p_value[current_a]
                    else:
                        yearn[year] =int(0)
            allschool[current_a] = yearn
    for key,values in allschool.items():
        ws1.write(row, col, key)
        col+=1
        for key2,x in values.items():
            ws1.write(row, col, x)  # 向Excel文件中写入每一项
            col += 1
        row += 1
        col = 0

    wb.save("数据表.xls")
```

#### ②绘制期刊值分布excel表格生成

所以的论文发布期刊分布组成

**输入**：name2paper2project.csv,

**输出**：journal.xls

![](/../blog/assets/blog_res/2021-09-13-NSF_Analysis/joural.png)

代码

```python
####绘制期刊值分布excel表格生成
import xlwt
import re
def writeinexcel2():
    csv_reader = csv.reader(open("name2paper2project.csv",encoding='utf-8'))
    dict_c={}
    for row in csv_reader:
        if row[16] in dict_c.keys():
            dict_c[row[16]]+=1
        else:
            dict_c[row[16]] = 1

    wb = xlwt.Workbook(encoding='utf-8')  # 新建一个excel文件
    ws1 = wb.add_sheet('first')  # 添加一个新表，名字为first
    ws1.write(0, 0, 'journal')
    ws1.write(0, 1, 'num')
    row = 1  # 写入的起始行
    col = 0  # 写入的起始列
    # 通过row和col的变化实现指向单元格位置的变化
    k = 1
    for key,values in dict_c.items():
        ws1.write(row, col, key)
        col+=1
        ws1.write(row, col, values)  # 向Excel文件中写入每一项
        row += 1
        col = 0

    wb.save("journal.xls")

```

