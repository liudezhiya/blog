# 数据处理DataCleaer.py



<img src="C:\Users\刘德智\Desktop\edge.png" style="zoom:50%;" />



```python

import re
import pandas as pd

#提取姓名列表传入姓名列表返回合并的姓名列表
def nameToRowmerge(addnamelist):
    for i in range(len(addnamelist)):
        name_list = str(addnamelist[i]).split(';')
        while '' in name_list:
            name_list.remove('')
        addnamelist[i] = name_list
    return addnamelist

#提取姓名列表增加后
def nameToRowadd(file,row):
    data = pd.read_csv(file)
    x = data[row].tolist()
    for i in range(len(x)):
        name_list = str(x[i]).split(';')
        x[i]=name_list[:len(name_list)-1]
    return x

#提取项目编号
def fund_number(file,row_name):
    data = pd.read_csv(file)
    fund = data[row_name].to_dict()
    for key, value in fund.items():
        matchObj = re.search('[0-9]{4}YFC[0-9]{7}', str(value))
        if (matchObj):
            fund[key] = matchObj.group()
        else:
            fund[key] = 0
    return fund


#姓名列表转为边
def nemeToEdge(list):
    edge = []
    for i in range(0, len(list) - 1):
        for j in range(i + 1, len(list)):
            if (list[i] != list[j] and list[i],list[j] !=''):
                edge_one = [];
                edge_one.append(list[i])
                edge_one.append(list[j])
            edge.append(edge_one)
    return edge

#姓名列表转为边
def  nemeToEdgeadd(list,number):
    edge = []
    for i in range(0, len(list)):
        for j in range(i + 1, len(list)):
            if (list[i] != list[j] and list[i],list[j] !=''):
                edge_one = [];
                edge_one.append(list[i])
                edge_one.append(list[j])
                edge_one.append(number)
                # print(edge_one)
            edge.append(edge_one)
    return edge

'''将同一个项目编号的负责人加到paper人员里面'''
#file 文件
#row 姓名列
#item 要匹配项目编号字典
#姓名列列表 3,4同一文件
def addname(file,row,item,row_name):
    data = pd.read_csv(file)
    x = data[row].to_dict()
    for i, j in x.items():
        for m, n in item.items():
            if (str(paper_umber[i])) == n and (n!=0):
                str_name = str(j) + ';' + str(row_name[m]) + ';'
                x[i] = str_name
    return x

'''将同一个项目编号的负责人加到paper人员里面'''
#返回列表
#file 文件
#row 姓名列
#item 要匹配项目编号字典
#姓名列列表 3,4同一文件
def addnamelist(file,row,item,row_name):
    data = pd.read_csv(file)
    x = data[row].tolist()
    for i in range(len(x)):
        for m, n in item.items():
            if (str(paper_umber[i])) == n and (n!=0):
                str_name = str(x[i])+ ';' +str(row_name[m])+ ';'
                x[i] = str_name
    return x

if __name__ == '__main__':
    path_paper = '135paper.csv'
    path_paper1 = '135paper1.csv'
    path_thesis = '135thesis.csv'
    path_alldata = '135alldata.csv'
    data_alldata = pd.read_csv(path_alldata)

    #提取paper Fund-基金 项目编号
    paper_fund='Fund-基金'
    paper_umber=fund_number(path_paper,paper_fund)

    # 提取paper Fund-基金 项目编号
    thesis_fund = 'Fund-基金'
    thesis_umber = fund_number(path_thesis, thesis_fund)

    # 提取alldata-基金项目编号
    alldata_fund = '项目编号'
    alldata_number = data_alldata[alldata_fund].to_dict()

    # thesis作者(次)
    data_thesis = pd.read_csv(path_thesis)
    thesis_name = data_thesis['Author-作者'].to_list()

    # alldata项目负责人(主要)
    alldata_name = data_alldata['项目负责人'].to_list()

    # 合并alldata和theis的负责人和姓名
    node = []
    node.extend(alldata_name)
    node.extend(thesis_name)

    # paper与thesis项目关联生成列表
    # addnamelist = addnamelist(path_paper, 'Author-作者', thesis_umber, thesis_name)
    addnamelist = addnamelist(path_paper, 'Author-作者', alldata_number, alldata_name)
    edge = []
    nameToRowmerge=nameToRowmerge(addnamelist)#全部姓名对[[],,]
    #查看合并组合结果
    # print(nameToRowmerge)
    for name_line in range(len(nameToRowmerge)):
        #姓名列表转为边
        # nameToRowmerge[name_line]一行名族列表
        # paper_umber[name_line]一行项目编码
        edge.extend(nemeToEdgeadd(nameToRowmerge[name_line],paper_umber[name_line]))

    # 导出为csv文件
    test=pd.DataFrame(edge)
    print(test.values)
    test.to_csv('d:/edgeadd.csv',encoding='gbk')
```

### 生成文件格式预览

![](C:\Users\刘德智\Desktop\data.png)

# 绘图drawNet.py

```python
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import warnings
warnings.simplefilter('ignore')

data = pd.read_csv('edgealldata.csv',encoding='gbk')
svos=data.values

g_nx = nx.Graph()
labels = {}

for i in range(len(svos)):#遍历所有数据()
#for i in range(8000,8050):#建议先少量节点测试

    name1=svos[i][1]
    name2=svos[i][2]
    obj=svos[i][3]
    g_nx.add_edge(name1, name2)
    labels[(name1, name2)] = obj

fig=plt.figure(figsize=(12,12),dpi=100)
plt.rcParams['font.family']='sans-serif'
plt.rcParams['font.sans-serif']='SimHei'
plt.rcParams['axes.unicode_minus'] = False
pos =nx.spring_layout(g_nx)
nx.draw_networkx_nodes(g_nx,pos,node_size=5)
nx.draw_networkx_edges(g_nx,pos,width=0.1)

#数据多这两个可不要
# nx.draw_networkx_labels(g_nx,pos,font_size=5)#节点标签
# nx.draw_networkx_edge_labels(g_nx,pos,labels,font_size=5)#边标签

plt.axis('off')
plt.show()
```

### 小数据测试

![](C:\Users\刘德智\Desktop\pic1.png)

### 待优化

![](C:\Users\刘德智\Desktop\pic2.png)