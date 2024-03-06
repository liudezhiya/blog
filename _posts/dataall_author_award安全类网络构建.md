# dataall_author_award安全类网络构建

[TOC]

### 1.构建数据来源

2018年自科数据（award -author-alldata）数据中选取“安全”主题数据

### 2.建边规则

①用alldata里面的admin分别与author里的name建边
②用alldata里面的admin与award里面author字段里面的姓名两两建边（其中award里面姓名是去除award里面含有中英文姓名的admin，如果有在author里面的作者的英文名转为author的姓名）

③统计重复边数记录作为权值，以保留第一次记录去重以权值降序排序后生成权值边，对参与组边人员去除生成节点。

### 3.代码实现

#### 3.1导入包

```python
import csv
import os
import re
import time
import pandas as pd
import glob
import pypinyin
```

#### 3.2方法说明

①将将中文姓名转为英文姓名，用于判断award姓名是不是含有alldata姓名

```python
#将中文姓名转为英文姓名
def nameToPingYing(str):
    result = pypinyin.pinyin(str, style=pypinyin.NORMAL)
    result = [i[0] for i in result]
    name1 = ''.join(result[1:]).capitalize() + ' ' + result[0].capitalize()
    name2 = result[0].capitalize() + ' ' + ''.join(result[1:]).capitalize()
    return [name1, name2]

```

②将alldata和athour姓名列表转为英语姓名

```python
    
#将姓名列表转为英文姓名字典
def namelistToPydict(namelist):
    chineseAndEngdict = {}
    for name in namelist:
        chineseAndEngdict[name]=nameToPingYing(name)
    return chineseAndEngdict

```

③将award里面在dataall和author里面出现的姓名转回中文姓名

```python
#将拼音转为中文
def EnameToCname(name,name_dict):
    # name.strip()
    for k, v in name_dict.items():
        if name in v:
            name = k
    return name
```

④将提取的姓名列表组合生成边

```python
#多姓名列表转为边
def nemeToEdge(namelist):
    edge = []
    for name_line in namelist:
        edgeline = []
        for i in range(0, len(name_line)):
            for j in range(i + 1, len(name_line)):
                if (name_line[i] != name_line[j]):
                    edge_one = []
                    edge_one.append(name_line[i])
                    edge_one.append(name_line[j])
                edgeline.append(edge_one)
        edge.extend(edgeline)
    return edge

```

⑤将边保存为csv文件

```python

#将df边生成csv文件
def edgeTocsv(edge):
    #为边指定列名
    data = pd.DataFrame(edge, columns=['source', 'target'])
    # test.drop_duplicates(inplace=True)去重 true删除重复记录
    timestr = time.strftime("%Y%m%d-%H%M%S")  # 20220715-085113
    # 查看数据
    # print(test.values)
    # 保存边到csv文件
    # test.to_csv('edge' + timestr+'.csv',encoding='utf8',index=False,errors='ignore')
    data.to_csv('edge.csv',encoding='utf-8',index=False,errors='ignore')


```

⑥将边转为权值边

```python
#去重统计权生成权值边
def edgeToWeight(edge_csv):
    df = pd.read_csv(edge_csv)
    df = df[['source', 'target']].value_counts().reset_index()
    df.columns=['source','target','weight']
    df=pd.DataFrame(df)
    df.to_csv('edge_weight.csv',index=False)

```

⑦将节点类型去重

```python
#将类型去重
def typeDistinct(type_scv):
    df = pd.read_csv(type_scv, encoding='utf-8')
    data = df.drop_duplicates('Id',keep='first')
    data.to_csv('type_distinct.csv',index=False)#False删除所有重复项
```

#### 3.3主函数

```python
if __name__ == '__main__':

    # 测试文件路径
    # alldata_directory = 'D:\\workspace\\pythonspace\\alldataanalysis\\alldata'
    # author_directory = 'D:\\workspace\\pythonspace\\alldataanalysis\\author'
    # award_directory = 'D:\\workspace\\pythonspace\\alldataanalysis\\award'

    # 测试文件路径all
    alldata_directory='D://workspace//data//18-22\data_all//2018'
    author_directory='D:\workspace\data//18-22//author//2018'
    award_directory='D:\workspace\data//18-22//award//2018'

    # 创建文件对象
    f = open('type.csv', 'w', encoding='utf-8', errors='ignore',newline='')
    f_edge = open('edge.csv', 'w', encoding='utf-8', errors='ignore',newline='')
    # 基于文件对象构建csv写入对象
    type_writer = csv.writer(f)
    edge_writer = csv.writer(f_edge)
    # 指定文件列名
    type_writer.writerow(['Id', 'Type'])
    edge_writer.writerow(['source', 'target'])
    #创建alldata文件遍历目录
    alldata_filenames = glob.glob(alldata_directory + "\*.csv")
    namelist = []#[[],[]]未成边组合

    # 计数
    alldatalinecount = 0
    authorlinecount = 0
    awardlinecount = 0


    for alldata_filename in alldata_filenames:  # alldata文件目录遍历

        '''alldata_filename alldata里面单个文件名路径'''
        alldataPureFileName = re.search('[0-9]{4}_[A-Z][0-9]{2}', alldata_filename)
        print('进行到：', alldataPureFileName.group())
        df = pd.read_csv(alldata_filename)
        uuid_admin = df[['proJName','uuid', 'admin']]

        #dataall负责人和author里人员生成英文名字典 用于将英文转为中文
        cToENamedict = []

        for i in range(len(uuid_admin)):  # alldata单个文件遍历
            # alldatalinecount+=1#统计全部

            '''************选择安全类型************'''
            #proJName提交项目名字段
            proJName=uuid_admin['proJName'][i]
            if '安全' in proJName:
                alldatalinecount += 1#统计安全类别
            #########以下在if下面###########
                #记录文件名id
                authorAndAwardFileId = uuid_admin['uuid'][i]
                # 类型
                admin = uuid_admin['admin'][i]  # 项目负责人
                admin_py=nameToPingYing(admin) #项目负责人转为英文名
                #写入alldata姓名类型
                type_writer.writerow([admin, 'admin'])

                '''************文件匹配到author****************'''

                author_pre_str = 'author' + '_' + alldataPureFileName.group()
                # 由alldata文件得到的author文件名
                authorAndAwardFilename = author_pre_str + '_' + authorAndAwardFileId + '.csv'
                # author读取文件路径名
                author_directory_p = author_directory + '\\' + authorAndAwardFilename

                for author_filename in os.listdir(author_directory):  # 对author文件目录																			进行遍历
                    if str(authorAndAwardFilename) == str(author_filename):#找对应文件
                        author_df = pd.read_csv(author_directory_p)
                        author_file = author_df[['uuid', 'name']]  # 提取匹配成功																			的'uuid'和'name'列
                        # 对匹配的author单个文件操作
                        oneAuthorFileNameList = [admin]#记录admin
                        for author in range(len(author_file)):  # 对author文件目录里单个																		文件进行遍历
                            authorlinecount+=1

                            # 类型
                            #author文件里面的author姓名
                            authorname = author_file['name'][author]
                            #写入author类型
                            type_writer.writerow([authorname, 'author'])

                            if authorname not in oneAuthorFileNameList:
                                #添加author到组边列表
                                oneAuthorFileNameList.append(authorname)
                                #如果要直接写入边就在这里[admin,authorname]

                        if len(oneAuthorFileNameList)>1:
                            #加入一个author文件里面的作者列表
                            namelist.append(oneAuthorFileNameList)
                            # dataall负责人和author里人员生成英文名字典 用于将英文转为中文
                            cToENamedict = namelistToPydict(oneAuthorFileNameList)


                '''***************文件匹配到award**********************'''
                award_pre_str = 'award' + '_' + alldataPureFileName.group()
                # 由alldata文件得到的author文件名
                authorAndAwardFilename = award_pre_str + '_' + authorAndAwardFileId + '.csv'
                # award单个文件路径
                award_directory_p = award_directory + '\\' + award_pre_str + '_' + authorAndAwardFileId + '.csv'

                for award_filename in os.listdir(award_directory):  # 对award文件目录进                                                         行遍历
                    # award_filename award里面单个文件名
                    if str(authorAndAwardFilename) == str(award_filename):
                        award_df = pd.read_csv(award_directory_p)
                        # print(award_df[['uuid','authors']])
                        aware_file = award_df[['uuid', 'authors']]  # 提取匹配成功award																 里的'uuid'和'authors'列
                        # 对匹配的award单个文件操作
                        # oneAwardFileNameList = [admin]#整个文件生成姓名列表
                        oneAwardFileNameList = []  # award每一行生成列表
                        for award in range(len(aware_file)):  # 对award文件目录中单个文件																		进行遍历
                            awardlinecount+=1
                            # print('aware',uuid_admin['admin'][i],aware_file['name']                                                                  [award])

                            # 类型
                            # print('award',aware_file['name'][award])
                            # awardname期刊一行作者符串
                            awardname = aware_file['authors'][award]
                            # type_writer.writerow([awardname, 'award'])


                            '''*************将一个整个文件里面的一行姓名生成一个列表**************************'''
                            # 将award里面单个文件里面的一行数据切割开生成列表
                            awardsourand = [admin]
                            #拆分一行award记录姓名字段
                            awardnamelinelist = re.split(r';', str(awardname))
                            #遍历一行award记录姓名字段
                            for name in awardnamelinelist:
                                # 如果是alldata和authou里面的中文姓名 写的英文文章 将英文名转																					为中文
                                name = EnameToCname(name, cToENamedict)
                                if name != admin:  # 匹配不是admin 去掉中文混入的英文
                                    #添加姓名到组边列表
                                    awardsourand.append(name)
                                    #写入类型
                                    type_writer.writerow([name, 'award'])
                            if len(awardsourand) > 1:  # 有其他作者才加
                                namelist.append(awardsourand)  # 加入的一篇期刊作者列表
                            '''****************************************************************'''


    f.close()
    f_edge.close()
    # print(namelist)#输出关联数据多列表
    edge = nemeToEdge(namelist)
    #将边保存为edge.csv文件
    edgeTocsv(edge)
    #将生成的边转为权值边生成权值边edge_weigth.csv文件
    edgeToWeight('edge.csv')
    #将节点类型去重生成type_distinct.csv文件
    typeDistinct('type.csv')

    # 输出个文件记录条数
    print('alldata文件行数:',alldatalinecount)
    print('author文件行数:',authorlinecount)
    print('award文件行数:',awardlinecount)
```

### 4.输出说明

#### 4.1控制台输出

输出进行到alldata目录下面那个文件

```python
print('进行到：', alldataPureFileName.group())
输出如下
进行到： 2018_A01
进行到： 2018_A02
进行到： 2018_A03
```

#### 4.1文件输出

```
边数据文件           edge.csv

权值边数据文件        edge_weight.csv

节点类型文件         type.csv

节点去重文件         type_distinct.csv
```



