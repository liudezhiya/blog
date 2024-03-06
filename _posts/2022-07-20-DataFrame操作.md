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

###  

[TOC]

# DataFrame基本函数整理

###  构造函数 

```
DataFrame([data, index, columns, dtype, copy]) #构造数据

创建 DataFrame 的四种方法
①创建一个空的数据框架
import pandas as pd
df = pd.DataFrame()
②从列表进行创建
data = [1,2,3,4,5]
df = pd.DataFrame(data) # 将列表数据转化为 一列
data = [['Alex',10],['Bob',12],['Clarke',13]]
df = pd.DataFrame(data,columns=['Name','Age']) # 将第一维度数据转为为行，第二维度数据转化为列，即 3 行 2 列，并设置列标签
data = [['Alex',10],['Bob',12],['Clarke',13]]
df = pd.DataFrame(data,columns=['Name','Age'],dtype=float) # 将数字元素 自动转化为 浮点数
③从 ndarrays / Lists 的 字典创建
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'],'Age':[28,34,29,42]} # 两组列元素，并且个数需要相同
df = pd.DataFrame(data) # 这里默认的 index 就是 range(n)，n 是列表的长度
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'],'Age':[28,34,29,42]}
df = pd.DataFrame(data, index=['rank1','rank2','rank3','rank4']) # 这里设定了 index 个数要和列表长度一致
④从 字典组成的列表 创建
data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}] # 列表对应的是第一维，即行，字典为同一行不同列元素
df = pd.DataFrame(data) # 第 1 行 3 列没有元素，自动添加 NaN (Not a Number)
   a   b     c
0  1   2   NaN
1  5  10  20.0
取特定的表头下的列元素
data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]

#With two column indices, values same as dictionary keys
df1 = pd.DataFrame(data, index=['first', 'second'], columns=['a', 'b']) # 指定表头都存在于 data，只取部分

#With two column indices with one index with other name
df2 = pd.DataFrame(data, index=['first', 'second'], columns=['a', 'b1']) # 指定表头中 b1 不存在，添加 b1 列，元素 NaN
print(df1)
print(df2)
        a   b
first   1   2
second  5  10
        a  b1
first   1 NaN
second  5 NaN
⑤从 Series 组成的字典 创建
d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']),
   'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}
# index 与序列长度相投
# 字典不同的 key 代表一个列的表头，pd.Series 作为 value 作为该列的元素
df = pd.DataFrame(d)
print(df)
   one  two
a  1.0    1
b  2.0    2
c  3.0    3
d  NaN    4
```



### 属性和数据

```python
DataFrame.axes                                #index: 行标签；columns: 列标签
DataFrame.as_matrix([columns])                #转换为矩阵
DataFrame.dtypes                              #返回数据的类型
DataFrame.ftypes                              #返回每一列的 数据类型float64:dense
DataFrame.get_dtype_counts()                  #返回数据框数据类型的个数
DataFrame.get_ftype_counts()                  #返回数据框数据类型float64:dense的个数
DataFrame.select_dtypes([include, include])   #根据数据类型选取子数据框
DataFrame.values                              #Numpy的展示方式
DataFrame.axes                                #返回横纵坐标的标签名
DataFrame.ndim                                #返回数据框的纬度
DataFrame.size                                #返回数据框元素的个数
DataFrame.shape                               #返回数据框的形状
DataFrame.memory_usage()                      #每一列的存储
```

### 类型转换

```
DataFrame.astype(dtype[, copy, errors])       #转换数据类型
DataFrame.copy([deep])                        #deep深度复制数据
DataFrame.isnull()                            #以布尔的方式返回空值
DataFrame.notnull()                           #以布尔的方式返回非空值
```

### 索引和迭代

```python
DataFrame.head([n])                           #返回前n行数据
DataFrame.at                                  #快速标签常量访问器
DataFrame.iat                                 #快速整型常量访问器
DataFrame.loc                                 #标签定位，使用名称
DataFrame.iloc                                #整型定位，使用数字
DataFrame.insert(loc, column, value)          #在特殊地点loc[数字]插入column[列名]某列数据
DataFrame.iter()                              #Iterate over infor axis
DataFrame.iteritems()                         #返回列名和序列的迭代器
DataFrame.iterrows()                          #返回索引和序列的迭代器
DataFrame.itertuples([index, name])           #Iterate over DataFrame rows as namedtuples, with index value as first element of the tuple.
DataFrame.lookup(row_labels, col_labels)      #Label-based “fancy indexing” function for DataFrame.
DataFrame.pop(item)                           #返回删除的项目
DataFrame.tail([n])                           #返回最后n行
DataFrame.xs(key[, axis, level, drop_level])  #Returns a cross-section (row(s) or column(s)) from the Series/DataFrame.
DataFrame.isin(values)                        #是否包含数据框中的元素
DataFrame.where(cond[, other, inplace, …])    #条件筛选
DataFrame.mask(cond[, other, inplace, …])     #Return an object of same shape as self and whose corresponding entries are from self where cond is False and otherwise are from other.
DataFrame.query(expr[, inplace])              #Query the columns of a frame with a boolean expression.
```

### 二元运算

```python
DataFrame.add(other[,axis,fill_value])        #加法，元素指向
DataFrame.sub(other[,axis,fill_value])        #减法，元素指向
DataFrame.mul(other[, axis,fill_value])       #乘法，元素指向
DataFrame.div(other[, axis,fill_value])       #小数除法，元素指向
DataFrame.truediv(other[, axis, level, …])    #真除法，元素指向
DataFrame.floordiv(other[, axis, level, …])   #向下取整除法，元素指向
DataFrame.mod(other[, axis,fill_value])       #模运算，元素指向
DataFrame.pow(other[, axis,fill_value])       #幂运算，元素指向
DataFrame.radd(other[, axis,fill_value])      #右侧加法，元素指向
DataFrame.rsub(other[, axis,fill_value])      #右侧减法，元素指向
DataFrame.rmul(other[, axis,fill_value])      #右侧乘法，元素指向
DataFrame.rdiv(other[, axis,fill_value])      #右侧小数除法，元素指向
DataFrame.rtruediv(other[, axis, …])          #右侧真除法，元素指向
DataFrame.rfloordiv(other[, axis, …])         #右侧向下取整除法，元素指向
DataFrame.rmod(other[, axis,fill_value])      #右侧模运算，元素指向
DataFrame.rpow(other[, axis,fill_value])      #右侧幂运算，元素指向
DataFrame.lt(other[, axis, level])            #类似Array.lt
DataFrame.gt(other[, axis, level])            #类似Array.gt
DataFrame.le(other[, axis, level])            #类似Array.le
DataFrame.ge(other[, axis, level])            #类似Array.ge
DataFrame.ne(other[, axis, level])            #类似Array.ne
DataFrame.eq(other[, axis, level])            #类似Array.eq
DataFrame.combine(other,func[,fill_value, …]) #Add two DataFrame objects and do not propagate NaN values, so if for a
DataFrame.combine_first(other)                #Combine two DataFrame objects and default to non-null values in frame calling the method.
```

### 函数应用&分组&窗口

```python
DataFrame.apply(func[, axis, broadcast, …])   #应用函数
DataFrame.applymap(func)                      #Apply a function to a DataFrame that is intended to operate elementwise, i.e.
DataFrame.aggregate(func[, axis])             #Aggregate using callable, string, dict, or list of string/callables
DataFrame.transform(func, *args, **kwargs)    #Call function producing a like-indexed NDFrame
DataFrame.groupby([by, axis, level, …])       #分组
DataFrame.rolling(window[, min_periods, …])   #滚动窗口
DataFrame.expanding([min_periods, freq, …])   #拓展窗口
DataFrame.ewm([com, span, halflife,  …])      #指数权重窗口
```

### 描述统计学

```python
DataFrame.abs()                               #返回绝对值
DataFrame.all([axis, bool_only, skipna])      #Return whether all elements are True over requested axis
DataFrame.any([axis, bool_only, skipna])      #Return whether any element is True over requested axis
DataFrame.clip([lower, upper, axis])          #Trim values at input threshold(s).
DataFrame.clip_lower(threshold[, axis])       #Return copy of the input with values below given value(s) truncated.
DataFrame.clip_upper(threshold[, axis])       #Return copy of input with values above given value(s) truncated.
DataFrame.corr([method, min_periods])         #返回本数据框成对列的相关性系数
DataFrame.corrwith(other[, axis, drop])       #返回不同数据框的相关性
DataFrame.count([axis, level, numeric_only])  #返回非空元素的个数
DataFrame.cov([min_periods])                  #计算协方差
DataFrame.cummax([axis, skipna])              #Return cumulative max over requested axis.
DataFrame.cummin([axis, skipna])              #Return cumulative minimum over requested axis.
DataFrame.cumprod([axis, skipna])             #返回累积
DataFrame.cumsum([axis, skipna])              #返回累和
DataFrame.describe([percentiles,include, …])  #整体描述数据框
DataFrame.diff([periods, axis])               #1st discrete difference of object
DataFrame.eval(expr[, inplace])               #Evaluate an expression in the context of the calling DataFrame instance.
DataFrame.kurt([axis, skipna, level, …])      #返回无偏峰度Fisher’s  (kurtosis of normal == 0.0).
DataFrame.mad([axis, skipna, level])          #返回偏差
DataFrame.max([axis, skipna, level, …])       #返回最大值
DataFrame.mean([axis, skipna, level, …])      #返回均值
DataFrame.median([axis, skipna, level, …])    #返回中位数
DataFrame.min([axis, skipna, level, …])       #返回最小值
DataFrame.mode([axis, numeric_only])          #返回众数
DataFrame.pct_change([periods, fill_method])  #返回百分比变化
DataFrame.prod([axis, skipna, level, …])      #返回连乘积
DataFrame.quantile([q, axis, numeric_only])   #返回分位数
DataFrame.rank([axis, method, numeric_only])  #返回数字的排序
DataFrame.round([decimals])                   #Round a DataFrame to a variable number of decimal places.
DataFrame.sem([axis, skipna, level, ddof])    #返回无偏标准误
DataFrame.skew([axis, skipna, level, …])      #返回无偏偏度
DataFrame.sum([axis, skipna, level, …])       #求和
DataFrame.std([axis, skipna, level, ddof])    #返回标准误差
DataFrame.var([axis, skipna, level, ddof])    #返回无偏误差 
```

### 从新索引&选取&标签操作

```python
DataFrame.add_prefix(prefix)                  #添加前缀
DataFrame.add_suffix(suffix)                  #添加后缀
DataFrame.align(other[, join, axis, level])   #Align two object on their axes with the
DataFrame.drop(labels[, axis, level, …])      #返回删除的列
DataFrame.drop_duplicates([subset, keep, …])  #Return DataFrame with duplicate rows removed, optionally only
DataFrame.duplicated([subset, keep])          #Return boolean Series denoting duplicate rows, optionally only
DataFrame.equals(other)                       #两个数据框是否相同
DataFrame.filter([items, like, regex, axis])  #过滤特定的子数据框
DataFrame.first(offset)                       #Convenience method for subsetting initial periods of time series data based on a date offset.
DataFrame.head([n])                           #返回前n行
DataFrame.idxmax([axis, skipna])              #Return index of first occurrence of maximum over requested axis.
DataFrame.idxmin([axis, skipna])              #Return index of first occurrence of minimum over requested axis.
DataFrame.last(offset)                        #Convenience method for subsetting final periods of time series data based on a date offset.
DataFrame.reindex([index, columns])           #Conform DataFrame to new index with optional filling logic, placing NA/NaN in locations having no value in the previous index.
DataFrame.reindex_axis(labels[, axis, …])     #Conform input object to new index with optional filling logic, placing NA/NaN in locations having no value in the previous index.
DataFrame.reindex_like(other[, method, …])    #Return an object with matching indices to myself.
DataFrame.rename([index, columns])            #Alter axes input function or functions.
DataFrame.rename_axis(mapper[, axis, copy])   #Alter index and / or columns using input function or functions.
DataFrame.reset_index([level, drop, …])       #For DataFrame with multi-level index, return new DataFrame with labeling information in the columns under the index names, defaulting to ‘level_0’, ‘level_1’, etc.
DataFrame.sample([n, frac, replace, …])       #返回随机抽样
DataFrame.select(crit[, axis])                #Return data corresponding to axis labels matching criteria
DataFrame.set_index(keys[, drop, append ])    #Set the DataFrame index (row labels) using one or more existing columns.
DataFrame.tail([n])                           #返回最后几行
DataFrame.take(indices[, axis, convert])      #Analogous to ndarray.take
DataFrame.truncate([before, after, axis ])    #Truncates a sorted NDFrame before and/or after some particular index value.
```

### 处理缺失值

```python
DataFrame.dropna([axis, how, thresh, …])      #Return object with labels on given axis omitted where alternately any
DataFrame.fillna([value, method, axis, …])    #填充空值
DataFrame.replace([to_replace, value, …])     #Replace values given in ‘to_replace’ with ‘value’.
```

### 删除某列空值所在的行

```
data.dropna(how = 'all')    # 传入这个参数后将只丢弃全为缺失值的那些行
data.dropna(axis = 1)       # 丢弃有缺失值的列（一般不会这么做，这样会删掉一个特征）
data.dropna(axis=1,how="all")   # 丢弃全为缺失值的那些列
data.dropna(axis=0,subset = ["Age", "Sex"])   # 丢弃‘Age’和‘Sex’这两列中有缺失值的行
```



### 从新定型&排序&转变形态

```python
DataFrame.pivot([index, columns, values])     #Reshape data (produce a “pivot” table) based on column values.
DataFrame.reorder_levels(order[, axis])       #Rearrange index levels using input order.
DataFrame.sort_values(by[, axis, ascending])  #Sort by the values along either axis
DataFrame.sort_index([axis, level, …])        #Sort object by labels (along an axis)
DataFrame.nlargest(n, columns[, keep])        #Get the rows of a DataFrame sorted by the n largest values of columns.
DataFrame.nsmallest(n, columns[, keep])       #Get the rows of a DataFrame sorted by the n smallest values of columns.
DataFrame.swaplevel([i, j, axis])             #Swap levels i and j in a MultiIndex on a particular axis
DataFrame.stack([level, dropna])              #Pivot a level of the (possibly hierarchical) column labels, returning a DataFrame (or Series in the case of an object with a single level of column labels) having a hierarchical index with a new inner-most level of row labels.
DataFrame.unstack([level, fill_value])        #Pivot a level of the (necessarily hierarchical) index labels, returning a DataFrame having a new level of column labels whose inner-most level consists of the pivoted index labels.
DataFrame.melt([id_vars, value_vars, …])      #“Unpivots” a DataFrame from wide format to long format, optionally
DataFrame.T                                   #Transpose index and columns
DataFrame.to_panel()                          #Transform long (stacked) format (DataFrame) into wide (3D, Panel) format.
DataFrame.to_xarray()                         #Return an xarray object from the pandas object.
DataFrame.transpose(*args, **kwargs)          #Transpose index and columns

Combining& joining&merging
DataFrame.append(other[, ignore_index, …])    #追加数据
DataFrame.assign(**kwargs)                    #Assign new columns to a DataFrame, returning a new object (a copy) with all the original columns in addition to the new ones.
DataFrame.join(other[, on, how, lsuffix, …])  #Join columns with other DataFrame either on index or on a key column.
DataFrame.merge(right[, how, on, left_on, …]) #Merge DataFrame objects by performing a database-style join operation by columns or indexes.
DataFrame.update(other[, join, overwrite, …]) #Modify DataFrame in place using non-NA values from passed DataFrame.
```

### 时间序列

```python
DataFrame.asfreq(freq[, method, how, …])      #将时间序列转换为特定的频次
DataFrame.asof(where[, subset])               #The last row without any NaN is taken (or the last row without
DataFrame.shift([periods, freq, axis])        #Shift index by desired number of periods with an optional time freq
DataFrame.first_valid_index()                 #Return label for first non-NA/null value
DataFrame.last_valid_index()                  #Return label for last non-NA/null value
DataFrame.resample(rule[, how, axis, …])      #Convenience method for frequency conversion and resampling of time series.
DataFrame.to_period([freq, axis, copy])       #Convert DataFrame from DatetimeIndex to PeriodIndex with desired
DataFrame.to_timestamp([freq, how, axis])     #Cast to DatetimeIndex of timestamps, at beginning of period
DataFrame.tz_convert(tz[, axis, level, copy]) #Convert tz-aware axis to target time zone.
DataFrame.tz_localize(tz[, axis, level, …])   #Localize tz-naive TimeSeries to target time zone.
```

### 作图

````python
DataFrame.plot([x, y, kind, ax, ….])          #DataFrame plotting accessor and method
DataFrame.plot.area([x, y])                   #面积图Area plot
DataFrame.plot.bar([x, y])                    #垂直条形图Vertical bar plot
DataFrame.plot.barh([x, y])                   #水平条形图Horizontal bar plot
DataFrame.plot.box([by])                      #箱图Boxplot
DataFrame.plot.density(**kwds)                #核密度Kernel Density Estimate plot
DataFrame.plot.hexbin(x, y[, C, …])           #Hexbin plot
DataFrame.plot.hist([by, bins])               #直方图Histogram
DataFrame.plot.kde(**kwds)                    #核密度Kernel Density Estimate plot
DataFrame.plot.line([x, y])                   #线图Line plot
DataFrame.plot.pie([y])                       #饼图Pie chart
DataFrame.plot.scatter(x, y[, s, c])          #散点图Scatter plot
DataFrame.boxplot([column, by, ax, …])        #Make a box plot from DataFrame column optionally grouped by some columns or
DataFrame.hist(data[, column, by, grid, …])   #Draw histogram of the DataFrame’s series using matplotlib / pylab.
````

转换为其他格式

```python
DataFrame.from_csv(path[, header, sep, …])    #Read CSV file (DEPRECATED, please use pandas.read_csv() instead).
DataFrame.from_dict(data[, orient, dtype])    #Construct DataFrame from dict of array-like or dicts
DataFrame.from_items(items[,columns,orient])  #Convert (key, value) pairs to DataFrame.
DataFrame.from_records(data[, index, …])      #Convert structured or record ndarray to DataFrame
DataFrame.info([verbose, buf, max_cols, …])   #Concise summary of a DataFrame.
DataFrame.to_pickle(path[, compression, …])   #Pickle (serialize) object to input file path.
DataFrame.to_csv([path_or_buf, sep, na_rep])  #Write DataFrame to a comma-separated values (csv) file
DataFrame.to_hdf(path_or_buf, key, **kwargs)  #Write the contained data to an HDF5 file using HDFStore.
DataFrame.to_sql(name, con[, flavor, …])      #Write records stored in a DataFrame to a SQL database.
DataFrame.to_dict([orient, into])             #Convert DataFrame to dictionary.
DataFrame.to_excel(excel_writer[, …])         #Write DataFrame to an excel sheet
DataFrame.to_json([path_or_buf, orient, …])   #Convert the object to a JSON string.
DataFrame.to_html([buf, columns, col_space])  #Render a DataFrame as an HTML table.
DataFrame.to_feather(fname)                   #write out the binary feather-format for DataFrames
DataFrame.to_latex([buf, columns, …])         #Render an object to a tabular environment table.
DataFrame.to_stata(fname[, convert_dates, …]) #A class for writing Stata binary dta files from array-like objects
DataFrame.to_msgpack([path_or_buf, encoding]) #msgpack (serialize) object to input file path
DataFrame.to_sparse([fill_value, kind])       #Convert to SparseDataFrame
DataFrame.to_dense()                          #Return dense representation of NDFrame (as opposed to sparse)
DataFrame.to_string([buf, columns, …])        #Render a DataFrame to a console-friendly tabular output.
DataFrame.to_clipboard([excel, sep])          #Attempt to write text representation of object to the system clipboard 
```



### p andas常用的数据查看方法有 

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
 
 re.split(pattern, string, maxsplit=0, flags=0) #maxsplit为最大分割次数，flags为正则表达式用到的通用标志：
  re.split(r'[,:;]', s)
  re.split(r'([,:;])', s)
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

### drop_duplicates方法介绍

方法形式为 **drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)**，返回删掉重复行的Dataframe。

**参数解析：** - **subset：**列名或列名序列，对某些列来识别重复项，默认情况下使用所有列。

- **keep：**可选值有first，last，False，默认为first，确定要保留哪些重复项。
  
- - first：删除除第一次出现的重复项，即保留第一次出现的重复项。
  - last：保留最后一次出现的重复项。
  - False：删除所有重复项。

- **inplace：**[布尔值](https://www.zhihu.com/search?q=布尔值&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2458719205})，默认为False，返回副本。如果为True，则直接在原始的Dataframe上进行删除。
  
- **ignore_index：**布尔值，默认为False，如果为True，则生成的行索引将被标记为0、1、2、...、n-1。
  

**返回：** - 返回删除重复项的Dataframe或None，当inplace=True时返回None。

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





#### 更改列名

```
# 更换列名，同时显示出来
df.rename(columns={'a':'A'}, inplace=True)
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
df.reset_index(inplace=True)
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

```
apply
```

```
data['words'] = data.abstract_of_keywords.apply(lambda x: extract_word(x))
```



### 将一列的列表展开成多列

```
df1=df['col'].apply(pd.Series,index=['col1','col2','col3'])

实例1
award_data[['astv1','astv2','astv3','astv4','astv5']]=award_data['abstractVectorlist'].apply(pd.Series,index=['astv1','astv2','astv3','astv4','astv5'])

实例2
award_data[['astv1','astv2','astv3','astv4','astv5']] = pd.DataFrame(award_data['abstractVectorlist'].apply(pd.Series).values)

```

下面报错

```
# award_data[['astv1','astv2','astv3','astv4','astv5']] = award_data.explode('abstractVectorlist')
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
    np.random.exponential(5, [2,2])#生成一个指数分布的随机数，均值为 5， 2 行 2 列
```

三，生成随机字符串

```
  s = string.ascii_letters+string.digits# string.ascii_letters大小写字母
  a=[random.choice(s) for i in range(1000)]
  b=[random.choice(s) for i in range(1000)]
```







# 遍历dataframe

|                        |                                |
| ---------------------- | ------------------------------ |
| DataFrame.iterrows()   | 按行顺序优先，接着依次按列迭代 |
| DataFrame.iteritems()  | 按列顺序优先，接着依次按行迭代 |
| DataFrame.itertuples() | 按行顺序优先，接着依次按列迭代 |

## 按行遍历

通过for迭代df.iterrows接口，idx是输出DataFrame内部的索引值,data输出每行单元格的值

```python3
for idx,data in df.iterrows():
    print("[{}]: {}".format(idx,data))

for date, row in df.iterrows():
    print(date)


```

 按行优先的遍历方式，还有itertuples( )函数，它将返回一个生成器，该生成器以元组生成行值。 

```text
for data in df.itertuples():
    print(data)
    
for row in df.itertuples():
    print(getattr(row, 'c1'), getattr(row, 'c2'))
```

## 按列遍历

现在，要遍历此DataFrame，我们将使用items( )或iteritems( )函数：

```python3
or colName,data in df.items():
    print("colName:[{}]\ndata:{}".format(colName,data))
```

 如果我们按列优先，仅遍历某一行依次遍历所有列 

```python3
for colName,data in df.iteritems():
    print("colName:[{}]\ndata:{}".format(colName,data[2]))
    
for date, row in df.iteritems():
    print(row[0], row[1], row[2])
```

 按行遍历的iterrows的性能是最差的，而按行遍历返回tuple的方式性能是最好的，其次是按列遍历的i考虑的teritems是可以考虑的 



```
def valuation_formula(x, y):    return str(x) + str(y)data['price'] = data.apply(lambda row: valuation_formula(row['Author-作者'], row['Organ-单位']), axis=1)
```

## python DataFrame的合并方法

#### inner（默认）

**使用来自两个数据集的键的交集**

```python
df_merge = pd.merge(df1, df2, on='id')
```

#### outer

**使用来自两个数据集的键的并集**

```python
df_merge = pd.merge(df1, df2, on='id', how="outer")
```

#### left

使用来自左数据集的键

```python
df_merge = pd.merge(df1, df2, on='id', how='left')
```

#### right

使用来自右数据集的键

```python
df_merge = pd.merge(df1, df2, on='id', how='right')
```

```python
df_merge = pd.merge(df1, df2, on='id')
```

 依然按照默认的Inner方式，使用来自两个数据集的键的交集。且重复的键的行会在合并结果中体现为多行 

# [concat](https://so.csdn.net/so/search?q=concat&spm=1001.2101.3001.7020)()

    pd.concat(objs, axis=0, join=‘outer’, ignore_index:bool=False,keys=None,levels=None,names=None, verify_integrity:bool=False,sort:bool=False,copy:bool=True)

| 参数         | 描述                                                         |
| ------------ | ------------------------------------------------------------ |
| objs         | Series，DataFrame或Panel对象的序列或映射                     |
| axis         | 默认为0，表示列。如果为1则表示行。                           |
| join         | 默认为"outer"，也可以为"inner"                               |
| ignore_index | 默认为False，表示保留索引（不忽略）。设为True则表示忽略索引。 |

```python
dfs = [df1, df2, df3]
result = pd.concat(dfs)
```

 如果想要在合并后，标记一下数据都来自于哪张表或者数据的某类别，则也可以给concat加上 **参数keys** 。 

```python
result = pd.concat(dfs, keys=['table1', 'table2', 'table3'])
```

 此时，添加的keys与原来的index组成元组，共同成为新的index。 

## 2.横向表合并（行对齐）

当axis为默认值0时：

```python
result = pd.concat([df1, df2])
```

**横向合并需要将axis设置为1** ：

```python
result = pd.concat([df1, df2], axis=1)
```

- axis=0时，即默认纵向合并时，如果出现重复的行，则会同时体现在结果中
- axis=1时，即横向合并时，如果出现重复的列，则会同时体现在结果中。

## 3.交叉合并

```python
result = pd.concat([df1, df2], axis=1, join='inner')
```



# Python DataFrame 添加行名和列名

这里想要给第一行，也就是[‘a’, ‘b’]

```python
data.index.name = 'index'
```

## numpy.matmul

 原型: `numpy.matmul(a, b, out=None)` 

两个numpy数组的矩阵相乘
(1). 如果两个参数 a , b a,b a,b都是 2 2 2维的，做普通的矩阵相乘。

 numpy.dot(a,b,out=None) 两个array之间的点乘 

对于array对象，*和np.multiply函数代表的是数量积，如果希望使用矩阵的乘法规则，则应该调用np.dot和np.matmul函数。

对于matrix对象，*直接代表了原生的矩阵乘法，而如果特殊情况下需要使用数量积，则应该使用np.multiply函数。

# **np.argsort()**

```sql
np.argsort(a, axis=-1, kind='quicksort', order=None)
```

 函数功能：将a中的元素从小到大排列，提取其在排列前对应的index(索引)输出。 





# 提取日期时间列的年份

```python
#注意：数据类型需为Datetime 类型，不满足需要先转换。
import pandas as pd
import numpy as np
import datetime
df=pd.read_csv("")
df['Time']= pd.to_datetime(df['Time']) 

#1 pandas.Series.dt.year() 和 pandas.Series.dt.month() 方法提取月份和年份
df['Year'] = df1['Time'].dt.year 
df['Month'] = df1['Time'].dt.month 
print(df)

#2 strftime() 方法提取年份和月份
df['Time']= pd.to_datetime(df['Time']) 
df['year'] = df['Time'].dt.strftime('%Y')
df['month'] = df['Time'].dt.strftime('%m')
print(df)


#3 pandas.DatetimeIndex.month与pandas.DatetimeIndex.year提取
df['year'] = pd.DatetimeIndex(df['Time']).year
df['month'] = pd.DatetimeIndex(df['Time']).month


import pandas as pd
dataframe['date'] = pd.to_datetime(dataframe['date'])
dataframe['date'] = dataframe['date'].dt.strftime('%Y-%m-%d')


# 将多列日期字符串转换为日期时间格式
time_columns = ['AwardEffectiveDate', 'AwardExpirationDate']
data[time_columns] = data[time_columns].apply(pd.to_datetime, format='%m/%d/%Y')

# 将日期时间格式转换为年月日格式的字符串
data[time_columns] = data[time_columns].apply(lambda x: x.dt.strftime('%Y-%m-%d'))
```

# 获取dataframe列名

 **通过columns字段获取，返回一个numpy型的array** 

```
df.columns.values
```

```
df.columns.tolist()
```

```
s=pd.Series(df.columns.values)
new_columns = pd.DataFrame(s.str.split('-').tolist())[0].tolist()
s.str.split('-')拆分列名
s.str.split('-').tolist()转为列表
pd.DataFrame(s.str.split('-').tolist())[0]转为dataframe取第一列
pd.DataFrame(s.str.split('-').tolist())[0].tolist()转为新列名列表
```

### 修改单个列名

```
data=data.rename(columns={'name':'id'})
```





获取文件列表

```
import os
def list_dir(file_dir,list_csv = []):
    '''
    # 递归获取.*,.csv文件存入到list_csv
    :param file_dir: 文件路径
    :param list_csv: 存放路径列表
    :return:list_csv
    '''
    dir_list = os.listdir(file_dir)
    for cur_file in dir_list:#cur_file文件名
        path = os.path.join(file_dir, cur_file)
        # 判断是文件夹还是文件
        if os.path.isfile(path):
            # print("{0} : is file!".format(cur_file))
            dir_files = os.path.join(file_dir, cur_file)
        # 判断是否存在.csv文件，如果存在则获取路径信息写入到list_csv列表中
        # if os.path.splitext(path)[1] == '.csv':
        if os.path.splitext(path)[1] == '.xls':
            csv_file = os.path.join(file_dir, cur_file)
            # print(os.path.join(file_dir, cur_file))
            # print(cur_file)
            list_csv.append(csv_file)
        if os.path.isdir(path):
            # print("{0} : is dir".format(cur_file))
            # print(os.path.join(file_dir, cur_file))
            list_dir(path)
    return list_csv
```

解决dataframe 列拆分最后一位为空问题

```\
data = data.fillna('Nan;')
sss=data['Author'].apply(lambda x: x.split(';')[0:-1])
```

# pandas 计数函数value_counts()

完整版函数

```python
value_counts(normalize=False, sort=True, ascending=False, bins=None, dropna=True)
```

参数:

```python
1.normalize : boolean, default False　默认false，如为true，则以百分比的形式显示

2.sort : boolean, default True　默认为true,会对结果进行排序

3.ascending : boolean, default False　默认降序排序

4.bins : integer, 格式(bins=1),意义不是执行计算，而是把它们分成半开放的数据集合，只适用于数字数据

5.dropna : boolean, default True　默认删除na值
```



### np.linalg.norm()用于求范数，linalg本意为linear(线性) + algebra(代数)，norm则表示范数。

用法

    np.linalg.norm(x, ord=None, axis=None, keepdims=False)

 1.x: 表示矩阵(一维数据也是可以的~)
 2.ord: 表示范数类型 

 **矩阵的向量**：
 ord=1：表示求列和的最大值
 ord=2：|λE-ATA|=0，求特征值，然后求最大特征值得算术平方根
 ord=∞：表示求行和的最大值
 ord=None：表示求整体的矩阵元素平方和，再开根号
 3.axis： 

| 参数 | 含义                                       |
| ---- | ------------------------------------------ |
| 0    | 表示按列向量来进行处理，求多个列向量的范数 |
| 1    | 表示按行向量来进行处理，求多个行向量的范数 |
| None | 表示整个矩阵的范数                         |

 4.keepdims：表示是否保持矩阵的二位特性，True表示保持，False表示不保持，默认为False 

# np.transpose函数

 anspose函数主要用来转换[矩阵](https://so.csdn.net/so/search?q=矩阵&spm=1001.2101.3001.7020)的维度。 



# 数据编码

数值型数据

- 自定义函数 + 循环遍历

- 自定义函数 + map

- 自定义函数 + apply

  ```
  df3 = df.copy()
  df3["Score_Label"] = df3["Score"].apply(lambda x: "A" if x > 90 else (
      "B" if 90 > x >= 80 else ("C" if 80 > x >= 70 else ("D" if 70 > x >= 60 else "E"))))
  ```

  

- 使用 pd.cut

  ```
  df4 = df.copy()
  bins = [0, 59, 70, 80, 100]
  df4["Score_Label"] = pd.cut(df4["Score"], bins)
  
  df4["Score_Label_new"] = pd.cut(df4["Score"], bins, labels=[
                                  "low", "middle", "good", "perfect"])
  ```

  使用 sklearn 二值化

  ```
  df5 = df.copy()
  binerize = Binarizer(threshold = 60)
  trans = binerize.fit_transform(np.array(df1["Score"]).reshape(-1,1))
  df5["Score_Label"] = trans
  ```
  
  

文本型数据

- 使用 replace

  ```
  df6 = df.copy()
  df6["Sex_Label"] = df6["Sex"].replace(["Male","Female"],[0,1]) 
  
  df6 = df.copy()
  value = df6["Course Name"].value_counts()
  value_map = dict((v, i) for i,v in enumerate(value.index))
  df6["Course Name_Label"] = df6.replace({"Course Name":value_map})["Course Name"]
  ```

- ### 使用map

  ```
  
  df7 = df.copy()
  Map = {elem:index for index,elem in enumerate(set(df["Course Name"]))}
  df7["Course Name_Label"] = df7["Course Name"].map(Map)
  
  ②
  name_dict = node.set_index('id')['name'].to_dict()
  depname_dict = node.set_index('id')['depname'].to_dict().fillna(1)空白填充
  weight_dict = node.set_index('id')['weight'].to_dict()
  
  data['Organ_Label'] = data['depname'].map(Map)映射key元数据，valuse 映射的
  ```
  
  #### 使用astype
  
  ```
  df8 = df.copy()
  value = df8["Course Name"].astype("category")
  df8["Course Name_Label"] = value.cat.codes
  ```
  
  #### 使用 sklearn   LabelEncoder
  
  ```
  from sklearn.preprocessing import LabelEncoder
  df9 = df.copy()
  le = LabelEncoder()
  le.fit(df9["Sex"])
  df9["Sex_Label"] = le.transform(df9["Sex"])
  le.fit(df9["Course Name"])
  df9["Course Name_Label"] = le.transform(df9["Course Name"])
  
  
  from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
  df9 = df.copy()
  le = OrdinalEncoder()
  le.fit(df9[["Sex","Course Name"]])
  df9[["Sex_Label","Course Name_Label"]] = le.transform(df9[["Sex","Course Name"]])
  ```
  
  ### 使用factorize
  
  ```
  edge['target_number']=pd.factorize(edge["target"])[0].astype(int)
  edge['index']=pd.factorize(edge["target"])[1].astype(int)
  
  df10 = df.copy()
  cat_columns = df10.select_dtypes(["object"]).columns
  
  df10[["Sex_Label", "Course Name_Label"]] = df10[cat_columns].apply(
      lambda x: pd.factorize(x)[0])
  
  ```

## 判断数据类型

```
if isinstance (num, int):
elif isinstance (num, float):
```

# 异常

```
import pandas as pd

dates=range(20161010,20161114)
pieces=[]
for date in dates:
    try:
        data=pd.read_csv('A_stock/overview-push-%d/stock overview.csv' %date, encoding='gbk')
        pieces.append(data)
    except Exception as e:
        pass
    continue
data=pd.concat(pieces)


try:
	##'有可能出现异常的代码放在这里'
except:
	##'当try中的代码出错时，执行这里的代码，代码写在这里'
try:
	##'有可能出现异常的代码放在这里'
except:
	pass
	continue
##在这里写继续执行新的语句
```

# Python对字典进行排序并返回字典

```python
import operator
dic_instance = {3: 1, 2: 23, 1: 17}
sort_key_dic_instance = dict(sorted(dic_instance.items(), key=operator.itemgetter(0)))  #按照key值升序
sort_val_dic_instance = dict(sorted(dic_instance.items(), key=operator.itemgetter(1)))  #按照value值升序
print(sort_key_dic_instance)  # output:{1: 17, 2: 23, 3: 1}
print(sort_val_dic_instance)  # output:{3: 1, 1: 17, 2: 23}
```

# 将DataFrame数据分割为多行 一行转多行

```python
对authors列，按照;进行拆分
column_authors = data['authors'].str.split(';', expand=True)
0,贾万祥,张平华,None
1,张国宏,None,None
2,韦宁,吕俊虎,李林
3,祁志伟,桑川,None
4,徐鹿,田斯文,None

使用stack行转列
column_authors = column_authors.stack()
0     0    贾万祥
      1    张平华
1     0    张国宏
2     0     韦宁

重置索引（删除多余的索引）并命名为df_name1
column_authors = column_authors.reset_index(level=1, drop=True, name='authors').rename('df_name1')
0       贾万祥
0       张平华
1       张国宏
2        韦宁
2       吕俊虎

使用join合并数据
df_new = data.drop(['authors'], axis=1).join(column_authors)
0   基于多特征识别的非线性网络流量异常检测方法  ...      贾万祥
0   基于多特征识别的非线性网络流量异常检测方法  ...      张平华
1  大数据背景下的公民个人网络信息与安全问题探讨  ...      张国宏
2     基于网络综合扫描的信息安全风险评估研究  ...       韦宁
2     基于网络综合扫描的信息安全风险评估研究  ...      吕俊虎



df_new = data.drop(['authors'], axis=1).join(data['authors'].str.split(';', expand=True).stack().reset_index(level=1, drop=True).rename('df_name1'))


.stack().str.strip()
```



```
df=df.drop('cont', axis=1).join(df['cont'].str.split('/', expand=True).stack().reset_index(level=1, drop=True).rename('tag'))

df=df['cont'].str.split('/', expand=True).stack().reset_index(level=0).set_index('level_0').rename(columns={0:'tag'}).join(df.drop('cont', axis=1))
cont 拆分列  tag新列名
```

## 分割成多列

```
df['City1'] = df['City'].map(lambda x:x. split(']') [0])
df['City2'] =df['City'].map(lambda x:x. split('l')[1])
```





```
product_field[['column1', 'column2', 'column3']] = product_field['v'].str.split('-', 2, expand = True)
```

# python 使用jieba.analyse提取句子级的关键字

jieba.analyse.extract_tags(sentence, topK=5, withWeight=True, allowPOS=())
参数说明 ：
sentence 需要提取的字符串，必须是str类型，不能是list
topK 提取前多少个关键字
withWeight 是否返回每个关键词的权重
allowPOS是允许的提取的词性，默认为allowPOS='ns', 'n', 'vn', 'v'，提取地名、名词、动名词、动词

```python
column_authors = data['authors'].str.split(';', expand=True)
column_authors = column_authors.stack()
column_authors = column_authors.reset_index(level=1, drop=True, name='authors').rename('name')
column_co = data['authors_aogans'].str.split('|', expand=True)
column_co = column_co.stack()
column_co = column_co.reset_index(level=1, drop=True, name='authors_aogans').rename('co')
print(column_co)
df_new = data.join(column_co)
df_new[['author', 'organ']] = df_new['co'].str.split('-',2, expand = True)
print(df_new.head())


df_new = data.drop(['authors'], axis=1).join(data['authors'].str.split(';', expand=True).stack().reset_index(level=1, drop=True).rename('df_name1'))
df_new = data.join(data['authors_aogans'].str.split(';', expand=True).stack().reset_index(level=1, drop=True).rename('co'))
```



# 将列表转为字符串

```
1、使用for循环
testlist = ['h','e','l','l','o']
teststr =''
for i in testlist:
    teststr += i 
print(teststr)
2、join方法：
testlist = ['h','e','l','l','o']
teststr = "".join(testlist)
print(teststr)
```

# 一、去掉二级列表中的括号

```
list_1 = [[1,2,3],[4,5],[6]]
list_2 = [int(x) for item in list_1 for x in item]
print(list_2)[1, 2, 3, 4, 5, 6]

list_0 = [1, 2, 3, 4, 5, 6]
list_3 = ','.join(str(i) for i in list_0)
list_4 = ' '.join(str(i) for i in list_0)
二、去掉外面的括号
print(list_3) 1,2,3,4,5,6
print(list_4) 1 2 3 4 5 6
三、去掉列表中的双引号
list_a = ['CAU',"CBU","CCU"]
list_b = ','.join(str(i) for i in list_a)
print(list_b)  CAU,CBU,CCU
```

### 对一列迭代生成新一列

```
data['words'] = data.abstract_of_keywords.apply(lambda x: extract_word(x)) 

```

### 分组求和

```
new_series = df.groupby(by=['a'])['b'].sum()
 
# 意思是对字段a进行分组然后通过字段B进行求和汇总
# 返回Series类型对象。 a会变成index b则成为值

2、使用聚合函数agg
DataFrame.agg（func，axis = 0，* args，** kwargs ）
参数func采用字典形式：{‘行名/列名’：‘函数名’}，其使用指定轴上的一个或多个操作进行聚合。

df1 = df.groupby(['Fruits']).agg({"Numbers":"sum"})
```

## 统计嵌入节点

```python
data1 = data.groupby("title").apply(lambda x :  str( x['node_id'].values ).replace('[','').replace(']',''))#''.join( str(x['node_id']) )
data2 = pd.DataFrame(data1)

data2.to_csv('dict.csv')

data = pd.read_csv('dict.csv')
```

### 筛选行

获得a列中值为1的行

```
data[data['a'].isin([1])]
```

获得a列中值为1或2的行

```
data[data['a'].isin([1,2])]
```

获得a列中值大于1、小于2的行

```
data[(data['a']<2)&(data['a']>1)]
```

同时，Pandas也提供了query()方法来对DataFrame进行过滤。

DataFrame.query(expr, inplace=False, **kwargs)

参数
expr：引用字符串形式的表达式以过滤数据。
inplace：如果该值为True, 它将在原始DataFrame中进行更改。

获得a列中值为1的行
data.query('a==1')

获得a列中值为1或2的行
data.query('a==1 | a==2')

获得a列中值大于1、小于2的行
data.query('2>a>1')





## 空值处理

```
、pandas中缺失值注意事项
pandas和numpy中任意两个缺失值不相等（np.nan \!= np.nan）
pandas读取文件时那些值被视为缺失值

2、pandas缺失值操作
pandas.DataFrame中判断那些值是缺失值：isna方法
pandas.DataFrame中删除包含缺失值的行：dropna(axis=0) 
pandas.DataFrame中删除包含缺失值的列：dropna(axis=1)
pandas.DataFrame中删除包含缺失值的列和行：dropna(how='any')
pandas.DataFrame中删除全是缺失值的行：dropna(axis=0,how='all')
pandas.DataFrame中删除全是缺失值的列：dropna(axis=1,how='all')
pandas.DataFrame中使用某个值填充缺失值：fillna(某个值)
pandas.DataFrame中使用前一列的值填充缺失值：fillna(axis=1,method='ffill')
pandas.DataFrame中使用前一行的值填充缺失值：fillna(axis=0,method='ffill')
pandas.DataFrame中使用字典传值填充指定列的缺失值  

、pandas中缺失值注意事项
pandas和numpy中任意两个缺失值不相等（np.nan \!= np.nan）
pandas读取文件时那些值被视为缺失值

2、pandas缺失值操作
pandas.DataFrame中判断那些值是缺失值：isna方法
pandas.DataFrame中删除包含缺失值的行：dropna(axis=0) 
pandas.DataFrame中删除包含缺失值的列：dropna(axis=1)
pandas.DataFrame中删除包含缺失值的列和行：dropna(how='any')
pandas.DataFrame中删除全是缺失值的行：dropna(axis=0,how='all')
pandas.DataFrame中删除全是缺失值的列：dropna(axis=1,how='all')
pandas.DataFrame中使用某个值填充缺失值：fillna(某个值)
pandas.DataFrame中使用前一列的值填充缺失值：fillna(axis=1,method='ffill')
pandas.DataFrame中使用前一行的值填充缺失值：fillna(axis=0,method='ffill')
pandas.DataFrame中使用字典传值填充指定列的缺失值  
```



删除列

```
删除名为col的列，使用df.drop(labels='col',axis=1)
删除第2到第4列，使用df.drop(labels=df.columns[2:5],axis=1)

多行多列
df.drop(index=['Bob', 'Dave', 'Frank'],columns=['state', 'point']
```

```
# 修改索引名称为"id"
df = df.rename_axis('id')
```

## 指定数据类型

```
import pandas as pd

# 读取数据
data = pd.read_csv('your_file.csv', dtype={30: str})#指定列
pd.read_csv('your_file.csv',dtype={"user_id":int, "username": object})#按照列名
```

### 空值统计

```
data.isnull().sum()

data.isnull().sum()/data.shape[0]
```

### 字典展开

CSV文件中的"Institution"列包含了字典类型的数据，你想将这个字典展开，使得每个键都成为新的一列

```
# 将"Institution"列的字典展开为新的数据框
df_institution = df['Institution'].apply(json.loads).apply(pd.Series)

# 将原始数据框和展开的数据框合并
df = pd.concat([df.drop(['Institution'], axis=1), df_institution], axis=1)
```

### 多列多个关键字筛选

```python
safe_list = [
            'Security',
            'Security',
            'Secure',
            'Safety',
            'Protection',
            'Defense',
            'Guard',
            'Risk',
            'Threat',
            'Vulnerability',
            'Firewall' ,
            'Encryption' ,
            'Authentication' ,
            'Authorization' ,
            'Surveillance' ,
            'Incident' ,
            'Breach',
            'Intrusion' ,
            'Cybersecurity',
            'Privacy' ,
            'Audit'
            ]
safedata2022 = data2022.loc[data2022['AwardTitle'].str.contains('|'.join(safe_list), case=False)  |  data2022['AbstractNarration'].str.contains('|'.join(safe_list), case=False)]
safedata2022.to_csv('2022年安全.csv', index=False)
```

### 保存控制台信息

```
import sys
import os
import time


# 控制台输出记录到文件
class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


if __name__ == '__main__':
    # 自定义目录存放日志文件
    log_path = './Logs/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # 日志文件名按照程序运行时间设置
    log_file_name = log_path + 'log-' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.log'
    # 记录正常的 print 信息
    sys.stdout = Logger(log_file_name)
    # 记录 traceback 异常信息
    sys.stderr = Logger(log_file_name)

   

```

### 字典保存

```
import pandas as pd
import pickle

# 读取Excel文件
data = pd.read_excel('2022JCR.xlsx')

# 假设你想要转换的列是'Column1'和'Column2'
dict_data = data.set_index('Column1')['Column2'].to_dict()

# 保存字典到pickle文件
with open('dict_data.pickle', 'wb') as handle:
    pickle.dump(dict_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

# 读取pickle文件
with open('dict_data.pickle', 'rb') as handle:
    loaded_data = pickle.load(handle)

print(loaded_data)

```

