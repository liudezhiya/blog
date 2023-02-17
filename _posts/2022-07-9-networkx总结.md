---
title: networkx总结
date: 2022-03-26 10:34:00 +0800
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

# 创建Graph

```python
G = nx.Graph()          # 无向图
G = nx.DiGraph()        # 有向图
G = nx.MultiGraph()     # 多重无向图
G = nx.MultiDigraph()   # 多重有向图
G.clear()               # 清空图
```

 Graph 是一组节点（顶点）和已识别的节点对（称为边、链接等）的集合。在NetworkX中，节点可以是任何 hashable 对象，例如文本字符串、图像、XML对象、另一个图形、自定义节点对象等。 

#  给Graph添加边

```python
G.add_edge(1, 2)             # default edge data=1
G.add_edge(2, 3, weight=0.9) # specify edge data
# 如果是边有许多的权，比如有长度和宽度的属性，那么：
G.add_edge(n1, n2, length=2, width=3)
 
elist = [(1, 2), (2, 3), (1, 4), (4, 2)]
G.add_edges_from(elist)
elist = [('a', 'b', 5.0), ('b', 'c', 3.0), ('a', 'c', 1.0), ('c', 'd', 7.3)]
G.add_weighted_edges_from(elist)
 
# 如果给结点的名称是其它符号，想离散化成从x开始的数字标记，那么：
G = nx.convert_node_labels_to_integers(G, first_label=x)

```

查看图属性

```
# 图属性查看：节点、边、相邻（相邻）和节点度数
G.number_of_nodes()
G.number_of_edges()
G.nodes
G.edges
# G.edges.data('weight')  # 返回边和权重信息
G.adj     # 通过字典的键值来表示 节点及其邻接节点, 因此可以直接通过键访问某个节点, 返回其对应的一级邻居及其权重。
# G.adj等价于G.adjacency() 或 G.adj.items(), 查看所有邻居，对于无向图来说，邻接迭代可以看到每个边两次。
# G.adj[1] 和G[1] 效果差不多，都能返回节点1 连线的节点等信息。
# G.edges[1, 2] 和 G[1][2]效果差不多， 返回连接节点1和节点2的边的属性。
G.degree  # 邻居的个数

```



设置节点吗属性 

```
①
nx.set_node_attributes(G, name_dict)


②
for i in sorted(G.nodes()):
    G.nodes[i]['name'] = name_dict[i]
    G.nodes[i]['depname'] = depname_dict[i]
    G.nodes[i]['weight'] = weight_dict[i]
    G.nodes[i]['year'] = year_dict[i]
    G.nodes[i]['Author_length'] = Author_length_dict[i]
```

遍历图节点
选择排名靠前连通子图

```
#极大连通子图
C = sorted(nx.connected_components(G), key=len, reverse=True)
C_max_connect = {}
# data = pd.read_csv('..//anquanCoutput//nationSF.csv')
for i in range(3):
    part = C[i]
    first_node = C[i].pop()
    print("最大连通分量{}: {} 长度为：{}".format(i+1,C[i],len(C[i])))
    print(first_node,G.nodes[first_node])
    data = []
    while part:
        node = part.pop()
        node_data = G.nodes[node]
        Keyword = node_data.get('Keyword')
        data.append(Keyword)
        print(node, node_data)
    print(data)
```

选择读前十节点

```
import operator
degree_coef =  {}
for node in G.nodes():
    # clus_coef[node] = nx.clustering(G,node)
    degree_coef[node] = nx.degree(G,node)
sorted_clus_coef = sorted(degree_coef.items(), key=operator.itemgetter(1),reverse=True)
node_degree_10 =  sorted_clus_coef[10:15]
print('nodes degree range 10:',node_degree_10)
```

判断度前十的节点是不是在最大联通图中、

```python
C = sorted(nx.connected_components(G), key=len, reverse=True)
C_max_connect = C[0:3]
print('C_max_connect:',C_max_connect)
degree10_15_in_max_connect  =  []
for node_degree in  node_degree_10:
    node = node_degree[0]
    for connect in C_max_connect :
        if node in connect:
            degree10_15_in_max_connect.append(node)
print('degree10_15_in_max_connect:{},个数：{}'.format(degree10_15_in_max_connect,len(degree10_15_in_max_connect)))
if  degree10_15_in_max_connect != []:
    for node in degree10_15_in_max_connect:
        print(node,G.nodes[node].get('name'),G.nodes[node].get('depname'))
print('='*50)
```

独立科研能力

```
df_degree_weight = []
for node in G.nodes():
    line = [node,G.nodes[node].get('name'),G.nodes[node].get('depname'),G.degree(node),G.nodes[node].get('weight')]
    df_degree_weight.append(line)
    # print('节点id:{} 度：{} 节点属性权重：{}'.format(node,G.degree(node),G.nodes[node].get('weight')))
df_degree_weight = pd.DataFrame(df_degree_weight)
df_degree_weight.columns=['id','name','depname','degree','weight']
df_degree_weight['Independent_research'] = df_degree_weight['weight']  /  df_degree_weight['degree']
df_degree_weight = df_degree_weight.sort_values(by=['Independent_research'],ascending = False)
print(df_degree_weight.head(10))
df_degree_weight.to_csv('..//anquanCoutput//degree_weight_independent.csv',index=None)

print(G.nodes.data())#这里data()里面可以加单个 节点属性，本质是字典，选择对应的值 返回节点id 和选择属性的元组构成的列表
```

