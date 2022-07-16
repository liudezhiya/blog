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

