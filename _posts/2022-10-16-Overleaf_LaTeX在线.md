---
title: Overleaf_LaTeX在线
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



# Overleaf_LaTeX在线

```
\begin{table*}[!htbp]
	\centering
	\begin{tabularx}{\textwidth}{
    >{\centering\arraybackslash\hsize=.5\hsize\linewidth=\hsize}m{1cm}
    >{\centering\arraybackslash\hsize=1.5\hsize\linewidth=\hsize}m{3cm}
    >{\centering\arraybackslash\hsize=.5\hsize\linewidth=\hsize}m{1cm}
    >{\centering\arraybackslash\hsize=1.5\hsize\linewidth=\hsize}m{3cm}
}
		\toprule
		Symbol        & describe                                                  & Symbol           & describe                                                                            \\
		\midrule
		$t$           & Node collection, and $V = \{ 1,2...,N\} $                 & $L_A$            & Number of authors per article                                                       \\
$v$           & Edge set of network snapshot $t$                          & $L_o$            & The number of institutions corresponding to the author                              \\
${E^{(t)}}$   & Adjacency matrix of network snapshot $t$                  & $\rho$           & P3OE network density                                                                \\
${A^{(t)}}$   & Adjacency matrix of network snapshot $t$                  & $\bigtriangleup$ & P3OE Network Triangle                                                               \\
${C^{(t)}}$   & Node label vector of network snapshot $t$                 & $k$              & P3OE network average density                                                        \\
${K_i}$       & Degree of the ith node in the network                     & $N_{i}$          & $i\subset{one,two...}$quantifies the intensity of different collaborative abilities \\
$Independent$ & The strength of the author's independent research ability & $Q$              & Strength of other cooperation capabilities                                           \\
		\bottomrule
	\end{tabularx}%
	\label{tab:addlabel}%
	\caption{A table with line breaks}
\end{table*}%
```

