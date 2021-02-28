---
title: '[CS231n]六·Neural Networks Part 2'
date: 2021-02-25 19:43:26
tags: 学习笔记
categories: [计算机视觉]
cover: /gallery/pexels_woman-book.jpg
description: Stanford CS231n Convolutional Neural Networks for Visual Recognition
---

[CS231n Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/)

https://www.bilibili.com/video/BV1nJ411z7fe

<!--more-->

# Data Preprocessing

- **Mean subtraction**：将所有数据减去平均值，即中心化。

- **Normalization**：将数据的各维度缩放到大致相同的规模。一般有两种方式：一是对中心化后的数据除以各维度的标准差，使得各维度的数据均值为 $0$，方差为 $1$；二是将最大值和最小值缩放为 $1$ 和 $-1$，这样所有数据都在 $[-1,1]$ 之中。

  <img src="data.png" width="80%" height="80%" />

- **PCA**：主成分分析 (PCA) 是一种常用的数据降维方法，具体内容可见[机器学习笔记](https://xyfjason.top/2021/01/26/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%85%A5%E9%97%A8-%E5%8D%81%E4%BA%8C%C2%B7%E4%B8%BB%E6%88%90%E5%88%86%E5%88%86%E6%9E%90/)。

- **Whitening**：假设数据是一个多元高斯分布，则 whitening 将数据变为均值为 $0$，协方差矩阵为单位矩阵的多元高斯分布。就具体实现而言，在主成分分析中，如果没有使维度减小，则其效果为去相关性 (decorrelate)；对去相关性后的数据除以各维度的特征值就得到了处理后的数据。

  <img src="whiten.png" width="80%" height="80%" />

> 在实际应用中，PCA 和 Whitening 不会在卷积神经网络中用到，但是中心化以及 normalization 很重要。

*在数据预处理时，非常重要的一点是预处理的统计量应该由 traning data 计算而来，然后应用到 validation / test data 上*。**不能够**先在整个数据集上预处理，再划分 train / val / test. 



# Weight Initialization

- **陷阱：all zero initialization**. 全部初始化为 $0$ 是错误的做法，因为这样同一层的所有神经元都没有区别，它们将输出同样的值、具有同样的梯度、进行同样的更新（因为它们是对称的），这是没有意义的。

- **Small random numbers**：初始化为接近 $0$ 的随机数是经常使用的方法。一般而言，可以取 $0.01$ 倍的标准正态分布或者均匀分布等。

- **Calibrating the variances with $1/\sqrt n$**：上一个方法存在一个问题，即神经网络输出的方差将随着输入量的增大而增大。在 [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) 中作者举了一个[例子](http://neuralnetworksanddeeplearning.com/chap3.html#weight_initialization)说明这一点：不妨假设 $1000$ 个输入中有 $500$ 个 $0$ 和 $500$ 个 $1$，考察第一个隐藏层中的一个神经元，$z=\sum_jw_jx_j+b$ 将是 $501$ 个标准正态分布之和，故服从 $N(0,501)$，方差高达 $501$！为了解决这个问题，我们计算：
  $$
  \newcommand{\var}{\text{var}}
  \newcommand{\E}{\mathbb{E}}
\begin{align}
  \var(z)&\approx\var\left(\sum_jw_jx_j\right)&&\text{for simplicity, ignore bias}\\
  &=\sum_{j}\var(w_jx_j)&&\text{independence}\\
  &=\sum_j(\E [w_j])^2\var(x_j)+(\E[x_j])^2\var(w_j)+\var(x_j)\var(w_j)\\
  &=\sum_j\var(x_j)\var(w_j)&&\text{assume }\E(x)=\E(w)=0\\
  &=n\ \var(w)\ \var(x)
\end{align}
  $$
  
  因此，若要 $\var(z)=\var(x)$，需要 $\var(w)=1/n$，故我们只需要将初始化的值除以 $1/\sqrt n$ 即可。实践中这样做能加快收敛。
  
  > 在 [Understanding the difficulty of training deep feedforward neural networks](https://link.zhihu.com/?target=http%3A//jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf) by Glorot et al. 这篇论文中，作者最终建议用 $\var(w)=2/(n_\text{in}+n_\text{out})$ 来初始化；在 [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://link.zhihu.com/?target=http%3A//arxiv-web3.library.cornell.edu/abs/1502.01852) by He et al. 中，作者对 ReLU 神经网络进行分析，得到结论是方差应初始化为 $2/n$. 
  >
  > **实践中，建议采用 ReLU 神经元并将权重方差设为 $2/n$.** 
  
- **Sparse initialization**：另一种初始化方式是将权重矩阵初始化为 $0$，但是为了打破对称性，每个神经元随机地连接下一层若干神经元（连接数目可由一个小的高斯分布生成）。

- **Initializing the biases**：常常将偏置初始化为 $0$. 

- **Batch Normalization**：在全连接层/卷积层与 activation function 之间插入一个 BatchNorm layer，因为 normalization 是可导的操作，这么做是可行的，具体查看论文：[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167). 



# Regularization

- **L2 regularization**：最常见的正则化方法，向 loss function 加上 $\frac{1}{2}\lambda w^2$. 其特点是使得网络倾向于*分散*的权重。

- **L1 regularization**：相对常见的正则化方法，加上 $\lambda |w|$. 其特点是使得网络的权重倾向于稀疏（即许多非常接近 $0$ 的数字，与 L2 正好相反）。结合 L1 和 L2 我们得到 Elastic net regularization，即向 loss function 加上 $\lambda_1|w|+\lambda_2 w^2$. 一般而言，如果不是特别关注某些特征的选择，L2 比 L1 效果会更好。

- **Max norm constraints**：给神经元的权重向量 $w$ 以约束：$||w||_2<c$，$c$ 一般取 $3$ 或 $4$。

- **Dropout**：及其有效和简单的正则化方法，在 [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://link.zhihu.com/?target=http%3A//www.cs.toronto.edu/%7Ersalakhu/papers/srivastava14a.pdf) by Srivastava et al. 中提出，能作为其他正则化方法的补充。实现方法是在训练中让神经元以概率 $p$（一个超参数）激活或设置为 $0$. 

  <img src="dropout.jpeg" width="80%" height="80%" />



我们无需对 bias 进行正则化，因为它们没有与数据直接相乘，并且正则化 bias 对结果的影响也微乎其微。



# Loss functions



## Classification

常用 SVM loss: 
$$
L_i=\sum_{j\neq y_i}\max(0,f_j-f_{y_i}+1)
$$
和 cross-entropy loss: 
$$
L_i=-\ln\left(\cfrac{e^{f_{y_i}}}{\sum_je^{f_j}}\right)
$$


## Attribute classification

不一定被分为某一类，而可以分到若干类，设 $y_{ij}$ 表示第 $i$ 个图像是否属于第 $j$ 类。我们可以对每一类训练一个 logistic regression classifier，然后用负的对数似然作为损失函数：
$$
L_i=-\sum_j y_{ij}\ln\sigma(f_j)+(1-y_{ij})\ln(1-\sigma(f_j))
$$


## Regression

可采取 L2 或 L1 距离作为损失函数，即：
$$
L_i=||f-y_i||_2^2=\sum_j(f_j-(y_{i})_j)^2
$$
或
$$
L_i=||f-y_i||_1=\sum_j|f_j-(y_i)_j|
$$
