---
title: '[CS231n]五·Neural Networks Part 1'
date: 2021-02-24 19:35:10
tags: 学习笔记
categories: [计算机视觉]
cover: /gallery/pexels_woman-book.jpg
description: Stanford CS231n Convolutional Neural Networks for Visual Recognition
---

[CS231n Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/)

https://www.bilibili.com/video/BV1nJ411z7fe

>  注：CS231n notes 的第三章 Optimization 和第四章 Backpropagation, Intuition 分别讲了梯度下降（$\text{SGD},\text{mini-batch GD}$）和反向传播的直观理解（感觉内容就是链式法则）。内容都比较简单和浅显，故不再记录于此。

<!--more-->

# 神经元模型

<img src="neuron.png" width="80%" height="80%" />

即取输入的线性组合（加上偏置项），再对其取 **activation function** $f$ 之后输出。



## 常用 activation function

- **Sigmoid**：
  $$
  \sigma(x)=\frac{1}{1+e^{-x}}
  $$
  Sigmoid 函数曾经很常用，但最近很少用了，这是因为它有两个大弊端：

  - Sigmoids 会 saturate，导致**梯度消失**。由于 $\sigma(x)$ 在 $|x|$ 较大时梯度非常小，而神经网络反向传播的本质是链式求导法则，要把梯度一级一级地乘起来，所以一旦遇到梯度很小的地方，梯度就无法继续反向传播了，这将导致前层的神经元的参数几乎不怎么更新。
  - Sigmoid 的输出不是以 $0$ 为中心的，这意味着后一层神经元接受的输入都是大于 $0$ 的数，从而导致参数 $w$ 的梯度全为正或者全为负。

- **Tanh**：
  $$
  \tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}
  $$
  其实 $\tanh(x)$ 仅仅是 $\sigma(x)$ 放缩和移动后的结果：
  $$
  \tanh(x)=2\sigma(2x)-1
  $$
  所以依然存在梯度消失的问题，但是不同于 sigmoid 的是它的值域为 $(-1,1)$，以 $0$ 为中心。因此实践中，$\tanh(x)$ 往往比 sigmoid 更被大家喜爱。

- **ReLU**：Rectified Linear Unit. 
  $$
  f(x)=\max(0,x)
  $$

  - (+) 与 sigmoid 或 tanh 相比，ReLU 大大加速了 $\text{SGD}$ 的收敛速度；
  - (+) ReLU 的计算非常简单（取 $\max$ 即可），而 sigmoid 和 tanh 涉及到指数等比较耗时的计算；
  - (-) 使用 ReLU 容易导致神经元 "die". 如果学习率设置的过大，容易使得参数变化过大，导致对数据集中的所有点 $wx+b$ 都小于 $0$，于是梯度不可逆地变成 $0$. 

- **Leaky ReLU**：
  $$
  f(x)=[x<0](\alpha x)+[x\geqslant 0]x
  $$
  其中，$\alpha$ 是一个较小的常数。这么做是为了解决 ReLU 中神经元 die 的问题。在 PReLU 中，$\alpha$ 被视作神经元的一个参数。

- **Maxout**：
  $$
  f(x)=\max(w_1x+b_1,w_2x+b_2)
  $$
  ReLU / Leaky ReLU 是 Maxout 的特殊情况。



## 一个神经元作为 linear classifier

选取合适的损失函数后，一个神经元可以用来实现一个 linear classifier. 

- **Binary Softmax classifier**：视神经元的输出 $\sigma\left(\sum_iw_ix_i+b\right)$ 为分类为正类的概率，采取 cross-entropy loss 训练，则我们得到了 binary Softmax classifier，也即 **logistic regression**. 
- **Binary SVM classifier**：使用 max-margin hinge loss 训练，则我们得到 binary Support Vector Machine. 



# 神经网络结构

一个 input layer，一个 output layer，若干 hidden layer. 对于最普通的神经网络而言，层与层之间全连接。

<img src="neural network.png" width="80%" height="80%" />

输出层的神经元经常没有 activation function，这样输出可以被视作分类的得分或者一些取实值的目标（比如回归值）。

<br>

一个单层的神经网络就可以任意近似任何函数，严格证明在 Approximation by superpositions of a sigmoidal function (1989) 这篇论文中，直观解释可以在博客 [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap4.html) 中找到。虽然如此，实践中多层的神经网络效果更好，也更容易学习。

神经网络中，隐藏层神经元数量越多，其表达能力越强大：

<img src="hidden neurons.png" width="70%" height="70%" />

当然也意味着越容易过拟合，可以采取正则化的方式避免过拟合：

<img src="overfitting.png" width="70%" height="70%" />