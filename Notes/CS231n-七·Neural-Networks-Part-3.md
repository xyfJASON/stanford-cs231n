---
title: '[CS231n]七·Neural Networks Part 3'
date: 2021-02-27 10:09:32
tags: 学习笔记
categories: [计算机视觉]
cover: /gallery/pexels_woman-book.jpg
description: Stanford CS231n Convolutional Neural Networks for Visual Recognition

---

[CS231n Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/)

https://www.bilibili.com/video/BV1nJ411z7fe

<!--more-->



# Gradient Checks

神经网络反向传播过程很容易写出 bug，为了检验我们计算的梯度是否正确，可以将 analytic gradient 和 numerical gradient 进行比较。这里有一些实现的 tips：

- 计算 numerical gradient 时，使用 centered formula，即：
  $$
  \frac{\mathrm df(x)}{\mathrm dx}\approx\frac{f(x+h)-f(x-h)}{2h}
  $$
  而非 $\frac{\mathrm df(x)}{\mathrm dx}\approx\frac{f(x+h)-f(x)}{h}$，这是因为前者误差为 $O(h^2)$，而后者有 $O(h)$. （泰勒展开即可证）

- 使用相对误差作为比较基准。如果使用绝对误差，比如 $10^{-4}$，那么在梯度本来就小于 $10^{-4}$ 的地方，这个阈值没有什么作用。因此，我们使用相对误差：
  $$
  \frac{|f'_{\text{analytic}}-f'_\text{numerical}|}{\max\left(|f'_\text{analytic}|,|f'_\text{numerical}|\right)}
  $$
  分母用 $\max$ 或者加和或者取任意一个都是可以的，但是要小心除零。

- 注意目标函数的不可导点。跨越了不可导点的 numerical gradient 可能和实际的 gradient 有较大差距，如果误差较大需要检查是不是这个原因。

- 只使用少量数据。解决上一条问题的一个方法是 gradient checks 时只用少量的数据，这样目标函数的不可导处将减少；同时只用检验 $2\sim3$ 处的梯度就够了。

- $h$ 不宜设置得太小。理论上 $h$ 越小越精确，但由于计算机浮点数的精度误差，太小了可能反而不精确。

- 小心正则化项占了梯度的主要部分。如果正则化项在梯度中占比很大，它可能掩盖住了 back propagation 的错误。一个方式是在梯度检验时去掉正则化。

- 记住去掉 dropout/augmentations. 梯度检验时需要去除带有**随机性质**的量，比如 dropout 或者数据的随机 augmentation. 提前设置一个随机数种子也是可以的。



# Sanity Checks

在实施学习之前，可以先进行 sanity checks，以保证我们的代码是正确的。

- 按概率推测损失函数初始值。例如一个十分类问题，随机初始化后我们应期望得到 $1/10$ 的正确率，也即负对数概率的 loss function 大致为：$-\ln(0.1)=2.302$. 
- 过拟合一个极小的数据集。用一个极小的数据集去训练，在去除正则化的情况下，我们应该期望得到 $100\%$ 的正确率或 loss function 为 $0$，即在这个极小的数据集上过拟合。



# Babysitting the learning process

学习过程中我们可以监测一些值随着迭代次数的增加的变化。

- **Loss function**. 作 loss function - epoch 图能直观的告诉我们学习的状况，帮助我们调整学习率等参数。

  <img src="loss.jpeg" width="80%" height="80%" />

- **Train/val accuracy**. 作训练集和验证集上的准确率的图能直观的告诉我们是否过拟合/欠拟合，帮助我们调整正则化项。

  <img src="train val accuracy.jpeg" width="80%" height="80%" />

- **Ratio of weights: updates**. 检测参数更新的幅度，帮助我们判断学习率是否合适，合适的幅度应该在 $10^{-3}$ 左右。

- **First-layer Visualizations**. 在图像处理的网络中，我们可以把第一层的神经元的参数还原成图像可视化。

  <img src="first.jpeg" width="80%" height="80%" />



# Parameter updates



## SGD and bells and whistels

- **Vanilla update**. 朴素的随机梯度下降

- **Momentum update**. 朴素的 SGD 在每一处都向梯度下降的方向更新，就像一个人下山一样。而如果我们做另一个比喻，想象一个小球从山坡滚下，它在每一处将具有保持原方向的惯性，这就是 momentum SGD 的灵感来源。具体地，momentum SGD 更新如下：
  $$
  \begin{align}
  v&:=\mu v-\alpha\ \mathrm dx\\
  x&:=x+v
  \end{align}
  $$
  其中，$\alpha$ 为学习率，$\mu$ 可以看作是摩擦因子。

- **Nesterov Momentum**. 这是另一种 momentum update. 不同于上一条，Nesterov Momentum 计算随“惯性”移动一段距离之后的梯度再更新，见下图：

  <img src="momentum.jpeg" width="80%" height="80%" />

  具体地，Nesterov Momentum 更新如下：
  $$
  \begin{align}
  x_\text{ahead}&:=x+\mu v\\
  v &:= \mu v-\alpha\ \mathrm dx_\text{ahead}\\
  x&:=x+v
  \end{align}
  $$



## Annealing the learning rate

在学习过程中不断减小 learning rate 往往有帮助，一下三种方式是常用的 learning rate decay 实现方式：

- **Step decay**. 每迭代几次后将 learning rate 减小，例如每 $20$ 次迭代后减小到原来的 $0.1$，或者每 $5$ 次迭代减小一半等。这些参数依赖于具体问题和模型。一种启发式方法是当 validation error 停止减小时就减小 learning rate. 
- **Exponential decay**. 依 $\alpha=\alpha_0e^{-kt}$ 减小，其中 $\alpha_0,k$ 是超参数，$t$ 是迭代次数。
- $1/t$ **decay**. 依 $\alpha=\alpha_0/(1+kt)$ 减小。

实践中，Step decay 稍稍更受喜爱一些，因为其解释性比其他的强。



## Second order methods

梯度下降及其延伸算法只需要用到一阶导数，即梯度。还有些最优化算法需要用到二阶导数。

- **Newton's method**. 
  $$
  x:=x-[Hf(x)]^{-1}\nabla f(x)
  $$
  其中，$Hf(x)$ 表示 Hessian matrix，$\nabla f(x)$ 表示 $f(x)$ 的梯度向量。

  注意到上述更新中不含有学习率这一超参数，这也是其优于一阶方法所在。

  <br>

  但是，Newton's method 有一个及其显著的缺点，即 Hessian matrix 过于庞大，$n$ 个参数的 Hessian matrix 是 $n\times n$ 的，计算耗时且难以存储。因此，出现了众多 *quasi-Newton* methods，例如 $\text{L-BFGS}$，它们不需要将 Hessian matrix 完整计算出来. 



## Per-parameter adaptive learning rate methods

在之前的算法中，学习率是全局的且对所有参数都一样。调整学习率是一个耗时耗力的过程，因此人们发明了许多自动调整学习率的方法，且这些学习率甚至能对每一个参数进行调整。

虽然这些方法也有超参数，但它们与纯粹的学习率相比，能在更大的范围内表现较好，因而更好调参。

- **Adagrad**：
  $$
  \begin{align}
  \text{cache}&:=\text{cache}+(\mathrm dx)^2\\
  x&:=x-\alpha\ \frac{\mathrm dx}{\sqrt{\text{cache}}+\text{eps}}
  \end{align}
  $$
  注意，上式中对向量的运算符定义为与 `numpy` 相同。

  对上式的理解是：$\text{cache}$ 一路记录了每个参数的梯度平方和，如果某个参数有较大的梯度，那么它的学习率将减小；相反，梯度较小的参数将有较大的学习率。

- **RMSprop**：RMSprop 在 Adagrad 的基础上进行了简单的修改：
  $$
  \begin{align}
  \text{cache}&:=\text{decay rate}\times \text{cache}+(1-\text{decay rate})\times (\mathrm dx)^2\\
  x&:=x-\alpha\ \frac{\mathrm dx}{\sqrt{\text{cache}}+\text{eps}}
  \end{align}
  $$
  也即是增加了 $\text{decay rate}$ 这一超参数，一般取值为 $0.9,0.99,0.999$. 

- **Adam**：可以看作是 RMSprop 带 momentum 的版本，其（简化的）更新如下：
  $$
  \begin{align}
  m&:=\beta_1m+(1-\beta_1)\mathrm dx\\
  v&:=\beta_2v+(1-\beta_2)(\mathrm dx)^2\\
  x&:=x-\alpha\ \frac{m}{\sqrt{v}+\text{eps}}
  \end{align}
  $$
  典型的参数是：$\text{eps}=10^{-8},\beta_1=0.9,\beta_2=0.999$. 

  完整的 Adam 算法存在一个 *bias correction* 机制：
  $$
  \begin{align}
  m&:=\beta_1m+(1-\beta_1)\mathrm dx\\
  \color{purple}{mt}&\color{purple}{:={m}/({1-\beta_1^t})}\\
  v&:=\beta_2v+(1-\beta_2)(\mathrm dx)^2\\
  \color{purple}{vt}&\color{purple}{:=v/(1-\beta_2^t)}\\
  x&:=x-\alpha\ \frac{\color{purple}{mt}}{\sqrt{\color{purple}{vt}}+\text{eps}}
  \end{align}
  $$



# Hyperparameter optimization

超参数的优化是一个黑盒优化问题，其优化的目标函数是神经网络的性能，因而有一个显著的特点：得到每一组超参数的结果会花费巨额的时间。因此我们并不能直接套用一般的优化方法对超参数进行优化。一般的，超参数优化方法有以下几种：

- **Grid search**：即每个超参数设置几个值，暴力遍历所有的可能。

- **Random search**：在论文 [Random Search for Hyper-Parameter Optimization](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf) by Bergstra and Bengio 中，作者论证了随机搜索比暴力遍历更为优秀，且更容易实现。

  <img src="grid.jpeg" width="80%" height="80%" />

- **Bayesian Hyperparameter Optimation**：利用贝叶斯方法进行超参数的寻找，核心是 exploration - exploitation trade-off. 



# Model Ensembles

实践中，一个提升神经网络性能的方法是训练多个独立的模型，使用它们的平均预测结果。这些独立的模型可以是不同的模型，同一种模型、不同初始化条件，甚至是同一个模型在训练的不同时间点处等。除了对模型结果进行平均，还可以对各模型的参数进行平均，有时也能提高神经网络性能。



