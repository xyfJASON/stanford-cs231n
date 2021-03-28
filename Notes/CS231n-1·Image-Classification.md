---
title: '[CS231n]1·Image Classification'
date: 2021-02-21 20:26:57
tags: 学习笔记
categories: [计算机视觉,CS231n]
cover: /gallery/pexels_woman-book.jpg
description: Stanford CS231n Convolutional Neural Networks for Visual Recognition
---

[CS231n Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/)

https://www.bilibili.com/video/BV1nJ411z7fe

<!--more-->



# Data-Driven Approach

在统计机器学习领域，不同于传统的算法，我们的算法是**数据驱动**的。以图像分类为例，数据驱动方法有以下 $3$ 个步骤：

1. 收集数据并标注；
2. 使用机器学习方法训练一个分类器；
3. 使用该分类器对新的图片进行分类。

实现上来讲，我们的算法至少包含两个函数：`train` 和 `predict`，前者进行机器学习训练，后者使用训练好的模型进行预测。



---



# Nearest Neighbor

最简单的分类器莫过于**最近邻**了。仍然以图像分类为例，最近邻的 `train` 过程仅仅是记录下所有数据及其标签，`predict` 过程是选取数据集中与当前图片最相近的图片，并以其标签作为结果输出。

尽管最近邻很简单，但是仍然有一个问题需要解决：怎么评判两张图片的相似程度。由于图片可以看作像素点组成的三个向量（$\text{RGB}$ 各一个），所以问题转化为两个向量相似程度的评判。常用的有：

- $\text{L1 distance}$：
  $$
  d_1(I_1,I_2)=\sum_p|I_1^p-I_2^p|
  $$
  这里 $I_1,I_2$ 是要比较的两个向量，上标 $p$ 表示向量的第 $p$ 个元素。该距离也称作曼哈顿距离。

- $\text{L2 distance}$：
  $$
  d_2(I_1,I_2)=\sqrt{\sum_p(I_1^p-I_2^p)^2}
  $$
  即欧几里得距离。

显而易见，最近邻算法准确度很差。如果出现一个噪声点，它将对周围一圈产生误导。

<img src="nn.png" height="30%" width="30%" />



---



# K-Nearest Neighbor

为了解决最近邻的问题，我们可以查看 $k$ 个最近的点，并以其中最多的标签作为结果。这就是 $k$ 近邻算法。



## Hyperparameters

在 $k$ 近邻算法中，$k$ 如何取值是我们设定的，而非模型学习出的，这类参数被称作**超参数**。

$k$ 近邻算法的另一个超参数是距离函数的选取，选 $\text{L1 distance}$ 和选 $\text{L2 distance}$ 的结果会不同。

<img src="knn k.png" height="80%" width="80%" />

<img src="nn l1 l2.png" height="70%" width="70%" />

<br>

为了找寻最好的超参数，我们需要将数据集分为 $3$ 类：training set, validation set, test set. 选取一组超参数后，我们在 training set 上训练模型，在 validation set 上测试该模型的效果，调整超参数使得效果达到最佳，最后以在 test set 上的效果作为该模型的真正效果。

<br>

**交叉验证**：在数据集不是很多的时候，我们可以选择 $k$ 折交叉验证的方式代替上述方式。我们分出一部分 test set 后，把剩下的数据集分成 $k$ 份（fold），循环地将每一份都视为 validation set、其他份视为 training set 进行 $k$ 次训练，以 $k$ 个训练效果的平均视为当前超参数下的训练效果，然后对超参数进行调整使得效果达到最佳，同样最后以在 test set 上的效果作为真正效果。

<img src="cross validation.png" height="50%" width="50%" />



## KNN on images never used

$\text{KNN}$ 算法非常简单，但是问题也非常突出：

1. 它基本没有训练过程，仅仅是把数据集记录下来，而在使用阶段需要遍历整个数据集，非常耗时（如果使用 $\text{KD-Tree}$ 实现，随机数据下复杂度是 $\log$ 的，要好一些，但是仍然不满意）；这与我们需要的正好相反——我们愿意花费较多的时间对模型进行训练，但一旦训练好了，在对新数据预测时要很快地出结果。
2. 不同的图像之间可以有相同的 $\text{distance}$，这种测度方式对图像而言不太好。
3. 当向量维度很高时，我们的数据点在高维空间中的分布是稀疏的，所以所谓的“最近点”可能其实并没有多么的相似。

