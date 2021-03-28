---
title: '[CS231n]6·Detection and Segmentation'
date: 2021-03-11 10:59:38
tags: 学习笔记
categories: [计算机视觉,CS231n]
cover: /gallery/pexels_woman-book.jpg
description: Stanford CS231n Convolutional Neural Networks for Visual Recognition
---

[CS231n Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/)

https://www.bilibili.com/video/BV1nJ411z7fe

<!--more-->



# Overview

<img src="cvtasks.png" width="80%" height="80%" />



---



# Semantic Segmentation

对图像的每一个像素进行分类。



## Idea 1: Sliding Window

<img src="slide.png" width="80%" height="80%" />

对每一个像素点用一个神经网络做一次分类。效率很差，进行了许多重复计算，nobody use it. 



## Idea 2: Fully Convolutional

<img src="fullyconv.png" width="80%" height="80%" />

经过一个卷积神经网络，每一个卷积层都**保持原图像的大小**，最终对每个像素都得到其各分类的得分，从而得到每个像素的分类结果。

问题：卷积非常耗时，因为每一个卷积层都保持原图像大小。

<br>

为了解决这个问题，我们可以先通过 $\text{max-pool}$ 或者 $\text{conv}$ 层进行 **downsampling**，然后再 **upsampling** 回原大小：

<img src="fullyconv2.png" width="80%" height="80%" />

我们很清楚如何 downsampling，问题在于如何进行 **upsampling**：

- **Unpooling**：

  <img src="unpooling.png" width="80%" height="80%" />

- **Max-unpooling**：

  <img src="maxunpooling.png" width="80%" height="80%" />

- **Transpose Convolution**：

  视输入矩阵为 filter 的权重，按权重将 filter 累加到输出矩阵中。

  <img src="transconv.png" width="80%" height="80%" />

  **Transpose Convolution** 只能恢复卷积前后的大小，并不能恢复值，其原理是：

  在 Convolution 操作时，我们可以将输入图像拉伸为 $1$ 维向量 $X\in\mathbb R^{D}$，输出图像拉伸为 $1$ 维向量 $Y\in\mathbb R^{D'}$，则卷积核可以写作一个稀疏矩阵 $C\in \mathbb R^{D'\times D}$，使得 $CX=Y$. 在 Upsampling 中，已知 $C,Y$，欲得到 $X$，Transpose Convolution 的操作是：$X=C^TY$，这也是其名称的由来。显然这样做并没有恢复值，除非 $CC^T=I$. 



---



# Object Detection



## Idea 1: Regression

构造神经网络输出检测物体的类别以及框住物体的矩形框的坐标 $(x,y)$ 和大小 $(w,h)$. 

由于在 Object Detection 中，我们并不能实现知道输入图像中有多少个 object，所以其输出大小是不固定的，并且当 object 很多时输出将很大。因此这不是一个很好的方法。



## Idea2: Sliding Window

滑动一个矩形框，每次对矩形框内的图像进行分类。

<img src="obde_sliding.png" width="80%" height="80%" />

该方法也有很明显的问题：矩形框的位置和大小都是未知的，我们需要暴力遍历各种矩形框，每次都通过一个巨大的 CNN 网络得到类别，这无疑十分耗时。



## Idea 3: Region Proposals & R-CNN

- **Region Proposals**: 先通过一个网络给出物体可能存在的若干矩形范围，后续工作只需在这些范围内进行。**Selective Search** 是一种给出 Region Proposals 的方法，其运行速度非常快。

- **R-CNN**：在每一个给定的矩形范围内，我们可以构建 CNN 网络，输出该矩形内物体的类别以及这个矩形区域应该如何修正。注意由于每个矩形大小不同，而我们的网络要求特定大小的输入，所以我们会先对这些矩形进行变形使之符合网络的输入要求。具体见下图：

  <img src="R-CNN.png" width="80%" height="80%" />

  该方法的不足在于，对每个矩形区域进行计算依然十分耗时耗力。

- **Fast R-CNN**：Idea 是我们先把图像输入进 CNN 网络，再对网络的输出结果进行矩形区域划分，这样图像只会经过一次 CNN 网络，从而提高运行速度。

  <img src="Fast R-CNN.png" width="60%" height="60%" />

  这里有个问题是如何在 CNN 网络的输出结果上进行 Region Proposals：

  - **RoI (Region of Intrest) Pool**：首先把原图像上的矩形区域按比例投影到输出矩阵上，然后取整使矩形端点落在整点上，对于这个矩形，我们将其尽可能地等分成需要的形状，并在每一块中作 $\text{max-pool}$，最终得到该区域的 feature. 

    <img src="RoI pool.png" width="100%" height="100%" />

    由于我们在这个过程中进行了两次近似——取整和划分，所以这种方法的问题在于我们得到的区域特征稍有偏离。

  - **RoI Align**：仍然把原图像上的矩形区域投影到输出矩阵上，然后直接等距离划分成需要的形状；现在考察每一个小块，它们的顶点不一定在整点上，我们在其中取 $4$ 个位置，对每个位置用它周围的 $4$ 个整点的值作双线性插值，然后取 $\text{max-pool}$，最终得到该区域的 feature. 

    <img src="RoI align.png" width="100%" height="100%" />

- **Faster R-CNN**：实测中发现，在 Fast R-CNN 中，Region Proposals 占了大部分时间。为了提高这部分的速度，Faster R-CNN 用一个 **Region Proposal Network** 在 CNN 的输出矩阵上作矩形区域划分，其余部分和 Fast R-CNN 相同。

  <img src="Faster R-CNN.png" width="60%" height="60%" />

  Region Proposal Network (RPN) 的原理如下：

  1. 想象以 feature map 的每一个位置为中心有固定大小的 anchor box，使用一个 CNN 预测这些 anchor box 中是否有物体，以及对 box 四边的修正值。
  2. 对于每一个位置，计算 $K$ 个不同的 anchor box，这样我们得到了 $K\times H\times W$ 个数值表示该 anchor 处是否有物体，以及 $4K\times H\times W$ 个数值表示 box 四边的修正值。
  3. 取得分最高的约 $300$ 个 anchor box 作为我们的 region proposals. 

  <img src="RPN.png" width="100%" height="100%" />

  在训练时，我们会将四个 loss 一起训练—— RPN 二分类（判读是否是物体）、RPN anchor box 位置的 loss、最终分类（判断是何物）的 loss、最终 box 的坐标的 loss。

- **YOLO / SSD / RetinaNet**：Faster R-CNN 其实有两个阶段：第一个阶段是得到 region proposals，通过一个基础的 CNN 和 RPN 完成，第二个阶段是对每一个 region 进行分类以及修正边界，通过 RoI pool / align 之后输入到 CNN 完成。而 YOLO / SSD / RetinaNet 等网络只用一个阶段，具体的内容不在此课程中讲授。

<br>

可以看出，Object Detection 有很多选择：首先，基础的网络有很多选择（VGG16 / ResNet-101 / Inception V2……）；接下来，我们可以选择二阶段的 Faster R-CNN 或者单阶段的 YOLO / SSD 或者混合的 R-FCN……这其中就有许多的 trade-offs. 例如，Faster R-CNN 比 SSD 更慢，但是准确率比后者更高；又如网络越大越深，效果越好，但训练也越难，耗时也越多……

这里有一篇综述论文：[Zou et al, “Object Detection in 20 Years: A Survey”, arXiv 2019](https://arxiv.org/pdf/1905.05055v2.pdf)



---



# Instance Segmentation



## Mask R-CNN

<img src="Mask R-CNN.png" width="80%" height="80%" />

类似于之前的 Faster R-CNN，不过最后一个卷积网络不是对区域内容进行分类，而是对每一类输出一个 mask。

Mask R-CNN 还可以做 pose estimation，即预测人体的姿势。

