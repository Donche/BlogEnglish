---
layout: post
title: Kaggle 入门---Titanic
category: 机器学习
catalog: true
mathjax: true
tags: 
    - 2017
    - Python
    - 机器学习
---

*正如我在上一篇里说过的，Kaggle 是个好地方，对于入门者来说。Titanic 应该属于Kaggle 里最经典的问题，就如同输出Hello World 在学习编程语言过程中的地位。原本还在纠结要不要做，只是最近实习结束休息在家，需找点事情打发时间。在试过几个很有意思的数据集之后又看到了Titanic，想了想以前写hello world的传统都坚持下来了，干脆Titanic 也跟风写一下。到头来，在入门时终究不能免俗，笑*

# 0. 前言

这个数据集的模型在Kaggle 上经过这几年的研究已经相当成熟，很多公开的kernel 都可以达到相当高的准确率，其中不乏一些高手。国内也有很多博客分析如何在Titanic 中达到前百分之十的排名云云。所以我再把这个主题重复一遍，主要就是归纳（？）一下各种分析数据的思想，使之在以后的相似问题中能灵活运用，或者说通过这样一个成熟的数据集学习他人经验，在某种程度上来说也算是一种捷径。

# 1. 问题描述

针对这个数据集的训练集来说，提供了共计891个乘客的经济社会地位（Pclass）、姓名、性别、年龄、配偶与兄弟姐妹数（SibSp）、父母与子女数（parch）、票号（Ticket）、票价（Fare）、舱位（cabin）和登船码头（Embarked），其中有大约177人没有年龄信息，687人没有舱位信息，2人没有登船码头信息，目标是根据这些信息构建模型预测出测试集乘客的生存情况。测试集共有418名乘客信息，其中86人没有年龄信息，1人没有票价信息，327人没有舱位信息。

# 2. Preprocessing

导入数据之后，

# 3. 数据建模

# 4. 优化模型

# 5. 结果与总结