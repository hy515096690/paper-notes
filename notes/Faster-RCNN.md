<div align=center>Faster-RCNN</div>



目录



- 一、网络模型

  ![img](https://pic4.zhimg.com/80/v2-e64a99b38f411c337f538eb5f093bdf3_720w.webp)

  - 1）首先对图像预处理，调整到规定尺度，然后送入cnn网络。
  - 2）CNN包含13个卷积层+13个relu激活+4个pooling池化。
  - 3）将CNN层提取到的feature map输入到RPN网络，先对其3x3卷积，分别生成锚点（positive anchors）和对应的边界回归偏移量（bounding box regression），然后计算出预测边框（proposals）。
  - 4）ROI Pooling层对proposals与feature map中提取到的proposal feature送入fcn与softmax进行object分类和边框回归。

- 二、CNN层

  ​		Faster R-CNN中对所有卷积都做了扩边处理（pad=1，做0填充），变成（M+2）X（N+2）后再做3X3卷积输出MxN，使Conv layers中的conv层不改变输入和输出矩阵大小。

<img src="https://pic2.zhimg.com/80/v2-3c772e9ed555eb86a97ef9c08bf563c9_720w.webp" alt="img" style="zoom:67%;" />

conv与relu不改变大小，只有pooling层使输出缩小1/2（kernel_size = 2, stride = 2）。

- 三、Region Proposal Networks (RPN)

  ​		Faster R-CNN抛弃之前的候选框滑动算法，输入大小为（M/16）x（N/16）的特征图，通过两个分支，==***一个通过softmax分类anchors获得positive和negative分类，另一条计算anchors的bounding box的偏移量，以获得精确的proposal。最后proposal层综合positive anchors和对应的bounding box regression偏移量修正后的proposals，同时舍掉尺寸不合适的proposals。***==

  <img src="https://img-blog.csdnimg.cn/7e6b844a71c04855ad060750213b2b16.png?" alt="img" style="zoom: 50%;" />

- 3.1 anchors

  设置9个预设anchor，大小分别为[1:1，1:2，2:1]。

  <img src="https://raw.githubusercontent.com/hy515096690/paper-notes/main/img/202211201344822.png" alt="image-20221120134425758" style="zoom:50%;" />

- 3.2 cls layer 分类

  <img src="https://raw.githubusercontent.com/hy515096690/paper-notes/main/img/202211201343424.png" alt="image-20221120134311308" style="zoom:50%;" />

​		一个anchor是一个feature map，对其进行3x3卷积（256个核），k为anchor的个数。(M/16)x(N/16)x256的特征通过1x1卷积得到(M/16)x(N/16)x2k的输出，因为这里是二分类判断positive和negative，所以该feature map上每个点的每个anchor对应2个值，表示目标和背景的概率（是因为这里是用的softmax，所以结果分类为2，这两个值加起来等于1；也可以用sigmoid，就只需要1个值了）。

- 3.3 reg layer 回归

  ​		(M/16)x(N/16)x256的特征通过1x1卷积得到(M/16)x(N/16)x4k的输出，因为这里是生成每个anchor的坐标偏移量（用于修正anchor），[tx,ty,tw,th]共4个所以是4k。这里输出的是==坐标偏移量==，不是坐标本身，要得到修正后的anchor还要用原坐标和这个偏移量运算后才能得到预测的anchor。

  偏移公式：

  ![image-20221120134116615](https://raw.githubusercontent.com/hy515096690/paper-notes/main/img/202211201341693.png)

其中[xa,ya,wa,ha]是anchor的中心点坐标和宽高，[tx.ty,tw,th]是这个回归层预测的偏移量，通过这个公式计算出修正后的anchor坐标[x,y,w,h]。计算如下：

![image-20221120134445891](https://raw.githubusercontent.com/hy515096690/paper-notes/main/img/202211201344939.png)

[px,py,pw,ph]表示原始anchor的坐标
[dx,dy,dw,dh]表示RPN网络预测的坐标偏移
[gx,gy,gw,gh]表示修正后的anchor坐标。

- 3.4 生成Proposal

  <img src="https://raw.githubusercontent.com/hy515096690/paper-notes/main/img/202211201345660.png" alt="image-20221120134538593" style="zoom:50%;" />

  proposal层总共输入有：

  （1）cls层生成的(M/16)x(N/16)x2k向量
  （2）reg层生成的(M/16)x(N/16)x4k向量
  （3）im_info=[M, N,scale_factor]

  

  

  （1）利用reg层的偏移量，对所有的原始anchor进行修正
  （2）利用cls层的scores，按positive socres由大到小排列所有anchors，取前topN（比如6000个）个anchors
  （3）边界处理，把超出图像边界的positive anchor超出的部分收拢到图像边界处，防止后续RoI pooling时proposals超出边界。
  （4）剔除尺寸非常小的positive anchor
  （5）对剩余的positive anchors进行NMS（非极大抑制）
  （6）最后输出一堆proposals左上角和右下角坐标值（[x1,y1,x2,y2]对应原图MxN尺度）

- 4 ROI Pooling

  RoI Pooling层则负责收集proposal，并计算出proposal feature maps（从conv layers后的feature map中扣出对应位置），输入有两个：
  （1）conv layers提出的原始特征feature map，大小(M/16)x(N/16)
  （2）RPN网络生成的Proposals，大小各不相同。一堆坐标（[x1,y1,x2,y2]）
  ROI Pooling是一种能够把所有图像大小整合到一起又不会简单粗暴造成破坏的方法，这里使用的是RoI pooling，由SSP（Spatial Pyramid Pooling）发展而来。

  ==原理：==

  RoI pooling会有一个预设的pooled_w和pooled_h，表明要把每个proposal特征都统一为这么大的feature map
  （1）由于proposals坐标是基于MxN尺度的，先映射回(M/16)x(N/16)尺度
  （2）再将每个proposal对应的feature map区域分为pooled_w x pooled_h的网格
  （3）对网格的每一部分做max pooling
  （4）这样处理后，即使大小不同的proposal输出结果都是pooled_w x pooled_h固定大小，实现了固定长度输出
  <img src="https://raw.githubusercontent.com/hy515096690/paper-notes/main/img/202211201345526.png" alt="image-20221120134556434" style="zoom:33%;" />