<div align=center>Faster-RCNN</div>



目录



- 一、网络模型

  ![img](https://pic4.zhimg.com/80/v2-e64a99b38f411c337f538eb5f093bdf3_720w.webp)

  - 1）首先对图像预处理，调整到规定尺度，然后送入cnn网络。
  - 2）CNN包含13个卷积层+13个relu激活+4个pooling池化。
  - 3）将CNN层提取到的feature map输入到RPN网络，先对其3x3卷积，分别生成锚点（positive anchors）和对应的边界回归偏移量（bounding box regression），然后计算出预测边框（proposals）。
  - 4）ROI Pooling层对proposals与feature map中提取到的proposal feature送入fcn与softmax进行object分类和边框回归。

- 二、CNN层

  