HRNet

HRNet网络是针对2D人体姿态估计（Human Pose Estimation或Keypoint Detection）任务提出的。目前对人体姿态任务基于深度学习的主要有两种方法：

1、基于`regressing`的方式，即直接预测每个关键点的位置坐标。

2、基于`heatmap`的方式，即针对每个关键点预测一张热力图（预测出现在每个位置上的分数）。==（效果最好）==

- # 一、HRNet网络结构

![img](https://raw.githubusercontent.com/hy515096690/paper-notes/main/img/202211231600921.png)

1、首先通过两个卷积核大小为`3x3`步距为2的卷积层（后面都跟有BN以及ReLU）共下采样了4倍。然后通过`Layer1`模块，这里的`Layer1`和ResNet中的`Layer1`类似，重复堆叠`Bottleneck`，`Layer1`只会调整通道个数，并不会改变特征层大小。

2、接着通过一系列Transition结构以及Stage结构，每通过一个Transition结构都会新增一个尺度分支。比如说Transition1，它在layer1的输出基础上通过并行两个卷积核大小为3x3的卷积层得到两个不同的尺度分支，即下采样4倍的尺度以及下采样8倍的尺度。在Transition2中在原来的两个尺度分支基础上再新加一个下采样16倍的尺度，==这里是直接在下采样8倍的尺度基础上通过一个卷积核大小为3x3步距为2的卷积层得到下采样16倍的尺度==。

3、`Stage`结构：对于每个尺度分支，首先通过N个Basic Block，然后融合不同尺度上的信息。对于每个尺度分支上的输出都是由所有分支上的输出进行融合得到的。比如说对于下采样4倍分支的输出，它是分别将下采样4倍分支的输出（不做任何处理） 、 下采样8倍分支的输出通过Up x2上采样2倍 以及下采样16倍分支的输出通过Up x4上采样4倍进行相加最后通过ReLU得到下采样4倍分支的融合输出。其他分支也是类似的。图中右上角的xN表示该模块（Basic Block和Exchange Block）要重复堆叠N次。

4、对于所有的Up模块就是通过一个卷积核大小为1x1的卷积层然后BN层最后通过Upsample直接放大n倍得到上采样后的结果（这里的上采样默认采用的是nearest最邻近插值）。Down模块相比于Up稍微麻烦点，每下采样2倍都要增加一个卷积核大小为3x3步距为2的卷积层（Conv和Conv2d不同，Conv2d就是普通的卷积层，而Conv包含了卷积、BN以及ReLU激活函数）。

![image-20221123182526094](https://raw.githubusercontent.com/hy515096690/paper-notes/main/img/202211231825156.png)

5、