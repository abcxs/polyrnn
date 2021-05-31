## 数据划分
- 切片大小：512
- 重叠：64
- 训练和测试数据：7:3
- 随机数种子1234

## 实验
比较模型：mask rcnn和panet

**对每一个实验，记录配置文件和地址，以及结果，随机数种子默认为1234**

baseline 进行不同的实验设置，具体挑选可后续选择

1. padding位置信息和non-local模块
2. 损失平滑(点的预测和多边形的预测)
3. attention结构
4. offset偏移分支
5. 训练流程和测试流程不一致
6. 解决显存问题
7. 特征融合（vertex,edge,mask）
    1. 仅提供损失
    2. concat
    3. 类似Bmask, 特征融合
8. 利用点的预测结果得分对最后结果进行调整

**对比实验**
- venus/0 faster rnn 4gpu, 0.01学习率，resnet50，12epoch
- venus/1 mask rcnn 同上
- venus/2 ms rnn 同上

**polyrnn**
由于使用lstm以及代码不完善，会OOM，限定每个GPU一个sample
暂定默认配置
1. 0.005, 4gpu, 24epoch
2. 未使用attention
3. 28
4. 最长序列20
5. 其他参数后续补充

- venus/3 0315
    - 从结果上看，前期epoch并不好，可能是损失比较苛刻，以及训练和测试流程不一致的问题
- venus/4 0315_1 使用attention
    - 根据实验结果，目前使用的attention结构并不好，需要修改
- venus/5 0315_2 最长序列使用30
- venus/6 0316 attention结构 + x, attention_type=2
- venus/7 0315_1 重做实验4
- venus/8 0316_2 多边形尺度为14
- venus/9 0317_2 原本测试attention_type=3,疯狂nan，暂时放弃，改利用更多的epoch
- venus/10 polygon_loss权重为0.1
- venus/11 0317_1 polygon_loss权重为0.5
- venus/12 0317_3 位置信息加载polyrnn_head头部分
- venus/13 0317_4 mask分支，仅利用mask损失
- venus/14 use_bn=True,lr=0.001
- venus/15 使用更小的池化大小，vertex_head通过上采样训练，结果很差，从损失上看仍然较大
- venus/16 0318_1 polygon_scale_factor=1.0 基本和3一样，不过多了边界去截断
- venus/17 0318_2 use_coord and coord_type=2
- venus/18 0318_3 时序为10
- venus/19 0318_4 polygon_scale_factor=1.1
- venus/20 0319 polygon_scale_factor=1.2
- venus/21 0319_1 refine_level=2, 16
- venus/22 0319_2 refine_level=1, non_local
- venus/23 0319_3 refine_level=0, 4
- venus/24 0321 空洞卷积 [2, 4, 6, 8], BN
- venus/25 0321_1 nonlocal 后使用上述模块
- venus/26 0321_2 polygon_scale_factor=1.05
- venus/27 0321_3 空洞卷积 [2, 4, 6, 8], GN
- venus/28 0321_4 backbone requires_grad=False
- venus/29 0321_5 101
- venus/30 0322 loss_type=1
- venus/31 0322_1 loss_type=2, 效果不忍直视
- venus/32 0322_2 loss_type=1, 48个epoch
- venus/33 0323 800训练
- venus/34 0323_1 poly_iou_thresh=0.5
- venus/35 0323_2 filter_multi_part=True
- venus/36 0323_3 poly_iou_thresh=0.3
- venus/37 0323_4 多尺度训练 800测试
- venus/38 0323_5 空间卷积放在polyrnn后面,误删，效果不好，不再重复做
- venus/39 0325 poly_radius=0
- venus/40 0326 HRNet 0.001
- venus/41 0326_1 loss_vertex weight=2.0 0.001
- venus/42 0326_2 loss_vertex weight=10.0 0.001
    - 42比41好，但不如16，是学习率的问题？
- venus/43 0326_3 mask sigmoid乘以 0.001
- venus/44 0326_4 loss_vertex weight=10.0 与42对比
- venus/45 0327 GaussianFocalLoss用于polyrnn
- venus/46 0327_1 重新训练baseline，对比16和3 
- venus_1/47 0329 epsilon=2, remove_abundant=True
- venus_1/48 0329_1 epsilon=2
- venus_1/49 0329_2 对比venus/43 更高的学习率0.005
- venus_1/50 0329_3 位置信息 多尺度 kernel_size=3, gaussian loss_vertex weight=10
- venus_1/51 0330 多边形尺度为32
- venus_1/52 0330_1 多边形尺度为16
- venus_1/53 0330_2 mask lr=0.01,对比venus/43和venus_1/49
- venus_1/54 0330_3 vertex_channels=64, edge_channels=64, type=1
- venus_1/55 0331 type=2
- venus_1/56 0331_1 type=4
- venus_1/57 0331_2 多次nan, type=5
- venus_1/58 0331_3 多边形尺度24
- venus_1/59 0331_4 with_offset=True
- venus_1/60 0331_5 4层卷积 num_convs=4
- venus_1/61 0331_6 num_convs=4 GN
- venus_1/62 0403 使用Faster预训练的权重，再训练，解决损失检测性能的问题等
- venus_1/63 0403_1 翻转+数据增强，效果似乎并不是太好，需要重新测试
- venus_1/64 只水平增强
- venus_1/65 0.3 0.3 水平，竖直概率
- venus_1/66 type=6


- inference/1 venus/44 0326_4 使用kernel_size=3,guassian
- inference/2 venus/44 0326_4 使用kernel_size=3,constant
- inference/3 venus/46 0327_1 使用kernel_size=3,constant
- inference/4 venus/46 0327_1 使用kernel_size=3,guassian
- inference/5 venus/45 0327 使用kernel_size=3,guassian
- inference/6 venus/45 0327 使用kernel_size=3,constant
- inference/7 venus/50 0329_3 使用640
- inference/8 venus/50 0329_3 使用704


**表格结果（仅列出最后一个epoch）**

work_dir | map | ap@50 | ap@75 | ap@small | ap@medium | ap@large
- | - | - | - | - | - | - 
0 | 40.4/- | 61.6/- | 45.2/- | 25.2/- | 48.5/- | 50.5/-
1 | 40.7/38.4 | 62.0/60.8 | 45.3/43.3 | 25.8/24.3 | 49.0/47.1 | 49.5/44.2
2 | 41.1/39.4 | 62.6/61.8 | 45.9/44.1 | 26.8/27.3 | 48.6/46.7 | 49.4/42.7
3 | 39.4/32.8 | 61.5/57.7 | 43.3/35.2 | 28.9/24.8 | 46.6/40.8 | 44.8/30.0
4 | 39.3/30.7 | 61.0/55.1 | 43.0/31.9 | 29.0/23.2 | 47.0/39.3 | 44.8/27.3
5 | 38.5/32.2 | 59.9/56.1 | 42.6/34.5 | 28.9/24.2 | 45.9/39.6 | 37.6/27.0
6 | 39.5/32.8 | 60.9/57.1 | 43.7/35.4 | 29.7/25.3 | 46.8/40.9 | 44.1/29.3
7 | 39.9/31.1 | 61.7/56.1 | 44.0/32.4 | 28.6/23.4 | 47.5/39.5 | 46.7/27.5
8 | 40.5/29.8 | 62.4/58.7 | 44.5/28.8 | 29.4/25.1 | 47.6/36.8 | 46.9/25.1
9 | 38.5/32.9 | 60.0/57.0 | 41.8/35.5 | 27.6/24.0 | 45.3/40.6 | 43.5/30.7
10 | 39.9/27.3 | 59.9/50.3 | 44.2/28.4 | 25.5/17.7 | 47.0/35.2 | 47.8/23.6
11 | 40.6/32.6 | 62.4/57.4 | 44.3/34.9 | 29.7/24.7 | 47.9/40.8 | 46.1/26.2
12 | 39.6/33.5 | 61.7/58.0 | 43.7/36.1 | 29.4/25.0 | 46.4/41.1 | 45.7/32.3
13 | 39.2/32.8 | 61.0/57.4 | 43.3/35.1 | 29.2/24.8 | 46.0/40.2 | 44.3/31.2
14 | 38.6/31.3 | 61.0/56.2 | 42.2/32.7 | 28.4/22.4 | 46.3/39.7 | 42.3/29.2
15 | 39.6/11.0 | 61.8/24.4 | 43.7/10.3 | 28.9/12.2 | 46.9/13.7 | 43.9/7.5
16 | 39.4/33.0 | 61.4/57.7 | 43.3/35.4 | 29.4/25.0 | 46.3/40.8 | 44.6/30.7
17 | 39.6/33.1 | 61.7/57.4 | 43.8/35.7 | 29.4/24.8 | 46.9/41.3 | 44.3/30.2
18 | 39.7/31.2 | 61.4/55.1 | 43.8/32.9 | 29.3/24.9 | 46.8/40.2 | 44.9/22.3
19 | 39.3/33.0 | 61.4/57.4 | 43.0/35.3 | 29.2/25.2 | 46.2/40.8 | 44.8/31.0
20 | 39.1/32.4 | 61.1/56.5 | 43.2/35.1 | 29.1/24.7 | 46.3/40.4 | 43.9/29.4
21 | 39.2/31.7 | 61.2/53.9 | 43.1/32.9 | 29.4/22.8 | 46.9/39.9 | 41.9/29.5
22 | 39.9/33.0 | 62.0/57.9 | 43.9/34.4 | 29.6/24.5 | 46.9/41.1 | 45.4/29.5
23 | 39.2/33.1 | 61.6/57.9 | 42.8/35.4 | 28.6/24.9 | 46.4/41.0 | 44.7/31.2
24 | 39.4/31.3 | 61.5/56.1 | 43.3/33.0 | 29.7/23.7 | 46.6/39.6 | 45.5/28.6
25 | 39.5/31.8 | 61.8/57.0 | 43.5/33.0 | 29.7/24.0 | 46.6/39.8 | 45.0/28.5
26 | 39.9/32.8 | 61.9/57.1 | 43.9/35.2 | 29.7/25.1 | 47.2/40.8 | 45.1/30.3
27 | 39.4/32.8 | 61.6/57.5 | 43.1/35.1 | 28.8/24.4 | 46.7/40.9 | 43.7/28.8
28 | 39.4/32.8 | 61.5/57.4 | 42.8/34.9 | 29.3/24.4 | 46.4/40.6 | 44.8/31.5
29 | 38.6/32.1 | 60.6/56.6 | 42.4/34.5 | 28.5/23.8 | 45.3/39.8 | 45.8/31.2
30 | 39.7/32.9 | 61.5/57.3 | 43.5/35.2 | 28.8/24.6 | 47.0/41.0 | 45.5/30.2
32 | 38.7/32.9 | 60.1/56.9 | 42.5/35.5 | 27.9/24.4 | 45.4/40.5 | 43.9/29.5
33 | 40.7/34.3 | 62.2/58.6 | 44.5/37.3 | 31.0/27.2 | 48.9/43.0 | 41.7/29.0
34 | 39.5/33.1 | 61.6/57.9 | 43.5/35.5 | 29.3/24.9 | 46.7/41.2 | 43.9/30.6
35 | 39.3/33.1 | 61.0/57.8 | 42.8/35.5 | 29.2/24.8 | 46.3/40.8 | 43.4/31.2
36 | 39.1/32.8 | 61.2/57.0 | 43.2/35.0 | 28.9/24.2 | 46.4/40.7 | 43.5/30.4
37 | 42.0/35.3 | 63.4/59.9 | 46.6/38.3 | 33.7/29.5 | 49.3/43.3 | 43.3/29.9
38 | 39.0/32.3 | 60.8/56.8 | 42.6/34.1 | 29.0/24.7 | 46.0/40.0 | 43.5/29.9
39 | 39.4/32.8 | 61.6/57.8 | 43.1/35.2 | 28.9/24.6 | 46.4/40.7 | 44.6/29.7
40 | 40.6/31.3 | 61.7/55.0 | 44.6/32.9 | 29.5/22.8 | 47.6/39.6 | 47.1/29.0
41 | 38.5/29.7 | 60.7/54.2 | 42.6/30.6 | 27.8/20.9 | 46.3/37.9 | 43.2/28.1
42 | 38.8/30.2 | 61.4/54.7 | 42.4/31.5 | 28.6/22.1 | 46.2/38.5 | 42.5/26.8
43 | 39.4/31.3 | 61.7/55.6 | 43.4/33.3 | 28.5/22.3 | 46.8/39.2 | 44.7/30.8
44 | 39.3/33.2 | 61.1/57.5 | 43.4/35.9 | 28.8/24.5 | 46.7/41.3 | 43.3/30.8
45 | 40.8/32.5 | 62.7/57.3 | 44.6/34.8 | 29.2/24.5 | 48.0/40.9 | 46.5/27.0
46 | 39.6/33.0 | 61.5/57.3 | 43.6/35.3 | 29.2/25.0 | 46.8/41.0 | 45.1/30.4
47 | 40.2/31.0 | 62.1/55.1 | 44.4/33.1 | 29.0/23.1 | 47.4/39.0 | 45.7/27.7
48 | 40.0/32.9 | 62.0/57.8 | 44.3/35.4 | 29.4/25.4 | 47.2/41.2 | 45.0/28.6
49 | 39.4/33.2 | 61.3/57.3 | 43.3/36.1 | 28.6/24.6 | 46.5/41.4 | 44.7/30.5
50 | 41.8/36.5 | 63.2/60.9 | 46.0/40.0 | 33.6/30.9 | 49.6/44.6 | 41.8/31.2
51 | 39.5/33.2 | 61.5/56.9 | 43.4/35.6 | 29.3/24.5 | 46.7/41.6 | 44.1/29.2
52 | 40.0/30.7 | 62.1/58.5 | 43.4/31.1 | 29.2/24.9 | 47.1/37.9 | 44.4/25.6
53 | 38.5/33.1 | 60.4/57.4 | 42.3/35.5 | 29.0/25.0 | 45.2/40.5 | 44.1/32.4
54 | 39.2/32.9 | 61.4/57.6 | 43.0/35.4 | 29.3/25.1 | 46.2/40.9 | 44.5/30.7
55 | 39.7/32.6 | 61.9/57.6 | 43.7/34.5 | 28.9/24.1 | 47.0/40.8 | 46.2/30.0
56 | 39.4/33.2 | 61.3/57.8 | 43.4/35.4 | 28.9/24.8 | 46.6/41.4 | 44.6/30.7
57 | 39.6/33.5 | 61.7/57.7 | 43.4/36.2 | 28.8/25.0 | 46.8/41.6 | 45.1/31.3
58 | 39.8/32.7 | 61.7/57.7 | 43.8/35.1 | 29.2/25.1 | 46.8/40.3 | 45.6/29.9
59 | 39.7/33.9 | 61.4/57.7 | 43.9/36.3 | 29.0/23.8 | 46.7/42.1 | 45.6/33.3
60 | 39.2/33.2 | 61.5/57.9 | 43.1/35.4 | 29.0/25.2 | 46.2/40.9 | 43.8/30.6
61 | 39.2/33.4 | 61.0/57.7 | 43.4/36.1 | 29.0/24.9 | 46.6/41.4 | 44.7/33.1
62 | 40.3/33.9 | 62.1/58.2 | 44.8/36.7 | 29.5/25.3 | 47.4/42.0 | 46.1/32.2
63 | 39.5/33.3 | 60.1/57.0 | 43.7/35.8 | 31.6/27.8 | 47.9/41.8 | 38.8/27.8
64 | 42.1/35.7 | 64.0/60.1 | 46.5/38.7 | 33.0/29.1 | 49.6/43.8 | 44.3/30.9
65 | 41.5/35.2 | 62.6/59.4 | 46.0/38.4 | 33.0/28.6 | 49.1/43.1 | 43.2/31.8
66 | 39.5/33.3 | 61.5/57.6 | 43.7/36.1 | 29.5/25.5 | 46.7/41.3 | 44.1/29.8

inference/1 | 33.9 | 57.7 | 36.8 | 25.8 | 41.7 | 31.9
inference/2 | 30.2 | 57.0 | 31.2 | 25.9 | 37.1 | 25.8
inference/3 | 29.9 | 57.5 | 30.5 | 26.1 | 37.0 | 25.3
inference/4 | 33.4 | 58.4 | 35.7 | 25.9 | 41.1 | 30.8
inference/5 | 33.4 | 59.1 | 35.5 | 26.8 | 41.3 | 27.6
inference/6 | 29.2 | 57.1 | 28.7 | 25.9 | 36.0 | 22.3
inference/7 | 42.1/36.4 | 63.9/61.0 | 46.3/40.3 | 32.7/29.7 | 49.9/44.7 | 47.0/34.5
inference/8 | 42.3/36.8 | 63.7/61.0 | 46.8/40.8 | 33.1/30.5 | 50.2/45.1 | 44.8/33.1

**分析**
1. ms rcnn在小物体提升明显，小物体上更好的感受野以及一种监督损失？
2. 实验3表示分割精度和检测精度相差较大，特别是大物体，或许需要更好的感受野？ap@75相比于ap@50,检测和分割精度相差更大，也就是更精细的定位上有问题
3. 实验4基于attention的效果并大好，甚至有所下降（attetion处代码有问题），可以放弃
5. 实验5增加序列长度，并没有获得更好的效果，很奇怪，从损失上看，比实验3较大，后期效果基本稳定，也许存在权重影响
6. 从实验7对比实验4来看，会稍微更好些，但是差别不大
7. 实验8用了更小的尺度，但是检测精度高了，而分镜精度主要影响大尺度物体，小尺度物体甚至有提升，损失权重需要调整，对比实验3
8. 实验16和实验19，scale_factor影响结果不大
9. 35对比16,在大物体上有少量收益


**代码日志**
1. 0316
    修改attention部分代码，增加参数attention_type，默认参数为1，另外参数2,3，对应不同的结构，暂时调参使用
    原始attention错误，relu激活之后，采用softmax，性能差是否有这方面的影响，需要重新做实验4
    注释 attention_type == 3时，疯狂nan
2. 0317
    小修改，原始配置文件，加入损失参数，不会影响实验
    添加坐标信息，在第一次的PolyRnnHead卷积处
3. 0318
    增加缩放尺度，应该不会很影响，建议测试（不确定之前是否截断图片尺度）,通过原始值为None，来确保和之前一样
    attention修改， attention_type=5, 仅在1的基础上使用sigmoid
    坐标信息，在lstm前每个都进行，coord_type=2
    polygon_size 不再限制(需要缩放vertex_head，对结果可能并不好)，此处是否可以先池化，后升维再做，显存瓶颈在lstm处
4. 0320
    FusionModule更改，增加空洞卷积等
5. 0322
    损失增加，loss_type
    loss_type=1, 从外部对损失改权重，看代码
    loss_type=2, 为设定对softmax加权修改
6. 0323
    空洞卷积加载Polyrnn后面，没啥用
7. 0325
    forward传递vertex_predict，增加beam_search
8. 0327
    loss_type=3, sigmoid损失
    mask_predict 分支
    初始化bias, focal_loss问题，vertex_pred分支已加上，polyrnn_head设定loss_type=3时使用，对应损失
    激活函数 softmax还是sigmoid, 通过act_test 仅利用在测试，后续测试性能
    加权平均 通过weight_kernel_params, kernel_size > 1进行 仅利用在测试，后续测试性能
9. 0329
    edge损失
10. 0403
    offset偏移量
    修正错误，从0325存在，第一点似乎被忽略，自实验编号0331_2和0331_4修正


**参数调整**
注意：实验结果从上面表格获取，此处作为总结

测试Polyon_size影响，后续可能会改代码流程

work_dir | polygon_size |map | ap@50 | ap@75 | ap@small | ap@medium | ap@large
- | - | - | - | - | - | - | - 
3 | 28 | 39.4/32.8 | 61.5/57.7 | 43.3/35.2 | 28.9/24.8 | 46.6/40.8 | 44.8/30.0
8 | 14 | 40.5/29.8 | 62.4/58.7 | 44.5/28.8 | 29.4/25.1 | 47.6/36.8 | 46.9/25.1

测试时序长度
work_dir | polygon_size |map | ap@50 | ap@75 | ap@small | ap@medium | ap@large
- | - | - | - | - | - | - | - 
3  | 20 | 39.4/32.8 | 61.5/57.7 | 43.3/35.2 | 28.9/24.8 | 46.6/40.8 | 44.8/30.0
5  | 30 | 38.5/32.2 | 59.9/56.1 | 42.6/34.5 | 28.9/24.2 | 45.9/39.6 | 37.6/27.0
18 | 10 | 39.7/31.2 | 61.4/55.1 | 43.8/32.9 | 29.3/24.9 | 46.8/40.2 | 44.9/22.3

**需要做的实验**
对输入放松
polygon得分和检测得分进行合并
offset偏移
类似snake，提取出来之后，去回归

**日常**
1. offset
2. GN
3. 4层
4. 数据增强，baseline结果


**有用的策略**
1. 位置信息
2. 多尺度
3. kernel_size=3, gaussian
4. loss_vertex weight=10
5. filter_multi_part=True, poly_iou_thresh=0.5 收益很小，可能有偶然性
6. 4conv + GN
7. offset偏移

重新验证下多尺度
baseline：4CONV+GN, filter_mutli_part=True, loss_vertex weight=10，vertex_edge_信息
ms baseline并不好
额外组件：位置信息，kernel_size,offset偏移，预训练参数
消融实验：
在baseline上进行调整
1. 投影尺寸(7, 14, 28, 16, 32) 5
2. 组件信息 4
3. kernel_size信息 4
4. 时序长度 3
5. 有无预训练模型，不同的结果 2
