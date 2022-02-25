# 作业四：撰写项目README并完成开源(摸鱼版)



## 评分标准

1.格式规范（有至少3个小标题，内容完整），一个小标题5分，最高20分

2.图文并茂，一张图5分，最高20分

3.有可运行的代码，且代码内有详细注释，20分

4.代码开源到github，15分

5.代码同步到gitee，5分

## 作业目的
使用MarkDown撰写项目并且学会使用开源工具。



## 参考资料：
- [如何写好一篇高质量的精选项目？](https://aistudio.baidu.com/aistudio/projectdetail/2175889)



# 作业内容

把自己的项目描述放在对应区域中，形成一个完整流程项目，并且在最后声明自己项目的github和gitee链接



## 一、项目背景介绍

该部分主要向大家介绍你的项目目前社会研究情况，研究热度，或者用简短精炼的语言让大家理解为什么要做出这个项目

本项目为飞桨领航团AI达人创造营第二期作业四，参考教程[【CV小白极速上手必看】基于PaddleClas实现图像分类](https://aistudio.baidu.com/ibdcpu6/user/296307/3516036/files/http%3A?_xsrf=2|3bec468d|b5809bdd744d5e6d2b3b79b08d14e469|1644580444)完成基于PaddleClas实现食物图像分类项目。



## 二、数据介绍

该部分主要想大家介绍你的数据集基本情况，比如数据集的名字、来源,或者是图像的数量，类别，最后在把随机一张图像通过代码展示

- 数据集名字：[AI达人训练营]五种图像分类数据集

- 数据集来源：百度AI Studio公开数据集

- 数据集介绍

  > 共包含五个数据集，都是训练集。猫十二分类、场景分类、垃圾分类、食物分类、蝴蝶分类。作为AI训练营baseline使用



```python
# 图片展示代码，项目未加载数据集，使用图片代替
import matplotlib.pyplot as plt
d = plt.imread('./dataset/foods/baby_back_ribs/319516.jpg')

plt.imshow(d)
```

![Image](https://ai-studio-static-online.cdn.bcebos.com/866a9fd8ec074578ab9be88f2ceb7873dabbcb1535984d838b1979cf59b2de90)



## 三、模型介绍

想写好一个好的精品项目，项目的内容必须包含理论内容和实践相互结合，该部分主要是理论部分，向大家介绍一下你的模型原理等内容。

项目采用paddleclas的2.2版本

模型选用ShuffleNetV2，被称为轻量级CNN网络中的桂冠。

模型的作者分析了ShuffleNetv1设计的不足，（这里不再详述细节，以使用为主），并在v1的基础上改进得到了ShuffleNetv2，两者模块上的对比如下图所示：

![Image](https://ai-studio-static-online.cdn.bcebos.com/cbee721fb86242d3a113a2aa3a7a8dad0872d8027b4a474abe436daf9a8e1a9d)

为了弥补不足，引入了channel split操作，如上图：(a)和(b)图是ShuffleNet v1中的基础单元。(c)图是引入了channel split操作的ShuffleNet v2的基本单元。(d)图是步长为2时的单元。

近来，深度CNN网络如ResNet和DenseNet，已经极大地提高了图像分类的准确度。但是除了准确度外，计算复杂度也是CNN网络要考虑的重要指标，过复杂的网络可能速度很慢，一些特定场景如无人车领域需要低延迟。另外移动端设备也需要既准确又快的小模型。为了满足这些需求，一些轻量级的CNN网络如MobileNet和ShuffleNet被提出，它们在速度和准确度之间做了很好地平衡。今天我们要讲的是ShuffleNetv2，它是旷视最近提出的ShuffleNet升级版本，并被ECCV2018收录。在同等复杂度下，ShuffleNetv2比ShuffleNet和MobileNetv2更准确。

在同等条件下，ShuffleNetv2相比其他模型速度稍快，而且准确度也稍好一点。同时作者还设计了大的ShuffleNetv2网络，相比ResNet结构，其效果照样具有竞争力。

从一定程度上说，ShuffleNetv2借鉴了DenseNet网络，把shortcut结构从Add换成了Concat，这实现了特征重用。但是不同于DenseNet，v2并不是密集地concat，而且concat之后有channel shuffle以混合特征，这或许是v2即快又好的一个重要原因。



## 四、模型训练

该部分主要是实践部分，也是相对来说话费时间最长的一部分，该部分主要展示模型训练的内容，同时向大家讲解模型参数的设置

### 4.1 数据集划分

```python
import os
# -*- coding: utf-8 -*-
# 根据官方paddleclas的提示，我们需要把图像变为两个txt文件
# train_list.txt（训练集）
# val_list.txt（验证集）
# 先把路径搞定 比如：foods/beef_carpaccio/855780.jpg ,读取到并写入txt 

# 根据左侧生成的文件夹名字来写根目录
dirpath = "foods"
# 先得到总的txt后续再进行划分，因为要划分出验证集，所以要先打乱，因为原本是有序的
def get_all_txt():
    all_list = []
    i = 0 # 标记总文件数量
    j = -1 # 标记文件类别
    for root,dirs,files in os.walk(dirpath): # 分别代表根目录、文件夹、文件
        for file in files:
            i = i + 1 
            # 文件中每行格式： 图像相对路径      图像的label_id（数字类别）（注意：中间有空格）。              
            imgpath = os.path.join(root,file)
            all_list.append(imgpath+" "+str(j)+"\n")

        j = j + 1

    allstr = ''.join(all_list)
    f = open('all_list.txt','w',encoding='utf-8')
    f.write(allstr)
    return all_list , i
all_list,all_lenth = get_all_txt()
print(all_lenth)

# 把数据打乱
all_list = shuffle(all_list)
allstr = ''.join(all_list)
f = open('all_list.txt','w',encoding='utf-8')
f.write(allstr)
print("打乱成功，并重新写入文本")

# 按照比例划分数据集 食品的数据有5000张图片，不算大数据，一般9:1即可
train_size = int(all_lenth * 0.9)
train_list = all_list[:train_size]
val_list = all_list[train_size:]

print(len(train_list))
print(len(val_list))

# 运行cell，生成训练集txt 
train_txt = ''.join(train_list)
f_train = open('train_list.txt','w',encoding='utf-8')
f_train.write(train_txt)
f_train.close()
print("train_list.txt 生成成功！")

# 运行cell，生成验证集txt
val_txt = ''.join(val_list)
f_val = open('val_list.txt','w',encoding='utf-8')
f_val.write(val_txt)
f_val.close()
print("val_list.txt 生成成功！")
```



### 4.2 新增标签文件

新建文件 *labellist.txt*

内容

```
0 beef_carpaccio
1 baby_back_ribs
2 baklava
3 apple_pie
4 beef_tartare
```



### 4.3 模型训练

使用ShuffleNetV2_x0_25，相关参数配置如下

```
# global configs
Global:
  checkpoints: null
  pretrained_model: null
  output_dir: ./output/
  # 使用GPU训练
  device: gpu
  # 每几个轮次保存一次
  save_interval: 1 
  eval_during_train: True
  # 每几个轮次验证一次
  eval_interval: 1 
  # 训练轮次
  epochs: 10
  print_batch_step: 1
  use_visualdl: True #开启可视化（目前平台不可用）
  # used for static mode and model export
  # 图像大小
  image_shape: [3, 224, 224] 
  save_inference_dir: ./inference
  # training model under @to_static
  to_static: False

# model architecture
Arch:
  # 采用的网络
  name: ShuffleNetV2_x0_25
  class_num: 5 
 
# loss function config for traing/eval process
Loss:
  Train:

    - CELoss: 
        weight: 1.0
  Eval:
    - CELoss:
        weight: 1.0


Optimizer:
  name: Momentum
  momentum: 0.9
  lr:
    name: Piecewise
    learning_rate: 0.015
    decay_epochs: [30, 60, 90]
    values: [0.1, 0.01, 0.001, 0.0001]
  regularizer:
    name: 'L2'
    coeff: 0.0005


# data loader for train and eval
DataLoader:
  Train:
    dataset:
      name: ImageNetDataset
      # 根路径
      image_root: ./dataset/
      # 前面自己生产得到的训练集文本路径
      cls_label_path: ./dataset/foods/train_list.txt
      # 数据预处理
      transform_ops:
        - DecodeImage:
            to_rgb: True
            channel_first: False
        - ResizeImage:
            resize_short: 256
        - CropImage:
            size: 224
        - RandFlipImage:
            flip_code: 1
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''

    sampler:
      name: DistributedBatchSampler
      batch_size: 128
      drop_last: False
      shuffle: True
    loader:
      num_workers: 0
      use_shared_memory: True

  Eval:
    dataset: 
      name: ImageNetDataset
      # 根路径
      image_root: ./dataset/
      # 前面自己生产得到的验证集文本路径
      cls_label_path: ./dataset/foods/val_list.txt
      # 数据预处理
      transform_ops:
        - DecodeImage:
            to_rgb: True
            channel_first: False
        - ResizeImage:
            resize_short: 256
        - CropImage:
            size: 224
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
    sampler:
      name: DistributedBatchSampler
      batch_size: 128
      drop_last: False
      shuffle: True
    loader:
      num_workers: 0
      use_shared_memory: True

Infer:
  infer_imgs: ./dataset/foods/beef_carpaccio/855780.jpg
  batch_size: 10
  transforms:
    - DecodeImage:
        to_rgb: True
        channel_first: False
    - ResizeImage:
        resize_short: 256
    - CropImage:
        size: 224
    - NormalizeImage:
        scale: 1.0/255.0
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
        order: ''
    - ToCHWImage:
  PostProcess:
    name: Topk
    # 输出的可能性最高的前topk个
    topk: 5
    # 标签文件 需要自己新建文件
    class_id_map_file: ./dataset/foods/label_list.txt

Metric:
  Train:
    - TopkAcc:
        topk: [1, 5]
  Eval:
    - TopkAcc:
        topk: [1, 5]
```



## 五、模型评估

该部分主要是对训练好的模型进行评估，可以是用验证集进行评估，或者是直接预测结果。评估结果和预测结果尽量展示出来，增加吸引力。



目标图像

![Image](https://ai-studio-static-online.cdn.bcebos.com/b07b3d3ffb1b47b8a274c2f5020f3ef193898e5ffa364e818ab4db8114894035)

结果

```
[{'class_ids': [1, 4, 2, 3, 0],
'scores': [0.98095, 0.01024, 0.00309, 0.00294, 0.00277],
'file_name': 'dataset/foods/baby_back_ribs/319516.jpg',
'label_names': ['baby_back_ribs', 'beef_tartare', 'baklava', 'apple_pie', 'beef_carpaccio']}]
```

结果为：baby_back_ribs



## 六、总结与升华

接近到项目的尾声时，需要进行主题的升华，也就是说明本项目的亮点，以及不足之处，后续应该如何改进等

通过实例项目，快速上手PaddleClas完成目标图像分类，下一步尝试在其他数据集上使用不同模型完成图像分类。



## 七、个人总结

该部分也是项目最后一阶段，需要作者介绍自己，以及作者的兴趣方向等，最后在下面附上作者的个人主页链接

作者：zaiyu404

> 2月20日 实验室摸鱼
> 2月21日 实验室摸鱼
> 2月22日 继续摸鱼
> 2月23日 胡适之啊胡适之！你怎么能如此堕落！先前订的学习计划你都忘了吗？子曰：“吾日三省吾身”不能再这样下去了！
> 2月24日 打王者
> 2月25日 ……摸鱼
