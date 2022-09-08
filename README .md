# 飞桨学习赛：遥感影像地块分割 - 22年8月第6名

### 赛题介绍
本赛题由 2020 CCF BDCI 遥感影像地块分割 初赛赛题改编而来。遥感影像地块分割, 旨在对遥感影像进行像素级内容解析，对遥感影像中感兴趣的类别进行提取和分类，在城乡规划、防汛救灾等领域具有很高的实用价值，在工业界也受到了广泛关注。现有的遥感影像地块分割数据处理方法局限于特定的场景和特定的数据来源，且精度无法满足需求。因此在实际应用中，仍然大量依赖于人工处理，需要消耗大量的人力、物力、财力。本赛题旨在衡量遥感影像地块分割模型在多个类别（如建筑、道路、林地等）上的效果，利用人工智能技术，对多来源、多场景的异构遥感影像数据进行充分挖掘，打造高效、实用的算法，提高遥感影像的分析提取能力。
赛题任务
本赛题旨在对遥感影像进行像素级内容解析，并对遥感影像中感兴趣的类别进行提取和分类，以衡量遥感影像地块分割模型在多个类别（如建筑、道路、林地等）上的效果。

### 数据说明
本赛题提供了多个地区已脱敏的遥感影像数据，各参赛选手可以基于这些数据构建自己的地块分割模型。

### 训练数据集
样例图片及其标注如下图所示：

![](https://ai-studio-static-online.cdn.bcebos.com/8087a965609d48a19a5e60f0330fa9054d04097644de48ffa3d557e7a8ad64ad)
![](https://ai-studio-static-online.cdn.bcebos.com/d18664ecf0514cb686c95958d30bbf8a2f5efb0691bc4d66a5f6317ab511d6d0)

![](https://ai-studio-static-online.cdn.bcebos.com/e42f2c222f204094ac3a0ea8582ca331b0452fb2b1704eabaae379d499906977)
![](https://ai-studio-static-online.cdn.bcebos.com/d5260bd5a820486a85aeb2105adfb6fa10284bd94453459f892755bc43e10b8a)


训练数据集文件名称：train_and_label.zip

包含2个子文件，分别为：训练数据集（原始图片）文件、训练数据集（标注图片）文件，详细介绍如下：

* **训练数据集**（原始图片）文件名称：img_train

	包含66,653张分辨率为2m/pixel，尺寸为256 * 256的JPG图片，每张图片的名称形如T000123.jpg。
* **训练数据集**（标注图片）文件名称：lab_train

	包含66,653张分辨率为2m/pixel，尺寸为256 * 256的PNG图片，每张图片的名称形如T000123.png。
* **标签说明**： 全部PNG图片共包括4种分类，像素值分别为0、1、2、3。此外，像素值255为未标注区域，表示对应区域的所属类别并不确定，在评测中也不会考虑这部分区域。

### 比赛项目关键点

本比赛实质上还是一个语义分割的任务，但是在于类与类之间的分割界面不容易区分，对模型的语义感知和理解能力上有要求。

其次由于该数据集类与类之间界限比较模糊，并且类别形状的不规则，给分割任务带来一定的挑战难度。
                                  
### 模型介绍

比赛使用的模型为：**SegFormer**

SegFormer是一款基于Transformer构建的具有简单结构的语义分割网络，其基础网络结构如下图所示：

![](https://ai-studio-static-online.cdn.bcebos.com/eef5b861bcfc40c7a8c6d92184d5e6a2518c2260e2ba4c0086dc15e6553a7b91)

原基础模型的Decoder部分，是一个MLP来将不同特种层输出的特征进行融合，然后经过一个MLP再进行上采样。

受到MSRA组在Xception上改进工作可变形卷积(Deformable-ConvNets)启发，Deformable-ConvNets对Xception做了改进，能够进一步提升模型学习能力，新的结构如下：

![](https://ai-studio-static-online.cdn.bcebos.com/21eb9ac43cd54e0f9e8f211e6c2598c4c10f9d439b6a42d39955eed46a588755)

分割模型在融合了自注意力后，使得分割模型具有对图像信息更细节的感知。

### 数据增方法

本项目所使用的增强方法主要如下：

* **RandomHorizontalFlip**
* **RandomVerticalFlip**
* **RandomPaddingCrop**
* **RandomBlur**
* **RandomRotation**



### 环境安装


```python
!git clone https://gitee.com/paddlepaddle/PaddleSeg.git
```

    Cloning into 'PaddleSeg'...
    remote: Enumerating objects: 19547, done.[K
    remote: Counting objects: 100% (4516/4516), done.[K
    remote: Compressing objects: 100% (2161/2161), done.[K
    remote: Total 19547 (delta 2756), reused 3826 (delta 2290), pack-reused 15031[K
    Receiving objects: 100% (19547/19547), 345.32 MiB | 11.64 MiB/s, done.
    Resolving deltas: 100% (12616/12616), done.
    Checking connectivity... done.



```python
# 安装所需依赖项
!pip install -r PaddleSeg/requirements.txt
```

    Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
    Requirement already satisfied: pyyaml>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r PaddleSeg/requirements.txt (line 1)) (5.1.2)
    Requirement already satisfied: visualdl>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r PaddleSeg/requirements.txt (line 2)) (2.2.0)
    Requirement already satisfied: opencv-python in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r PaddleSeg/requirements.txt (line 3)) (4.1.1.26)
    Requirement already satisfied: tqdm in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r PaddleSeg/requirements.txt (line 4)) (4.36.1)
    Requirement already satisfied: filelock in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r PaddleSeg/requirements.txt (line 5)) (3.0.12)
    Requirement already satisfied: scipy in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r PaddleSeg/requirements.txt (line 6)) (1.6.3)
    Requirement already satisfied: prettytable in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r PaddleSeg/requirements.txt (line 7)) (0.7.2)
    Requirement already satisfied: sklearn in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r PaddleSeg/requirements.txt (line 8)) (0.0)
    Requirement already satisfied: numpy in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (1.20.3)
    Requirement already satisfied: pre-commit in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (1.21.0)
    Requirement already satisfied: six>=1.14.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (1.16.0)
    Requirement already satisfied: shellcheck-py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (0.7.1.1)
    Requirement already satisfied: matplotlib in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (2.2.3)
    Requirement already satisfied: bce-python-sdk in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (0.8.53)
    Requirement already satisfied: Flask-Babel>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (1.0.0)
    Requirement already satisfied: requests in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (2.22.0)
    Requirement already satisfied: flask>=1.1.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (1.1.1)
    Requirement already satisfied: flake8>=3.7.9 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (4.0.1)
    Requirement already satisfied: protobuf>=3.11.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (3.20.1)
    Requirement already satisfied: Pillow>=7.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (7.1.2)
    Requirement already satisfied: pandas in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (1.1.5)
    Requirement already satisfied: scikit-learn in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from sklearn->-r PaddleSeg/requirements.txt (line 8)) (0.24.2)
    Requirement already satisfied: pyflakes<2.5.0,>=2.4.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (2.4.0)
    Requirement already satisfied: importlib-metadata<4.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (4.2.0)
    Requirement already satisfied: pycodestyle<2.9.0,>=2.8.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (2.8.0)
    Requirement already satisfied: mccabe<0.7.0,>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (0.6.1)
    Requirement already satisfied: Jinja2>=2.10.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (3.0.0)
    Requirement already satisfied: itsdangerous>=0.24 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (1.1.0)
    Requirement already satisfied: Werkzeug>=0.15 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (0.16.0)
    Requirement already satisfied: click>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (7.0)
    Requirement already satisfied: Babel>=2.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (2.8.0)
    Requirement already satisfied: pytz in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (2019.3)
    Requirement already satisfied: future>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (0.18.0)
    Requirement already satisfied: pycryptodome>=3.8.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (3.9.9)
    Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (1.1.0)
    Requirement already satisfied: cycler>=0.10 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (0.10.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (3.0.9)
    Requirement already satisfied: python-dateutil>=2.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (2.8.2)
    Requirement already satisfied: toml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (0.10.0)
    Requirement already satisfied: virtualenv>=15.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (16.7.9)
    Requirement already satisfied: aspy.yaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (1.3.0)
    Requirement already satisfied: identify>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (1.4.10)
    Requirement already satisfied: nodeenv>=0.11.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (1.3.4)
    Requirement already satisfied: cfgv>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (2.0.1)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (2019.9.11)
    Requirement already satisfied: idna<2.9,>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (2.8)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (1.25.6)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (3.0.4)
    Requirement already satisfied: joblib>=0.11 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn->sklearn->-r PaddleSeg/requirements.txt (line 8)) (0.14.1)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn->sklearn->-r PaddleSeg/requirements.txt (line 8)) (2.1.0)
    Requirement already satisfied: typing-extensions>=3.6.4 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from importlib-metadata<4.3->flake8>=3.7.9->visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (4.3.0)
    Requirement already satisfied: zipp>=0.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from importlib-metadata<4.3->flake8>=3.7.9->visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (3.8.1)
    Requirement already satisfied: MarkupSafe>=2.0.0rc2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Jinja2>=2.10.1->flask>=1.1.1->visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (2.0.1)
    Requirement already satisfied: setuptools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from kiwisolver>=1.0.1->matplotlib->visualdl>=2.0.0->-r PaddleSeg/requirements.txt (line 2)) (56.2.0)
    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip available: [0m[31;49m22.1.2[0m[39;49m -> [0m[32;49m22.2.2[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m


### 解压数据集


```python
!unzip -q data/data80164/train_and_label.zip
!unzip -q data/data80164/img_test.zip
```

### 数据处理



```python
import os
import numpy as np

datas = []
image_base = 'img_train'   # 训练集原图路径
annos_base = 'lab_train'   # 训练集标签路径

# 读取原图文件名
ids_ = [v.split('.')[0] for v in os.listdir(image_base)]

# 将训练集的图像集和标签路径写入datas中
for id_ in ids_:
    img_pt0 = os.path.join(image_base, '{}.jpg'.format(id_))
    img_pt1 = os.path.join(annos_base, '{}.png'.format(id_))
    datas.append((img_pt0.replace('/home/aistudio', ''), img_pt1.replace('/home/aistudio', '')))
    if os.path.exists(img_pt0) and os.path.exists(img_pt1):
        pass
    else:
        raise "path invalid!"

# 打印datas的长度和具体存储例子
print('total:', len(datas))
print(datas[0][0])
print(datas[0][1])
print(datas[10][:])
```

    total: 66652
    img_train/T053746.jpg
    lab_train/T053746.png
    ('img_train/T084102.jpg', 'lab_train/T084102.png')



```python
import numpy as np

# 四类标签，这里用处不大，比赛评测是以0、1、2、3类来对比评测的
labels = ['建筑', '耕地', '林地',  '其他']

# 将labels写入标签文件
with open('labels.txt', 'w') as f:
    for v in labels:
        f.write(v+'\n')

# 随机打乱datas
np.random.seed(5)
np.random.shuffle(datas)

# 验证集与训练集的划分，0.05表示5%为训练集，95%为训练集
split_num = int(0.05*len(datas))

# 划分训练集和验证集
train_data = datas[:-split_num]
valid_data = datas[-split_num:]

# 写入训练集list
with open('train_list.txt', 'w') as f:
    for img, lbl in train_data:
        f.write(img + ' ' + lbl + '\n')

# 写入验证集list
with open('valid_list.txt', 'w') as f:
    for img, lbl in valid_data:
        f.write(img + ' ' + lbl + '\n')

# 打印训练集和测试集大小
print('train:', len(train_data))
print('valid:', len(valid_data))
```

    train: 63320
    valid: 3332


### 模型调参

* **train_batch_size (int)**: 训练数据batch大小。同时作为验证数据batch大小。默认2。
* **eval_dataset (paddlex.datasets)**: 评估数据读取器。
* **save_interval_epochs (int)**: 模型保存间隔（单位：迭代轮数）。默认为1。
* **log_interval_steps (int)**: 训练日志输出间隔（单位：迭代次数）。默认为2。
* **save_dir (str)**: 模型保存路径。默认’output’
* **pretrain_weights (str)**: 若指定为路径时，则加载路径下预训练模型。
* **optimizer (paddle.fluid.optimizer)**: 优化器。当该参数为None时，使用默认的优化器：使用fluid.optimizer.Momentum优化方法的学习率衰减策略。
* **learning_rate (float)**: 默认优化器的初始学习率。默认0.01。
* **lr_decay_power (float)**: 默认优化器学习率衰减指数。默认0.9。
* **use_vdl (bool)**: 是否使用VisualDL进行可视化。默认False。
* **sensitivities_file (str)**: 若指定为路径时，则加载路径下敏感度信息进行裁剪；若为字符串’DEFAULT’，则自动下载在Cityscapes图片数据上获得的敏感度信息进行裁剪；若为None，则不进行裁剪。默认为None。
* **eval_metric_loss (float)**: 可容忍的精度损失。默认为0.05。
* **early_stop (bool)**: 是否使用提前终止训练策略。默认值为False。
* **early_stop_patience (int)**: 当使用提前终止训练策略时，如果验证集精度在early_stop_patience个epoch内连续下降或持平，则终止训练。

### 模型训练


**更改训练文件 PaddleSeg/paddleseg/core/train.py**

```
def train(model,
          train_dataset,
          val_dataset=None,
          optimizer=None,
          save_dir='output',
          iters=10000,
          batch_size=2,
          resume_model=None,
          save_interval=1000,
          log_iters=10,
          num_workers=0,
          use_vdl=False,
          losses=None,
          keep_checkpoint_max=5)
```

**配置模型文件：PaddleSeg/configs/deeplabv3p/deeplabv3p_resnet101_os8_cityscapes_1024x512_80k.yml**
```
_base_: '../_base_/cityscapes.yml'

batch_size: 2
iters: 20000

model:
  type: DeepLabV3P
  backbone:
    type: ResNet101_vd
    output_stride: 8
    multi_grid: [1, 2, 4]
    pretrained: https://bj.bcebos.com/paddleseg/dygraph/resnet101_vd_ssld_v2.tar.gz  #加载预训练权重文件
  num_classes: 4
  backbone_indices: [0, 3]
  aspp_ratios: [1, 12, 24, 36]
  aspp_out_channels: 256
  align_corners: False
  pretrained: null
```
**配置训练数据集文件：PaddleSeg/configs/_base_/cityscapes.yml**
```
train_dataset:
  type: Dataset
  dataset_root: /home/aistudio
  train_path: /home/aistudio/train_list.txt
  num_classes: 4
  transforms:
    - type: ResizeStepScaling
      min_scale_factor: 0.5
      max_scale_factor: 2.0
      scale_step_size: 0.25
    - type: RandomPaddingCrop
      crop_size: [1024, 512]
    - type: RandomHorizontalFlip
    - type: RandomDistort
      brightness_range: 0.4
      contrast_range: 0.4
      saturation_range: 0.4
    - type: Normalize
  mode: train

val_dataset:
  type: Dataset
  dataset_root: /home/aistudio
  val_path: /home/aistudio/valid_list.txt
  num_classes: 4
  transforms:
    - type: Normalize
  mode: val


optimizer:
  type: sgd
  momentum: 0.9
  weight_decay: 4.0e-5

learning_rate:
  value: 0.0001
  decay:
    type: poly
    power: 0.9
    end_lr: 0.0

loss:
  types:
    - type: CrossEntropyLoss
  coef: [1]
```

### 开始训练


```python
!python PaddleSeg/train.py \
        --config PaddleSeg/configs/deeplabv3p/deeplabv3p_resnet101_os8_cityscapes_1024x512_80k.yml \
        --use_vdl \
        --do_eval \
        --save_interval 10000 \
        --save_dir output
```

### 模型评估
> PaddleSeg/paddleseg/core/val.py
```
def evaluate(model,
             eval_dataset,
             aug_eval=False,
             scales=1.0,
             flip_horizontal=True,
             flip_vertical=False,
             is_slide=False,
             stride=None,
             crop_size=None,
             num_workers=0,
             print_detail=True)
```


```python
!python PaddleSeg/val.py \
        --config PaddleSeg/configs/deeplabv3p/deeplabv3p_resnet101_os8_cityscapes_1024x512_80k.yml \
        --model_path PaddleSeg/output/best_model/model.pdparams 
```

### 模型预测

> PaddleSeg/paddleseg/core/predict.py


```
def predict(model,
            model_path,
            transforms,
            image_list,
            image_dir=None,
            save_dir='output',
            aug_pred=False,
            scales=1.0,
            flip_horizontal=True,
            flip_vertical=False,
            is_slide=False,
            stride=None,
            crop_size=None)
```
```
pred_im = utils.visualize(im_path, pred, weight=0.0)
result_saved_path = os.path.join(result_saved_dir, im_file)
mkdir(result_saved_path)
cv2.imwrite(pred_saved_path, pred_im)
```


```python
!python PaddleSeg/predict.py \
       --config PaddleSeg/configs/deeplabv3p/deeplabv3p_resnet101_os8_cityscapes_1024x512_80k.yml \
       --model_path output/best_model/model.pdparams \
       --image_path img_testA \
       --save_dir result
```


```python
# 由预测结果生成提交文件
!zip -r result.zip result/result/
```

### 总结

对于这个分数，其实还要很多可以改进的地方看其他开源的方案。
感谢百度提供这样的学习平台，同学可根据自己喜欢的方向，通过aistudio上的课程和提供的训练平台，进行理论与实践并存的学习。
#### 飞桨领航团实战速成营：[https://aistudio.baidu.com/aistudio/education/group/info/16606](https://aistudio.baidu.com/aistudio/education/group/info/16606)
#### B站视频链接（P5、P6）：[https://www.bilibili.com/video/BV1pp4y1871g](https://www.bilibili.com/video/BV1pp4y1871g)
#### 图像分割遥感项目实战：[https://aistudio.baidu.com/aistudio/projectdetail/1703449](https://aistudio.baidu.com/aistudio/projectdetail/1703449)

