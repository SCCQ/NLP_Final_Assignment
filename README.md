# NLP_Final_Assignment

#### 任务介绍

实现中文评论领域的面向目标情感分类，基于[TNet模型](https://www.aclweb.org/anthology/P18-1087/)做修改，并将其应用到新的任务场景中

#### 语料库

语料库使用裁剪处理过的腾讯AI实验室的中文语料库，即embeddings目录下的TAChinese.txt

#### 数据集与测试集

这里使用的数据集是收集的百度外卖评论的语句，并做了合适标注

#### 目录介绍

| 名称         | 内容         |
| ------------ | ------------ |
| TNet         | 搭建模型结构 |
| datasets     | 数据集预测集 |
| display      | 运行结果展示 |
| embeddings   | 语料库       |
| logs         | 训练过程文件 |
| models       | 模型保存文件 |
| hparams.ini  | 相关参数     |
| requirements | 必要环境     |
| train.py     | 运行文件     |

#### 使用

首先pip install -r requirements.txt，在项目根目录运行python train.py，可根据数据集和设备调整路径与hparams.ini中的参数

#### 附录

其他详细内容请见提交的报告