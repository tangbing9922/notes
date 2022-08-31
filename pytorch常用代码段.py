'''
整理一些常用的代码段
'''
import numpy as np
import torch

'''
1.可复现性
在不同的硬件设备上, 完全的可复现性保证不了, 即使随机种子相同.
但是在同一个设备上是可以保证的.
具体做法是,在程序开始的时候固定 torch 的随机种子, 同时也把 numpy 的随机种子固定
'''
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#2.显卡设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 如果需要指定多张显卡, 如0, 1号 显卡
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# 也可以在命令行运行代码时设置显卡：
#CUDA_VISIBLE_DEVICES=0,1 python train.py

#3.清除显存
torch.cuda.empty_cache()

# 二 