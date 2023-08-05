# 待办事项
## 0905
- [x] 周三之前完成 0902讨论的问题报告
- [ ] 学习 量化 /  基于神经网络的LDPC(或其他信道编码)
[MVision/CNN/Deep_Compression/quantization at master · Ewenwan/MVision (github.com)](https://github.com/Ewenwan/MVision/tree/master/CNN/Deep_Compression/quantization)

下文的量化模块 可以参考， 有code
[AI+无线通信——Top7 (Baseline)分享与总结](https://blog.csdn.net/jasneik/article/details/115834361)

- [x] 阅读 L-DeepSC 量化操作 
[A Lite Distributed Semantic Communication System for Internet of Things | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/document/9252948)

- [ ] 无线通信的笔记本周整理完

## 0906
- [ ] 将 目前的一些量化的方案 和 信道 的方案 过一下
- [ ] 什么是率失真函数？
率失真函数： 在信源给定和满足保真度准则的前提下，存在一个函数，这个函数就是在给定失真范围内带约束条件转移概率的函数，且有一个极小值R，给定一个失真就对应一个最小码率（极小值），它们一一对应构成一个R(D)函数。信源概率分布可统计，对应视频编码所有序列像素值统计概率。失真度可人为定义，根据人眼视觉定义等，编码方法mpeg2,mpeg4,264,265,266都是在找这个信道（转移概率函数），针对特定信源的转移概率函数，即给定信源它有一个压缩极限.
- [ ] 通信中的星座图的概念弄清， 看肖老师和lite deepsc 论文中的星座图的设计
- [ ] DL笔记 添加 有关GAN 的概念
- [ ] 研究一下xy老师的论文和分享视频，有点意思
- [x] 打印设备标签

## 0908 
- [ ] 什么是完美的CSI和不完美的CSI？什么是发射端的CSI，接收端的CSI？
- [x] 即解决什么叫发送端的SNR，接收端的SNR
该问题其实是指发送端到基站的无线信道和基站到接收端的无线信道的状态不同

## 0913
- [ ] 联合信道编码和调制， 或者是基于神经网络的信道编码
Joint Channel Coding and Modulation via Deep Learning
"Turbo Autoencoder: Deep learning based channel code for point-to-point communication channels"-2019-NeurIPS

检索关键词  channel coding modulation

- [ ] 星座图设计的相关论文 , 星座图设计是不是可以理解为调制设计
- [ ] 总结往上数的两条， 本周需要读桌面的0913论文，DeepJSCC-Q 这个论文是joint source-channel coding + 量化 调制都有
- [ ] matlab awgn 设定发射功率 学习

## 0915
- [ ] 弄清楚0914讨论的16QAM和64QAM同一Ebno下的性能差原因， Eb指的是信息比特的能量吗？
- [ ] 开始弄清lite-DeepSC的代码和原理
感觉他这复信号的代码可以参考一下，量化的东西太杂了看着

## 0919
- [ ] Constellation Design for Deep Joint Source-Channel Coding 本文采用的量化去进行星座图设计，其代码来源Conditional Probability Models for Deep Image Compression(CVPR2018)
可以建议尹义然看一些图像压缩相关的文章，里面都涉及到量化操作，也就可以看作是星座图设计。
- [ ] 论文修改：1.语言精简且前后摘要结论中不要过多句式重复的句子 2. 增加power限定，修改噪声的表述0均值，以及设定不同的方差是为了增强模型对不同信道的鲁棒性。
![[figure/Pasted image 20220919174301.png]]

## 0920
- [x] lite-deepsc中averagemeter的代码和概念在BNN中首先提出[Binarized Neural Networks (nips.cc)](https://papers.nips.cc/paper/2016/file/d8330f857a17c53d217014ee776bfd50-Paper.pdf)
[BinaryNet.pytorch/utils.py at b99870af6e73992896ab5db5ea26b83d2adb1201 · itayhubara/BinaryNet.pytorch (github.com)](https://github.com/itayhubara/BinaryNet.pytorch/blob/b99870af6e73992896ab5db5ea26b83d2adb1201/utils.py#L86)

看看原论文里咋讲的
- [x] 增加系统复杂度的分析，
一篇WCNC领域联合主席的deepdl based IRS的论文里面的设定，也是功率归一化
![[figure/Pasted image 20220920153141.png]]

- [x] 借鉴DeepJSCC-Q 这个论文(这个团队的其他论文也要看)的一些表述，在是否转换成比特流和调制等等。
![[figure/Pasted image 20220920153629.png]]
下面这一段也很有意义：
![[figure/Pasted image 20220920153920.png]]

5个Fellow的论文...
Beyond Transmitting Bits: Context, Semantics, and Task-Oriented Communications
这算是篇综述，再看看
![[figure/Pasted image 20220920155404.png]]


有关语义保真度的定义：
### [Have your text and use it too! end-to-end neural data-to-text generation with **semantic fidelity**](https://arxiv.org/abs/2004.06577)
![[figure/Pasted image 20220920213513.png]]


## 0921 
- [ ] 什么是熵编码？
熵编码是无损编码

## 0922 
- [ ] 分层调制 是什么概念 [IEEE Xplore Full-Text PDF:](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=497113)


## 0925

- [x] 和老师讨论一下ICC-DeepJSCC-Q中的SNR那里的问题，他这发送和接收的SNR和信道的SNR，有三个加噪声的过程。

## 0927
- [x] 摘要有些要改的，参考0926还是25的自己的批注，还有contribution第二点有句话要改


## 0929

- [ ] 什么是扩散模型
用于从文本到语音的扩散模型
[yangdongchao/Text-to-sound-Synthesis: The source code of our paper "Diffsound: discrete diffusion model for text-to-sound generation" (github.com)](https://github.com/yangdongchao/Text-to-sound-Synthesis)

- [ ] 什么是对比学习？


## 1010
 - [ ]  将超参数单独写在一个class中，由于原始的deepsc的maxlength会变成31 ，这在后面做VQVAE的时候对超参数的选择不利，reshape的时候出现非整数，所以重新训练一个max_length = 32的DeepSC
 - [ ] 周四之前比较linear_quanti和VQ-VAE的性能，比较传统分离方法、无量化方法、有量化方法的性能，让尹义然尝试deep+传统的方式
 - [ ] 准备开题的东西，在周四周五之前备好材料和一些模糊的问题。
 - [ ] 看肖老师的主页视频和相关论文

  vae [变分自编码器（一）：原来是这么一回事 - 科学空间|Scientific Spaces (kexue.fm)](https://kexue.fm/archives/5253)

## 1011
- [ ] 了解什么是联邦学习


## 1102
- [ ] 本周论文：
阅读 一些画了upper bound 的论文：
如：202209_arxiv_Vector Quantized Semantic Communication System


## 1108
别人的功率功率归一化
[OFDM-guided-JSCC/channel.py at main · mingyuyng/OFDM-guided-JSCC (github.com)](https://github.com/mingyuyng/OFDM-guided-JSCC/blob/main/models/channel.py)
def Normalize(x, pwr=1):

'''

Normalization function

'''

power = torch.mean(x ** 2, (-2,-1), True)

alpha = np.sqrt(pwr/2)/torch.sqrt(power)

return alpha * x
因为OFDM两个信道所以根号下pwr/2？


关于h
h = tf.complex(

tf.random.normal([tf.shape(x)[0], 1], 0, 1 / np.sqrt(2)),

tf.random.normal([tf.shape(x)[0], 1], 0, 1 / np.sqrt(2)),

)
[ADJSCC/util_channel.py at e5332e95faf592aab9f440992de96029162dc7dd · alexxu1988/ADJSCC (github.com)](https://github.com/alexxu1988/ADJSCC/blob/e5332e95faf592aab9f440992de96029162dc7dd/util_channel.py#L51)
![[figure/Pasted image 20221108210124.png]]

1107之后是改了路损的训练结果


## 1121
- [ ] 
JSAC的论文代码非线性变换那篇。
[wsxtyrdd/NTSCC_JSAC22 (github.com)](https://github.com/wsxtyrdd/NTSCC_JSAC22)

## 1128

- [ ]  把 瑞利信道 的 SRD 和SD  模型 以及 融合的模型重新训练一下，之前的channel 有问题
- [ ]  把 有AF 模块的模型进行训练
效果不太好

## 1205

- [ ] 这周一周二至少把量化的模块加入，然后 把 AF模块 重新弄一下，实在不行就改改



## 0529

- [ ] 周一 把 论文的仿真图跑出来：0.85做阈值的，0.8做阈值的，SSC + SC的区分：SSC是QUANT和不QUANT，以及有无SC的

共4条线 加上 总是转发的方案 对比 SEE
- [ ] 论文内容更改在周四上午之前完成，每天晚上好好准备工作555
- [ ] 训练不同的K，即神经网络输出的维度 的 模型 考虑16，32，48，64