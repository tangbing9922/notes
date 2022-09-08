# DeepLearning笔记

## 激活函数

### Relu

> Relu函数（Rectified Linear Unit，Relu）线性整流函数

一种人工神经网络中常用的激活函数,通常线性整流函数指：

![img](https://bkimg.cdn.bcebos.com/formula/ae9d12662d9e1073200f081659ff7ea3.svg)

而在神经网络中线性整流作为神经元的激活函数，定义了该神经元在线性变换

之后的非线性输出结果。换言之，对于进入神经元的来自上一层神经网络的输入向量x，使用线性整流激活函数的神经元会输出

![img](https://bkimg.cdn.bcebos.com/formula/24175eeaf4905a7acc3025fa7f3f660f.svg)

至下一层神经元或作为整个神经网络的输出。

### sigmoid

Sigmoid函数是一个在[生物学](https://baike.baidu.com/item/生物学/1358)中常见的[S型函数](https://baike.baidu.com/item/S型函数/19178062)，也称为[S型生长曲线](https://baike.baidu.com/item/S型生长曲线/5581189)。

在信息科学中，由于其单增以及反函数单增等性质，Sigmoid函数常被用作神经网络的[激活函数](https://baike.baidu.com/item/激活函数/2520792)，将变量映射到0,1之间。

![Sigmoid 曲线](https://bkimg.cdn.bcebos.com/pic/c9fcc3cec3fdfc03f23fbf16d73f8794a5c226dc?x-bce-process=image/resize,m_lfit,w_800,limit_1/format,f_auto)

igmoid函数也叫[Logistic函数](https://baike.baidu.com/item/Logistic函数/3520384)，用于隐层神经元输出，取值范围为(0,1)，它可以将一个实数映射到(0,1)的区间，可以用来做二分类。在特征相差比较复杂或是相差不是特别大时效果比较好。Sigmoid作为激活函数有以下优缺点：

优点：平滑、易于求导。

缺点：激活函数计算量大，反向传播求误差梯度时，求导涉及除法；反向传播时，很容易就会出现梯度消失的情况，从而无法完成深层网络的训练。

Sigmoid函数由下列公式定义

![img](https://bkimg.cdn.bcebos.com/formula/7e627f6301407c3d610d2c2cab711f3f.svg)

****

## 超参数选择

### 搜索空间

一般会事先确定一个超参数的搜索空间

例如下图(以图像分类为例)：

![image-20211228155734096](C:\Users\bing\AppData\Roaming\Typora\typora-user-images\image-20211228155734096.png)

exponentially adv.以指数方式

搜索空间的范围一般是依据经验给定的一些值的范围

### 搜索算法

而在确定搜索空间之后 超参数的搜索算法 有以下两大类

1.**Black-box** 黑盒类：每一次的训练任务都当作一个黑盒，每次挑选一组超参数丢给模型训练，根据训练得到的关键的性能衡量指标 判断超参数的选择好坏。不考虑 模型优化的 问题，好处是 能够适用任何机器学习算法。但每次都要训练完模型(可能开销比较大)

2.**Multi-fidelity**(现在更多讨论、使用)：核心思想是，现今的任务训练开销过大，可能需要 多卡 且 耗时，所以可以不用把整个数据集 跑完（不关心最后的精度等指标怎样， 只关心超参数之间的效果怎么样）。 做法如下：

​        . 对数据集 下采样（超参数如果在小数据集上效果较好的话，在完整数据集上效果也不会差）

​        . 将模型变小（SGD的超参数在resnet18上效果差不多的话，在resnet152上可能是不错的）

​        . 在训练时会对数据扫很多遍， 但是对于不好的超参数 来说。 它训练一遍就知道它的效果怎么样了， 所以不需要等到完全训练完，看到效果不好的，就及时停止。

​        . 以上三点 表达的 意思 是说，通过比较便宜 但 又 跟完整训练有关系的任务来近似一个值， 然后对超参数 进行排序。

总结 ： black-box 虽然会贵一点 但是 当 任务计算量比较小或 优化算法不知道的话，这个方法会比较好一点；Multi-fidelity知道一些任务的细节，可以将任务弄小一点，这样每次试验的时候成本没有那么高。 

Black-Box:
    Grid Search:
    Random Search:
    Bayesian Optimization:
    Simulated Annealing
    Genetic Algorithms
Multi-Fidelity:
    Modeling Learning Curve
    Bandit Based(Successive Halving & Hyperband) 

## 多层感知机MLP

### 单个神经元

神经网络中计算的基本单元是神经元，一般称作[节点]node或者[单元]unit。节点从其他节点接收输入，或者从外部源接受输入，然后计算输出。每个输入都有[权重w]，权重取决于其他输入的相对重要性。

### 前馈神经网络

前馈神经网络是最先发明也是最简单的人工神经网络。它包含了安排在多个层中的神经元(节点)。相邻层的节点有连接或者边(edge)。所有的连接都配有权重。

![image-20220111104449205](C:\Users\bing\AppData\Roaming\Typora\typora-user-images\image-20220111104449205.png)

​                                    图 一个前馈神经网络的例子

线性模型的限制：显然实际数据中很难满足线性的情况，所以线性模型有着很大的限制。

### 多层感知机

在网络中加入一个或多个隐藏层来克服线性模型的限制，使其能处理更普遍的函数关系类型。最简单的方法是将许多全连接层堆叠在一起。每一层都输出到上面的层，直到生成最后的输出。把前L-1层看作表示，把最后一层看作线性预测器。***这种架构通常称为多层感知机(multilayer perceptron),通常缩写为MLP。***

![image-20220111111141401](C:\Users\bing\AppData\Roaming\Typora\typora-user-images\image-20220111111141401.png)

与一般的前馈神经网络或者说线性神经网络，MLP主要就是**引入了非线性激活函数**！

线性仿射：

![image-20220111111638924](C:\Users\bing\AppData\Roaming\Typora\typora-user-images\image-20220111111638924.png)

引入非线性激活函数：

![image-20220111111714660](C:\Users\bing\AppData\Roaming\Typora\typora-user-images\image-20220111111714660.png)

## BN batch normalization原理学习
[Batch Normalization（BN层）详解 - 简书 (jianshu.com)](https://www.jianshu.com/p/b05282e9ca57)

# pytorch  learning

## pytorch计算图机制

### torch.Tensor

torch.Tensor.detach()

### torch shape相关

Tensor views

### optimizer

### 模型保存、加载相关

### torch.nn.module

#### forward()方法

我们在使用Pytorch的时候，模型训练时，不需要调用forward这个函数，只需要在实例化一个对象中传入对应的参数就可以自动调用 forward 函数。

```python
 class Module(nn.Module):
    def __init__(self):
        super().__init__()
        # ......

    def forward(self, x):
        # ......
        return x

data = ...#输入数据

# 实例化一个对象
model = Module()

#前向传播
model(data)

#而不是使用下面的
#model.forward(data)
```

实际上model(data)是等价于model.forward(data),  model(data)之所以等价于model.forward(data)，就是因为在类（class）中使用了__call__函数

### PyTorch 中 ModuleList 和 Sequential、ModuleDict: 区别和使用场景

Pytorch中的一些基础概念在构建网络的时候很重要，如nn.Module, nn.ModuleList, nn.Sequential

这些类 称之为 容器 **containers**，可以添加模块 (module)到它们之中。 容器之间很容易混淆，下面主要学习 nn.Modulelist 和 nn.Sequential 并 判断在什么时候用哪一个比较合适

#### nn.ModuleList 类

可以把任意 nn.Module 的 子类（如 nn.Conv2d，nn.Linear之类的）加到这个list里， 方法和Python自带的list一样，无非是extend，append等操作。  但是   不同于一般的list，加入到nn.ModulieList里面的module 是会自动注册到整个网络上的，同时module的parameters也会自动添加到整个网络中。  

具体看下面几个例子

```python
class net1(nn.Module):
    def __init__(self):
        super(net1, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10,10) for i in range(2)])
    def forward(self, x):
        for m in self.linears:
            x = m(x)
        return x

net = net1()
print(net)
# net1(
#   (modules): ModuleList(
#     (0): Linear(in_features=10, out_features=10, bias=True)
#     (1): Linear(in_features=10, out_features=10, bias=True)
#   )
# )

for param in net.parameters():
    print(type(param.data), param.size())
# <class 'torch.Tensor'> torch.Size([10, 10])
# <class 'torch.Tensor'> torch.Size([10])
# <class 'torch.Tensor'> torch.Size([10, 10])
# <class 'torch.Tensor'> torch.Size([10])
```

可以看到 这个网络包含 两个全连接层，它们的权重weights 和 偏置bias 都在这个网络之内。

接下来看第二个网络，它们使用Python自带的list:

```python
class net2(nn.Module):
    def __init__(self):
        super(net2, self).__init__()
        self.linears = [nn.Linear(10,10) for i in range(2)]
    def forward(self, x):
        for m in self.linears:
            x = m(x)
        return x

net = net2()
print(net)
# net2()
print(list(net.parameters()))
# []
```

显然 使用Python的list添加的全连接层和它们的parameters并没有自动注册到网络中。

当然，还是可以使用forward来计算输出结果。但是如果用net2实例化的网络进行训练的时候，因为这些层的parameters不在整个网络之中， 所以其网络参数也不会被更新，即无法更新。

综上，nn.ModuleList的作用为： 它是一个存储不同module，并且自动将每个module的parameters添加到网络的容器中。 值得注意的是，nn.ModuleList并没有定义一个网络，它只是将不同的模块存储在一起，模块之间没有什么先后顺序可言，比如：

```python
class net3(nn.Module):
    def __init__(self):
        super(net3, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10,20), nn.Linear(20,30), nn.Linear(5,10)])
    def forward(self, x):
        x = self.linears[2](x)
        x = self.linears[0](x)
        x = self.linears[1](x) 
        return x

net = net3()
print(net)
# net3(
#   (linears): ModuleList(
#     (0): Linear(in_features=10, out_features=20, bias=True)
#     (1): Linear(in_features=20, out_features=30, bias=True)
#     (2): Linear(in_features=5, out_features=10, bias=True)
#   )
# )
input = torch.randn(32, 5)
print(net(input).shape)
# torch.Size([32, 30])
```

根据net3的结果，可以看出ModuleList里面的顺序并不能决定什么，网络的执行顺序还是根据forward函数来决定的。

#### nn.Sequential

不同于nn.ModuleList,它已经实现了内部的forward方法，即里面的模块是按照顺序进行排列的。

```
模块将按照它们在构造函数中传递的顺序添加到其中。 或者，可以传入模块的 OrderDict。Sequential的 forward（） 方法接受任何输入，并将其转发到它包含的第一个模块。 然后，它按顺序将输出"链接"到每个后续模块的输入，最后返回最后一个模块的输出。
```

所以需要确保前一个模块的输出大小  和 下一个模块的 输入大小 一致，如下例所示：

```python
class net5(nn.Module):
    def __init__(self):
        super(net5, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(1,20,5),
                                    nn.ReLU(),
                                    nn.Conv2d(20,64,5),
                                    nn.ReLU())
    def forward(self, x):
        x = self.block(x)
        return x

net = net5()
print(net)
# net5(
#   (block): Sequential(
#     (0): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
#     (1): ReLU()
#     (2): Conv2d(20, 64, kernel_size=(5, 5), stride=(1, 1))
#     (3): ReLU()
#   )
# )
```

下面给出两个nn.Sequential初始化的例子

```python
# Example of using Sequential
model1 = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )
print(model1)
# Sequential(
#   (0): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
#   (1): ReLU()
#   (2): Conv2d(20, 64, kernel_size=(5, 5), stride=(1, 1))
#   (3): ReLU()
# )

# Example of using Sequential with OrderedDict
import collections
model2 = nn.Sequential(collections.OrderedDict([
          ('conv1', nn.Conv2d(1,20,5)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(20,64,5)),
          ('relu2', nn.ReLU())
        ]))
print(model2)
# Sequential(
#   (conv1): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
#   (relu1): ReLU()
#   (conv2): Conv2d(20, 64, kernel_size=(5, 5), stride=(1, 1))
#   (relu2): ReLU()
# )
```

 nn.Sequential 就是一个 nn.Module 的子类，也就是 nn.Module 所有的方法 (method) 它都有。并且直接使用 nn.Sequential 不用写 forward 函数，因为它内部已经帮你写好了。如果确定nn.Sequential里面的顺序是正确的需要的，那么完全可以直接使用nn.Sequential,不过这样做也失去了部分灵活性，因为不能自己制定forward函数里面的内容。

#### 到底用哪个

场景1： 当 网络中有很多相似或者重复的层，一般考虑用for 循环来创建它们，这时一般考虑用ModuleList

```python
class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.linears = nn.ModuleList([nn.linear(10,10) for _ in range(3)])
    def forward(self, x):
        for layer in self.linears:
            x = layer(x)
        return x
net1 = net()
print(net1)
# net6(
#   (linears): ModuleList(
#     (0): Linear(in_features=10, out_features=10, bias=True)
#     (1): Linear(in_features=10, out_features=10, bias=True)
#     (2): Linear(in_features=10, out_features=10, bias=True)
#   )
# )
```

当然 也可以使用Sequential实现

```python
class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.linear_list = [nn.Linear(10,10) for _ in range(3)]
        self.linears = nn.Sequential(*self.linear_list)
    def forward(self, x):
```

 值得注意的是  * 操作符，使用 * 操作符可以把一个list拆开成一个个独立的元素。同时这个list中的模块必须是按照想要的顺序来进行排列的。

**那么 * 操作符到底是什么作用呢？**以及为何有这样的作用

python 脚本中经常出现

```python
def test(*args, **kwargs):
```

其中 ***args **表示可变参数

#### 总结

ModuleList 就是一个储存各种模块的 list，这些模块之间没有联系，没有实现 forward 功能，但相比于普通的 Python list，ModuleList 可以把添加到其中的模块和参数自动注册到网络上。而Sequential 内的模块需要按照顺序排列，要保证相邻层的输入输出大小相匹配，内部 forward 功能已经实现，可以使代码更加整洁。

#### nn.Sequential(*list)的用法

python语法中，定义方法(函数)的参数时，即定义形参时，加*代表这个位置接收任意多个非关键字参数，转化成元组方式。

而如果*号在实参位置，代表的时将输入迭代器拆成一个个元素。

```python
def test_demo(*args):
    print(len(args))
    for i in args:
        print(i)


test_list = [9, 6, 1]
test_demo(test_list)
#1
#[9, 6, 1]
#但如果在输入实参时带上*，会把test_list打散，作为
#三个值写入
test_demo(*test_list)
#3
#9
#6
#1
```

而因为nn.Sequential()在定义时要求传入的必须是orderdict或者是一系列的模型nn.Module 的子类，所以遇到list必要用*号进行转化，否则会报错！



### nn与nn.functional 有什么区别及使用时机

torch.nn是一个定义的类，以class xx来定义，torch.nn.fuctional是一个函数接口，如nn.functional.Linear()

二者都继承于nn.Module

#### 二者的相同之处

nn.Xxx 和 nn. functional.xxx（如torch.nn.Conv2d和torch.nn.functional.conv2d）的 实际功能是相同的，运行效率也 几乎相同。

nn.functional.xxx是函数接口，而nn.Xxx是nn.functional.xxx的类封装，并且`**nn.Xxx**`都继承于一个共同祖先nn.Module 。即nn.Xxx除了具有nn.functional.xxx功能之外，内部还附带了`nn.Module`相关的属性和方法，例如`train(), eval(),load_state_dict, state_dict `等。

#### 不同之处

1） 两者的调用方式不同

nn.Xxx 需要先实例化并传入参数，然后以函数调用的方式调用实例化的对象并传入输入数据。

```python
import torch

inputs = torch.rand(64, 3, 244, 244)
conv = torch.nn.Conv2d(in_channels = 3, out_channels = 64,
                       kernel_size = 3, padding = 1)
out = conv(inputs)
```

`nn.functional.xxx`同时传入输入数据和weight, bias等其他参数 。

```python
weight = torch.rand(64,3,3,3)
bias = torch.rand(64) 
out = nn.functional.conv2d(inputs, weight, bias, padding=1)
```

**`nn.Xxx`继承于`nn.Module`， 能够很好的与`nn.Sequential`结合使用， 而`nn.functional.xxx`无法与`nn.Sequential`结合使用。**

```python
fm_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2)
  )
```

nn.Xxx 不需要你自己定义和管理weight；而nn.functional.xxx需要自己定义weight，每次调用的时候都需要手动传入weight，不利于代码复用。

使用nn.Xxx定义一个CNN

```python
class CNN(nn.Module):
        def __init__(self):
        super(CNN1, self).__init__()

        self.cnn1_weight = nn.Parameter(torch.rand(16,1,5,5))
        self.bias1_weight = nn.Parameter(torch.rand(16))

        self.cnn2_weight = nn.Parameter(torch.rand(32, 16, 5, 5))
        self.bias2_weight = nn.Parameter(torch.rand(32))

        self.linear1_weight = nn.Parameter(torch.rand(4 * 4 * 32, 10))
        self.bias3_weight = nn.Parameter(torch.rand(10))

    def forward(self, x):
        x = x.view(x.size(0), 1)
        out = F.conv2d(x, self.cnn1_weight, self.bias1_weight)
        out = F.relu(out)
        out = F.max_pool2d(out)

        out = F.conv2d(x, self.cnn2_weight, self.bias2_weight)
        out = F.relu(out)
        out = F.max_pool2d(out)

        out = F.linear(x, self.linear1_weight, self.bias3_weight)
        return out
```

使用nn.functional.xxx定义一个与上面相同的CNN

```python
import torch.functional as F
import torch
import torch.nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

            def __init__(self):
        super(CNN1, self).__init__()

        self.cnn1_weight = nn.Parameter(torch.rand(16,1,5,5))
        self.bias1_weight = nn.Parameter(torch.rand(16))

        self.cnn2_weight = nn.Parameter(torch.rand(32, 16, 5, 5))
        self.bias2_weight = nn.Parameter(torch.rand(32))

        self.linear1_weight = nn.Parameter(torch.rand(4 * 4 * 32, 10))
        self.bias3_weight = nn.Parameter(torch.rand(10))

    def forward(self, x):
        x = x.view(x.size(0), 1)
        out = F.conv2d(x, self.cnn1_weight, self.bias1_weight)
        out = F.relu(out)
        out = F.max_pool2d(out)

        out = F.conv2d(x, self.cnn2_weight, self.bias2_weight)
        out = F.relu(out)
        out = F.max_pool2d(out)

        out = F.linear(x, self.linear1_weight, self.bias3_weight)
        return out
```

上述两种方法定义得到的CNN一样，但是一般 具有学习参数的（例如 conv2d， linear， batch_norm）等  推荐采用nn.Xxx方式。  没有学习参数的（例如，maxpool， loss func， activation func）根据个人选择 二选一。  但是 关于 dropout， 有人强烈推荐使用 nn.Xxx  因为一般情况下 只有在训练阶段才进行dropout， 在eval阶段不会进行dropout。 使用nn.Xxx定义dropout，在调用model.eval()之后, model中所有的dropout layer都关闭。 但以`nn.function.dropout`方式定义dropout，在调用`model.eval()`之后并不能关闭dropout。

总结一下  nn.Xxx 是已经封装好了，nn.functional.xxx更底层更灵活自主性更高。

所以 一般还是用nn.Xxx就好.

### pytorch 模型定义时候是否可以不写forward()方法

经过测试，如果在定义模型的class中最后不写forward()方法，执行该模型的时候 会默认 按照 init()中 self 各模块的书写顺序执行forward顺序！

### nn.Module 子类中 self.register_buffer()方法介绍

首先回顾一下模型保存：torch.save(model.state_dict())
其中model.state_dict() 是一个存着模型各个部分参数的字典

在model中，我们需要更新其中的参数，训练结束将参数保存下来。但在某些时候，我们可能希望模型中的某些参数参数不更新（从开始到结束均保持不变），但又希望参数保存下来（model.state_dict() ），这时就会用到 **register_buffer()** 。

就是在内存中定一个常量，同时，模型保存和加载的时候可以写入和读出，一般情况下PyTorch将网络中的参数保存成orderedDict形式，参数包含两种，一种是模型中各种module含的参数，即nn.Parameter，当然可以在网络中定义其他的nn.Parameter参数，另一种就是buffer，前者每次optim.step会得到更新，而后者不会更新。


## python相关知识

### super函数

super函数简介

通常情况下，我们在子类中定义了和父类同名的方法，那么子类的方法就会覆盖父类的方法。而super关键字实现了对父类方法的改写(增加了功能，增加的功能写在子类中，父类方法中原来的功能得以保留)。也可以说，super关键字帮助我们实现了在子类中调用父类的方法

所以使用pytorch 创建神经网络的时候 往往需要继承 nn.Module类

```python
import torch.nn as nn
class CNN(nn.module):
    def __init__(self):
        super(CNN, self).__init__()
```

## 什么是对比学习 contrastive learning 