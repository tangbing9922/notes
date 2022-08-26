转自博客[Git 远程操作详解](http://www.ruanyifeng.com/blog/2014/06/git_remote.html)
# Git 远程操作详解
    本文详细介绍Git远程操作
    
# 什么是Git？

git 是目前世界上最先进的分布式版本控制系统。


回顾之前使用Microsoft Word写长篇大论时，经常出现 想删除一个段落，
又害怕将来想恢复找不回来怎么办？有办法，先把当前文件“另存为……”一个新的Word文件，再接着改，改到一定程度，再“另存为……”一个新文件，这样一直改下去，最后你的Word文档变成了这样：

![fig1](./figure/word.jfif)

过了一周，你想找回被删除的文字，但是已经记不清删除前保存在哪个文件里了，只好一个一个文件去找，真麻烦。

看着一堆乱七八糟的文件，想保留最新的一个，然后把其他的删掉，又怕哪天会用上，还不敢删，真郁闷。

更要命的是，有些部分需要你的财务同事帮助填写，于是你把文件Copy到U盘里给她（也可能通过Email发送一份给她），然后，你继续修改Word文件。一天后，同事再把Word文件传给你，此时，你必须想想，发给她之后到你收到她的文件期间，你作了哪些改动，得把你的改动和她的部分合并，真困难。

于是你想，如果有一个软件，不但能自动帮我记录每次文件的改动，还可以让同事协作编辑，这样就不用自己管理一堆类似的文件了，也不需要把文件传来传去。如果想查看某次改动，只需要在软件里瞄一眼就可以，岂不是很方便？

简而言之， Git 是一个版本控制系统！

## Git的诞生

Linus在1991年创建了开源的Linux，从此，Linux系统不断发展，已经成为最大的服务器系统软件了。
随着Linux的壮大，代码库之大让Linus很难继续通过手工方式管理了，社区的弟兄们也对这种方式表达了强烈不满。
Linus花了两周时间自己用C写了一个分布式版本控制系统，这就是Git！

## 集中式 vs 分布式
Linus 一直痛恨的CVS及SVN都是集中式的版本控制系统，而Git是分布式版本控制系统，集中式和分布式版本控制系统有什么区别呢？

集中式版本控制系统，版本库是集中存放在中央服务器的，而干活的时候，用的都是自己的电脑，所以要先从中央服务器取得最新的版本，然后开始干活，干完活了，再把自己的活推送给中央服务器。中央服务器就好比是一个图书馆，你要改一本书，必须先从图书馆借出来，然后回到家自己改，改完了，再放回图书馆。

![Git2](./figure/Git2.jfif)

集中式版本控制系统最大的毛病就是必须 **联网才能工作**，如果在局域网内还好，带宽够大，速度够快，可如果在互联网上，遇到网速慢的话，可能提交一个10M的文件就需要5分钟!


## git 三大分区
1.工作区 ：在电脑里面能看到的目录

2.暂存区 ：英文名叫stage或index。一般存放在.git 目录下的index文件中，所以也把暂存区叫做索引

3.版本库 ：工作区有一个隐藏目录.git，就是git的版本库。

## 本地 git 上传文件 至 远程仓库(远程仓库为空)    

1. 在本地某文件夹 下 点击 git bash here

2. git init
   
3. git add . #将该目录下的文件添加到暂存区
 /#当对工作区修改(或新增的文件)执行git add命令时，暂存区的目录树被更新，同时工作区修改的文件内容被写入到对象库的一个新的对象中，该对象的ID被记录在暂存区的文件索引中。
4. git commit -m "修改情况(可改)"
\# 将暂存区内容添加到仓库中。
5. git push -u origin main

# 创建仓库命令
1.git init

2.git clone

## git init
初始化仓库
## git clone
拷贝一份远程仓库，也就是下载一个项目。
远程操作的第一步，通常便是从远程主机克隆一个版本库，此时就需要用到 **git clone** 命令

例如

`
git clone <版本库的网址>
`

比如 克隆jQuery的版本库

`
git clone https://github.com/jquery/jquery.git
`

该命令会在本地主机生成一个目录，与远程主机的版本库同名。如果要制定不同的目录名，可以将目录名作为 **git clone**命令的第二个参数。

`
git clone <版本库的网址> <本地目录名>
`

**git clone**支持多种协议，除了HTTPS以外，还支持SSH、Git、本地文件协议等，下面是一些例子

```
$ git clone http[s]://example.com/path/to/repo.git/
$ git clone ssh://example.com/path/to/repo.git/
$ git clone git://example.com/path/to/repo.git/
$ git clone /opt/git/project.git 
$ git clone file:///opt/git/project.git
$ git clone ftp[s]://example.com/path/to/repo.git/
$ git clone rsync://example.com/path/to/repo.git/
```

通常来说，Git 协议下载速度最快，SSH协议用于需要用户认证的场合
## git remote
为了便于管理 ，Git要求每个远程主机都必须指定一个主机名。 **git remote**命令就用于管理主机名。

不带选项的时候， **git remote** 命令列出所有远程主机。

使用**-v**选项可以查看远程主机的网址

```
git remote -v
克隆版本库的时候，所使用的远程主机自动被Git命名为**origin**。 

如果想用其他的主机名，需要使用**git clone**命令的-o 选项指定。

**git remote show**命令加上主机名，可以查看该主机的详细信息

git remote show <主机名>


git remote add <主机名> <网址>

用于添加远程主机

git remote rm <主机名>

用于删除远程主机

git remote rename <原主机名> <新主机名>
```

## git fetch
一旦远程主机的版本库有了更新(git 术语叫做commit)， 需要将这些更新取回本地， 这是就要用到**git fetch** 命令。

该命令用于远程获取代码库
该命令执行完后需要执行git merge 远程分支到你所在的分支
从远程仓库提取数据并尝试合并到当前分支

git merge

该命令就是在执行git fetch之后，紧接着执行git merge 远程分支到你所在的任意分支。

`
git fetch <远程主机名>
`

上面命令将某个远程主机的更新， 全部取回本地。

**git fetch**命令通常用来查看其他人的进程，因为它取回的代码对你本地的开发代码没有影响。

默认情况下，**git fetch**取回所有分支(branch)的更新。如果只想取回特定分支的更新， 可以指定分支名。
`
git fetch <远程主机名> <分支名>
`

所取回的更新，在本地主机上要用“远程主机名/分支名”的形式读取。 比如origin主机的master，就要用origin/master读取

现在GitHub新版本改用main分支

```
git branch -r
origin/master
git branch命令的-r选项，可以用来查看远程分支，-a选项查看所有分支
git branch -a
* master
  remotes/origin/master
```

取回远程主机的更新以后，可以在它的基础上, 使用git checkout 命令创建一个新的分支


## git pull
拉取 远程更新

git pull 命令用于从远程获取代码并合并本地的版本。

git pull 其实就是 git fetch 和 git merge FETCH_HEAD 的简写。


## git push
推送 更新
git push 命令用于从将本地的分支版本上传到远程并合并

```
git push <远程主机名> <本地分支名>:<远程分支名>

如果本地分支名与远程分支名相同，则可以省略冒号

git push <远程主机名> <本地分支名>

以下命令将本地的 master 分支推送到 origin 主机的 master 分支
git push origin master

如果本地版本与远程版本有差异，但又要强制推送可以使用 --force 参数：
git push --force origin master

一般不使用force参数

```
## git checkout


## git add

git add 命令可将该文件添加到暂存区。

文件修改后，我们一般都需要进行 git add 操作，从而保存历史版本。