转自博客[Git 远程操作详解](http://www.ruanyifeng.com/blog/2014/06/git_remote.html)
# Git 远程操作详解
    本文详细介绍5个Git远程操作

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
5. git push

## git clone
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

```
## git checkout


## git add

git add 命令可将该文件添加到暂存区。

文件修改后，我们一般都需要进行 git add 操作，从而保存历史版本。