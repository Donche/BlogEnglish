---
layout: post
title: 八数字推盘穷举算法
category: 实例练习
catalog: true
tags: 
  - 2017
  - Algorithm
---

# 1. 问题分析
八数字推盘（又名重排九宮）是Noyes Palmer Chapman在1870年代发明的，为九宫格布局，有八个方块和一个空位（供方块移动之用）。简而言之，就是试图  ~~（以最小步数）~~  将九宫格从一个状态转换成另外一个状态，如下图：   
![image1](https://pravj.github.io/images/states.jpg)   

在并没有对推盘问题进行深入研究的前提下，我们可以用穷举法解决。   
对于代码结构，考虑使用两个array存储初始状态和目标状态，将空位用零来表示，用IO流读取两个状态。而方块的移动用上下左右表示。我们以零为参考，每次将零与周边数字进行交换。利用名为swit的函数迭代计算过程并判断九宫格状态。考虑到方块移动过程中需要判断移动能否进行，于是增加名为exchange的内联函数，若交换可进行则返回true，反之返回false。   

# 2. main函数
main函数主要负责输入起始状态量，以及调用swit函数进行计算。我们设定最大步数为18步（当然八数字的最优解最多是31步，设成31应该更好）。使用direc记录每步的方向，用1、2、3和4来分别表示上下左右，0表示该步还没有运行。我们需要给swit里传入数组的始末状态、走过的每步的方向、当前所处的步数。
代码实现如下
```c++
int main(){
  array<int,9> a,target;
  cout <<"input initial numbers ( 3 * 3 ) :" << endl;
  int i = 0;
  while(i != 9)
    cin >> a[i++];
  cout <<"input target numbers ( 3 * 3 ) :" << endl;
  i = 0;
  while(i != 9)
    cin >> target[i++];
  array<int,18> direc{0};//1:up, 2:down, 3:left, 4:right
  int testTime = 0;
  if(a != target){
    if(!swit(a, direc, testTime, target))
      cout << "no solution find." << endl;
  }
  else
    cout <<"It's the same！" << endl;
  return 0;
}
```
* 对于上下左右，我是以零为标准的。也就是说，1是零向上走，所以是零上方的方块向下推。输出结果也同理

# 3. swit函数
这个函数的主要部分是一个循环，用来试验零向不同方向走的可能性。每循环一遍，标记0运动方向的变量n自加1，然后在其内部向下一步迭代。若某步达到目标，则会在下一次迭代函数的开始判断状态，随后输出结果并返回true。   

在循环过程中需要挪动方块时，使用标准库的swap函数。而与零互换的数字的下标变化量存储在offs中。例如offs[0]为-3，即说明将0向上移动一格（方向1）时，与零互相交换的数下标为零的下标-3。   

在迭代过程中，我们还应注意避免程序陷入无意义的循环。例如反复重复某两步。为此需要将每步与前一步进行对比，若将要循环的方向是上一步的相反方向则跳过此步。

```c++
bool swit(array<int,9> a, array<int,18> direc, int times, array<int,9> target){
  //check status
  if(a == target){
    for(auto dir : direc)
      cout << dir << " ";
    cout << endl;
    return true;
  }
  //check step number
  if(++times == 19)
    return false;
  //enter loop
  else{
    int offs[4] = {-3, 3, -1, 1}
    int pre;
    auto pzero = find(a.begin(), a.end(), 0);
    int n = direc[times-1];
    while(n != 4){
      int m = n;
      m += ((n % 2) ? 0 : 2);
      if(exchange(a,n+1) && pre != m){
        //
        swap(*pzero,*(pzero+offs[n]));
        direc[times-1] = ++n;
        if(swit(a, direc, times)){
          return true;
        }
        else{
          swap(*pzero,*(pzero+offs[n-1]));
        }
      }
      else{
        direc[times-1] = ++n;
      }
    }
      return false;
  }
}
```

# 4. exchange函数

这个函数输入当前九宫格状态和需要移动的方向，以此来判断交换是否可以进行。据此我们可以用switch-case 来具体地对每一个状态进行判断。如下：
```c++
inline bool exchange(array<int,9> &a, const int dir){
  auto pzero = find(a.cbegin(), a.cend(), 0);
  if (pzero == a.cbegin()){
    if(dir == 2 || dir == 4)
      return true;
    else
      return false;
  }
  if (pzero == a.cbegin()+1){
    if(dir == 2 || dir == 4 || dir == 3)
      return true;
    else
      return false;
    }
    //and so on...
  else
    return false;
}
```

这样写看起来是很蠢的，九个位置，一共花了大约六十行。如果万一需要写15数字推格呢？所以还是稍微改进一下

# 5. 改进

既然是整型与数组之间的对应关系，我们可以考虑使用map和set来解决这段代码。使用map来记录0所处位置和可以移动方向之间的对应关系，而用set记录可移动的方向。这样可以大大缩短代码行数，实现如下
```c++
inline bool exchange(array<int,9> &a, const int dir){
  auto pzero = find(a.cbegin(), a.cend(), 0);
  map<int, set<int>> map_exchange;
  map_exchange[0] = set<int>{2,4};
  map_exchange[1] = set<int>{2,3,4};
  map_exchange[2] = set<int>{2,3};
  map_exchange[3] = set<int>{1,2,4};
  map_exchange[4] = set<int>{1,2,3,4};
  map_exchange[5] = set<int>{1,2,3};
  map_exchange[6] = set<int>{1,4};
  map_exchange[7] = set<int>{1,3,4};
  map_exchange[8] = set<int>{1,3};
  auto vset = map_exchange.find(pzero - a.cbegin());
  if( vset != map_exchange.cend()){
    if( (vset->second).find(dir) != (vset->second).cend() )
      return true;
    else
      return false;
  }
  else
    return false;
}
```


# 改进？

把这两个版本的代码运行一下就会发现，第一个很显然快很多。一般情况下第一种得出结果只需要半秒左右，而第二个我还可以气定神闲地数好几个数。   
所以说，貌似对于这种数据量并不大的小程序来说，使用高级的数据结构反而会拖累代码运行速度。

Over.

---
# 04.13.2017补
突然发现还有编译器优化这种功能...在vs控制台下面启用编译器优化后果然速度变快了很多。第一种秒出结果，第二种依然会慢一点，大概半秒左右。但是在Debug时并不适合开编译器优化，会有一些风险。这个以后有机会专门写一篇吧。暂且留个参考链接，[MSDN Magazine Blog关于编译器的相关内容](https://msdn.microsoft.com/zh-cn/magazine/dn904673.aspx)
