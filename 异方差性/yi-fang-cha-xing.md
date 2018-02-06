# 异方差性

本章内容是介绍异方差现象及其影响，首先介绍如何识别或检验存在异方差性，如果不存在则OLS方法可以使用；如果存在异方差性，则应当如何处理。

## 一、异方差及其影响

什么是数据的异方差性

异方差性（heteroskedasticity\)是解释变量的方差是不相同的，即`Var(u|x)≠常数`。这个现象不符合多元线性回归的基本假设，即高斯-马尔科夫假定。因为异方差性使得经典OSL的假设不再成立。

MLR1-6假定中只放松同方差假定，即数据存在异方差性，则OSL的结果基本还是成立的，只是估计参数的方差不再是无偏估计。

异方差性的影响

## 二、异方差性检验

### 1、Breusch-Pagan检验

### 2、White检验

工具库载入同上一节，输入导入和预处理工作是通过新增一个空数据表，然后将取对数后的数据逐一增加到该数据表中，加入常数项后，形成解释变量数据表。

```py
data=pd.read_excel('d:/econometrics/hprice1.xls',header=None)
data.rename(columns={0:'price',3:'lotsize',4:'sqrft',2:'bdrms'},inplace=True)
data.exog=pd.DataFrame()
data.exog['log_lotsize']=np.log(data[['lotsize']])
data.exog['log_sqrft']=np.log(data[['sqrft']])
data.exog['bdrms']=data[['bdrms']]
```

## 三、异方差性处理

### 1、加权最小二乘估计

  


