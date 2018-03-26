# 异方差性

本章内容是介绍异方差现象、影响和处理方式。首先介绍如何识别或检验存在异方差性，如果不存在则OLS方法可以使用；如果存在异方差性，则应当如何纠正处理。

## 一、异方差及其影响

异方差性（heteroskedasticity\)是指不同解释变量的误差项的方差是不相同的，即$Var\(u\|x\_1,x\_2,\cdots,x\_k\)≠常数$ ，u的方差是随着解释变量而变化的。通俗的理解是每一行数据（记录）得到的误差是不同的。这个现象不符合多元线性回归的基本假设，即高斯-马尔科夫假定。因为异方差性使得经典OSL的假设不再成立。通过下图可以看出异方差性的形象展示：

![](/assets/cigs and income.png)

通过上图可以看出随着收入的提高香烟消费量的范围明显增加了，误差项增加了，所以说收入对香烟消费而言是异方差的。该图的绘制代码参照本节香烟消费的例子。

MLR1-6假定中只放松同方差假定，即数据存在异方差性，则OSL的结果基本还是成立的，只是估计参数的方差不再是无偏估计。计算出来的统计量不再服从t分布、卡方分布和F分布，导致OSL的结果的可信程度降低。

## 二、异方差性检验

判断OSL模型是否是异方差性最可靠的方法是通过专门的检验进行，本节介绍两种检验异方差性的检验，Breusch-Pagan检验和White检验。

### 1、Breusch-Pagan检验

Breusch-Pagan检验的步骤是：

1. 按照常规OSL计算得到残差平方$\hat{u}^2$。
2. 对$\hat{u}^2$和$x_1,x\_2,\cdots,x\_k$进行回归，得到新的R方$R^2_{u^2}$。
3. 构造F统计量利用F\(k,n-k-1\)计算p值检验。
4. 或者构造LM统计量利用$\chi^2\_k$ 检验。

通过例子“住房价格方程中的异方差性”示范Breusch-Pagan检验的方法（原书225页，例8.4，数据HPRICE1.xls）。

首先载入工具库，导入并整理数据，形成解释变量数据表data.exog。

```python
import pandas as pd
import statsmodels.api as sm
import numpy as np
from scipy import stats

data=pd.read_excel('d:/econometrics/hprice1.xls',header=None)
data.rename(columns={0:'price',3:'lotsize',4:'sqrft',2:'bdrms'},inplace=True)
data.exog=data[['lotsize','sqrft','bdrms']]
data.exog=sm.add_constant(data.exog)
```

第一步常规OSL模型拟合数据计算残差平方。

```python
osl_model=sm.OLS(data.price,data.exog)
osl_model_result=osl_model.fit()
osl_model_result.summary()
u2=osl_model_result.resid**2
```

第二步根据残差平方与解释变量的回归方程计算新的R方。

```python
osl_bp=sm.OLS(u2,data.exog) # 第二次回归
osl_bp_result=osl_bp.fit()
osl_bp_result.rsquared # R方
```

osl\_bp\_result的类型是statsmodels的一种对象RegressResults，该对象中包含很多方法可以自动计算常见的统计和计量经济指标。比如R方就可以通过“.rsquared”读取。

第三步计算统计量和相应p值

LM统计量可以根据公式计算$LM=nR^2\_{u^2}$，n是观察数据的数量。F统计量及其p值可以直接从RegressResults中读取。

```python
n=len(data)
LM=n*osl_bp_result.rsquared  # 构建LM统计量
pvalue_chi2=1-stats.chi2.cdf(LM,3) # 卡方分布p值
osl_bp_result.fvalue #F值
osl_bp_result.f_pvalue # F分布相应的p值
```

结果显示LM=14.1，对应自由的为3（3个解释变量）的卡方分布的p值=0.0028，这是一个非常小的数值，所以有很充足的理由拒绝原假设，即同方差性的假设，也就是说模型是异方差的。

直接读取F统计量=0.534，F分布的p值=0.002，同样拒绝原假设，结论与LM的卡方检验一致。

### 2、White检验

White检验的步骤：

1. 运行常规OSL模型，得到残差平方$\hat{u}^2$和拟合值平方$\hat{y}$。
2. 对拟合值及其平方进行OSL拟合，计算新R方$R^2\_{u^2}$。
3. 构造F统计量，利用F检验（自由度2和n-3）；或者构造LM统计量，利用卡方分布检验（自由度为2）。

继续上节例子（原书P227，例8.5，数据HPRICE1.xls），注意数据相同，解释变量和被解释变量取对数值。回归方程：$$log(price)=\beta_0+\beta_1log(lotsize)+\beta_2log(sqrft)+\beta_3bdrms$$

第一步：工具库载入同上一节，输入导入和预处理工作是通过新增一个空数据表，然后将取对数后的数据逐一增加到该数据表中，加入常数项后，形成解释变量数据表。

```python
data=pd.read_excel('d:/econometrics/hprice1.xls',header=None)
data.rename(columns={0:'price',3:'lotsize',4:'sqrft',2:'bdrms'},inplace=True)
data.exog=pd.DataFrame()
data.exog['log_lotsize']=np.log(data[['lotsize']])
data.exog['log_sqrft']=np.log(data[['sqrft']])
data.exog['bdrms']=data[['bdrms']]
data.exog=sm.add_constant(data.exog)
```

第二步：根据上述回归方程建立模型并拟合数据计算残差平方和拟合值。

```python
osl_model=sm.OLS(np.log(data.price),data.exog)
osl_model_result=osl_model.fit()
osl_model_result.summary()
u2=osl_model_result.resid**2  # 残差平方
osl_model_result.fittedvalues # 拟合值
```

拟合系数如下：

```python
  OLS Regression Results                                                      ===============================================================================
                  coef    std err          t      P>|t|      [95.0% Conf. Int.]
-------------------------------------------------------------------------------
const          -1.2970      0.651     -1.992      0.050        -2.592    -0.002
log_lotsize     0.1680      0.038      4.388      0.000         0.092     0.244
log_sqrft       0.7002      0.093      7.540      0.000         0.516     0.885
bdrms           0.0370      0.028      1.342      0.183        -0.018     0.092
==============================================================================
```

根据下属方程进行第二次OSL计算。

$$ \hat{u}^2=\delta_0+\delta_1\hat{y}+\delta_1\hat{y}^2+误差项 $$

White的原假设是模型是同方差的，即$H\_0: \delta\_1=0和\delta\_2=0$ 。

```python
n=len(data)
# 建立解释变量数据表
db=pd.DataFrame()
db['y1']=osl_model_result.fittedvalues
db['y2']=db['y1']**2
db=sm.add_constant(db)
# 构建模型并拟合数据
osl_white=sm.OLS(u2,db)
osl_white_result=osl_white.fit()
```

第三步：直接读取F检验p值，计算LM统计量和卡方检验p值。

```python
LM_white=n*osl_white_result.rsquared
osl_white_result.f_pvalue
pvalue_chi2=1-stats.chi2.cdf(LM_white,2)
```

White检验LM=3.45，自由度为2的卡方分布的p值=0.178，这是一个比较大的值，所以不能够拒绝原假设，所以模型是同方差的，常规OSL结果是可信的。

在看F检验的p值=0.183，这是更强的证据维持原假设，F检验和LM卡方检验结论是一致的。

和上节模型相比，同样的数据都是OSL为基础的模型但是异方差性差别很大，差异就在于对数化处理，这具有一般性意义，即对解释变量和被解释变量取对数能够消除异方差性。

## 三、异方差性处理

### 1、加权最小二乘估计

加权最小二乘估计是假定每条数据的方差是解释变量的某个函数，即$Var\(u\|x\)=\sigma^2h\(x\)$。h\(x\)是解释变量的函数。通常h\(x\)是未知的，所以我们需要先估计h\(x\)然后经过变换可以转化成为同方差模型。常用的方法是假定h\(x\)是所欲解释变量的线性函数，即：$$Var(u|x)=\sigma^2exp(\delta_0+\delta_1x_1+\delta_2x_2+\cdots+\delta_kx_k)$$

在以上假设的基础上一般的纠正异方差性的广义最小二乘估计步骤是：

1. 通过常规OSL得到残差$\hat{u}$。
2. 计算残差平方的对数值$log\(\hat{u}^2\)$。
3. $log\(\hat{u}^2\)$与$x\_1,x\_2,\cdots,x\_k$ 回归并得到拟合值$\hat{g}$。
4. 计算权重=$1/exp\(\hat{g}\)$

实例计算“对香烟的需求”（原书p235，例8.7，数据SOMKE.xls）

我们研究的回归方程是：$$cigs=\beta_0+\beta_1log(income)+\beta_2log(cigpric)+\beta_3educ+\beta_4age+\beta_5age^2+\beta_6restaurn$$

工具库载入和数据处理

```python
import pandas as pd
import statsmodels.api as sm
import numpy as np
from scipy import stats

data=pd.read_excel('d:/econometrics/smoke.xls',header=None)
data.rename(columns={5:'cigs',1:'cigpric',4:'income',0:'educ',3:'age',6:'restaurn'},inplace=True)
data.exog=pd.DataFrame()
data.exog['log_income']=np.log(data[['income']])
data.exog['log_cigpric']=np.log(data[['cigpric']])
data.exog['educ']=data['educ']
data.exog['age']=data['age']
data.exog['age2']=data['age']**2
data.exog['restaurn']=data['restaurn']
data.exog=sm.add_constant(data.exog)
```

先进行常规OSL分析，并进行异方差检验。

```python
osl_model=sm.OLS(data.cigs,data.exog)
osl_model_result=osl_model.fit()
osl_model_result.summary()
```

对结果进行初步分析，发现所有参数都不是统计上显著的。

```python
 OLS Regression Results  
===============================================================================
                  coef    std err          t      P>|t|      [95.0% Conf. Int.]
-------------------------------------------------------------------------------
const          -3.6398     24.079     -0.151      0.880       -50.905    43.625
log_income      0.8803      0.728      1.210      0.227        -0.548     2.309
log_cigpric    -0.7509      5.773     -0.130      0.897       -12.084    10.582
educ           -0.5015      0.167     -3.002      0.003        -0.829    -0.174
age             0.7707      0.160      4.813      0.000         0.456     1.085
age2           -0.0090      0.002     -5.176      0.000        -0.012    -0.006
restaurn       -2.8251      1.112     -2.541      0.011        -5.007    -0.643
==============================================================================
```

经过Breusch-Pagan检验该模型，LM统计量的卡方分布p值=0.000012，F分布p值=0.000015，都非常小，所以很明显具有异方差性。于是我们采用加权最小二乘估计进行异方差纠正处理。

通过本章开头的图示我们也可以看出cigs和income的异方差性，绘制代码如下：

```python
#异方差图示
import matplotlib.pyplot as plt
plt.scatter(data.income,data.cigs)
plt.title('cigs and income')
plt.xlabel('income')
plt.ylabel('cigs')
plt.show()
```

首先在常规OSL基础上得到残差平方对数值，根据步骤3所示回归方程求得拟合值，然后计算权重。

```python
u2_log=np.log(osl_model_result.resid**2) # 残差平方对数值
osl_gls=sm.OLS(u2_log,data.exog) #建立新的回归方程
osl_gls_result=osl_gls.fit()  # 拟合数据
g_hat=osl_gls_result.fittedvalues #得到你拟合值
h_hat=np.exp(g_hat) 
weights=1/h_hat # 计算权重
```

利用statsmodels自带的加权最小二乘法的工具进行拟合求出WLS估计的参数。

```python
wls_model=sm.WLS(data.cigs,data.exog,weights=weights) # WLS方法
wls_model.result=wls_model.fit()
wls_model.result.summary()
wls_model.result.f_pvalue
```

sm.WLS是自带的最小二乘估计方法，其中的参数weights就是我们刚刚计算的权重。回归参数：

```python
                            WLS Regression Results                               ===============================================================================
                  coef    std err          t      P>|t|      [95.0% Conf. Int.]
-------------------------------------------------------------------------------
const           5.6355     17.803      0.317      0.752       -29.311    40.582
log_income      1.2952      0.437      2.964      0.003         0.437     2.153
log_cigpric    -2.9403      4.460     -0.659      0.510       -11.695     5.815
educ           -0.4634      0.120     -3.857      0.000        -0.699    -0.228
age             0.4819      0.097      4.978      0.000         0.292     0.672
age2           -0.0056      0.001     -5.990      0.000        -0.007    -0.004
restaurn       -3.4611      0.796     -4.351      0.000        -5.023    -1.900
==============================================================================
```

与常规OSL参数相比参数绝对值都不同，当时正负号都相同。显著性角度分析，除了香烟价格（log\_ciggric）不显著外，其他参数都是统计显著的。进一步解读，收入\(（income）每增加10%，香烟消费量将增加1.295\*10%≈0.13\(支/每天\)，教育每增加一年香烟消费将每天减少0.46支，年龄是二次关系在大约42.9岁前年龄每增长1岁，香烟每天消费增加0.48支，之后年龄每增加1岁，香烟每天消费减少0.0056支，餐馆限制吸烟将减少消炎消费量。至于为什么香烟价格不显著，作者Wooldridge的解释是：

> 原因之一是香烟价格志穗样本中不同的州而变化，所以log\(cigpric\)的波动性比log\(income\)、educ和age都要小得多。



