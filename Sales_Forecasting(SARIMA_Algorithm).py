import numpy as np
import pandas as pd
from pandas import to_datetime
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from prophet import Prophet
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
#一、导入数据
train = pd.read_csv('train.csv',
                    parse_dates=True,low_memory=False)
store = pd.read_csv('store.csv',low_memory=False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
print(f'train文件的标签列{train.index}')

#二、数据清洗和初步分析
print(f'train尺寸：{train.shape}')#(1017209, 8)

#1、增加变量
train['Date'] = pd.to_datetime(train['Date'])
train['Year'] = train['Date'].dt.year
train['Month'] = train['Date'].dt.month
train['Day'] = train['Date'].dt.day
train['WeekOfYear'] = train['Date'].apply(lambda x:x.isocalendar()[1])
train['SalePerCustomer'] = train['Sales']/train['Customers']
print(train['SalePerCustomer'].describe())#平均顾客消费9.5，最低是0，最高65

#2、初步分析：经验累积分布函数ECDF
"""
经验累积分布函数（Empirical Cumulative Distribution Function，简称 ECDF）：
基于样本数据来估计总体累积分布的函数。对于给定的一组样本数据，通过对样本中数据点从小到大进行排序，然后计算小于或等于每个特定值的数据点在样本总数中所占的比例。
直观呈现出数据在各个取值区间上的累计概率情况，帮助我们更好地了解数据的整体分布形态
"""
sns.set(style='ticks')
c = '#386B7F'
plt.figure(figsize=(12,6))

plt.subplot(311)
cdf = ECDF(train['Sales'])
plt.plot(cdf.x,cdf.y,label='data_models',color=c)
plt.xlabel('Sales')
plt.ylabel('ECDF')

plt.subplot(312)
cdf = ECDF(train['Customers'])
plt.plot(cdf.x,cdf.y,label='data_models',color=c)
plt.xlabel('Customers')
plt.ylabel('ECDF')

plt.subplot(313)
cdf = ECDF(train['SalePerCustomer'])
plt.plot(cdf.x,cdf.y,label='data_models',color=c)
plt.xlabel('SalePerCustomer')
plt.ylabel('ECDF')
plt.tight_layout()
plt.show()
"""
一、ECDF数据分布情况分析：
1、偏态分布：sales和CUSTOMER 80%集中在1000以下
2、接近20%的sales customer为0
二、零销售额原因预判：商店关门、没有顾客、数据记录问题、商品因素
"""

#3、零销售额原因分析和缺失值处理
#3.1 商店关门:导致销售额为0
close_stores = train[(train['Open'] == 0) & (train['Sales'] == 0)]
print(close_stores.head())
print(f'关店的数量\n{close_stores.shape}')#关店的数量(172817, 13)

#3.2 商店开门但是0销售额
open_zero_sales = train[(train['Open'] != 0) & (train['Sales'] == 0)]
print(open_zero_sales.head(5))
print(f'开店但零销售额数量\n{open_zero_sales.shape}')#开店但零销售额数量(54, 13)

#3.3 0人均销售额
zero_SalePerCustomer = train[(train['SalePerCustomer'] == 0) | (train['SalePerCustomer'].isna())]
print(zero_SalePerCustomer.head(5))
print(f'人均0销售额数量\n{zero_SalePerCustomer.shape}')#人均0销售额数量(172871, 13)
"""
一、零销售额原因分析：
1、日人均零销售额数量为172871，其中商店关门占172817，商店开门但是0销售额占54
二、缺失值处理：将关店和0销售额的数据剔除，不参加预测
"""

#3.4 剔除关店和0销售额情况，组成新的train数据
print('关店且0销售额的天数应该剔除，不参加预测')
train = train[(train['Open'] != 0) & (train['Sales'] != 0)]
print(f'剔除后新的train大小：\n{train.shape}')#(844338, 13)

#3.5 store数据集处理
print(f'store数据集预览\n{store.head()}')
#3.5.1 查看数据空值情况
null = store.isnull().sum()
print(f'store空值预览：\n{null}')
#CompetitionDistance、CompetitionOpenSinceMonth/Year和Promo2SinceWeek/Year、PromoInterval 存在空值

#3.5.2 CompetitionDistance缺失值处理
null_CompetitionDistance = store[pd.isnull(store['CompetitionDistance'])]
print(f'null_CompetitionDistance\n{null_CompetitionDistance}')

#3.5.2.1 ECDF查看分布情况，决定填充方式
sns.set(style='ticks')
c = '#386B7F'
plt.figure(figsize=(12,6))
cdf = ECDF(store['CompetitionDistance'])
plt.plot(cdf.x,cdf.y,label='store_ECDF',color=c)
plt.xlabel('CompetitionDistance')
plt.ylabel('ECDF')
plt.show()
"""
CompetitionDistance缺失值分析：
1、CompetitionDistance的空值数据打印可知，这部分数据是缺失了导致的空值
2、通过ECDF可知CompetitionDistance是偏态分布，所以用中位数填充空值
"""
#3.5.2.2 中位数填充
store['CompetitionDistance'].fillna(store['CompetitionDistance'].median(),inplace=True)

#3.5.3 CompetitionOpenSinceMonth、Year缺失值处理
null_CompetitionOpenSinceMonth = store[pd.isnull(store['CompetitionOpenSinceMonth'])]
print(f'null_CompetitionOpenSinceMonth\n{null_CompetitionOpenSinceMonth.head(10)}')
null_CompetitionOpenSinceYear = store[pd.isnull(store['CompetitionOpenSinceYear'])]
print(f'null_CompetitionOpenSinceYear\n{null_CompetitionOpenSinceYear.head(10)}')
null_dis_null_CompetitionOpenSinceMonth = null_CompetitionOpenSinceMonth[null_CompetitionOpenSinceMonth['CompetitionDistance'] != 0 ]
print(f'null_dis_null_CompetitionOpenSinceMonth\n{null_dis_null_CompetitionOpenSinceMonth.shape}')
null_dis_null_CompetitionOpenSinceYear = null_CompetitionOpenSinceYear[null_CompetitionOpenSinceYear['CompetitionDistance'] != 0 ]
print(f'null_dis_null_CompetitionOpenSinceYear\n{null_dis_null_CompetitionOpenSinceYear.shape}')#均为(354, 10)

#3.5.3.1 ECDF查看分布情况，决定填充方式
sns.set(style='ticks')
c = '#386B7F'
plt.figure(figsize=(12,6))

plt.subplot(211)
cdf = ECDF(store['CompetitionOpenSinceMonth'])
plt.plot(cdf.x,cdf.y,label='store_ECDF',color=c)
plt.xlabel('CompetitionOpenSinceMonth')
plt.ylabel('ECDF')

plt.subplot(212)
cdf = ECDF(store['CompetitionOpenSinceYear'])
plt.plot(cdf.x,cdf.y,label='store_ECDF',color=c)
plt.xlabel('CompetitionOpenSinceYear')
plt.ylabel('ECDF')
plt.tight_layout()
plt.show()
"""
CompetitionOpenSinceMonth、Year缺失值分析：
1、CompetitionOpenSinceMonth、Year的空值数据打印可知，这部分数据是缺失了导致的空值，同时CompetitionOpenSinceMonth、Year的空值354个，
这354个数据的CompetitionDistance均不为空值，所以不能全部按0来填充
2、通过ECDF可知CompetitionOpenSinceMonth、Year是偏态分布，所以用中位数填充空值
"""
#3.5.3.2 中位数填充
store['CompetitionOpenSinceMonth'].fillna(store['CompetitionOpenSinceMonth'].median(),inplace=True)
store['CompetitionOpenSinceYear'].fillna(store['CompetitionOpenSinceYear'].median(),inplace=True)

#3.5.4 Promo2SinceWeek/year、PromoInterval 缺失值处理
null_Promo2SinceWeek = store[pd.isnull(store['Promo2SinceWeek'])]
print(f'null_Promo2SinceWeek\n{null_Promo2SinceWeek.head()}')
null_Promo2SinceYear = store[pd.isnull(store['Promo2SinceYear'])]
print(f'null_Promo2SinceWeek\n{null_Promo2SinceYear.head()}')
null_PromoInterval = store[pd.isnull(store['PromoInterval'])]
print(f'null_Promo2SinceWeek\n{null_PromoInterval.head()}')
#3.5.4.1 检查空值情况下，Promo2不为0的情况
Promo2_is_1_null_Promo2SinceWeek =null_Promo2SinceWeek[null_Promo2SinceWeek['Promo2'] != 0]
Promo2_is_1_null_Promo2SinceYear =null_Promo2SinceYear[null_Promo2SinceYear['Promo2'] != 0]
Promo2_is_1_null_PromoInterval =null_PromoInterval[null_PromoInterval['Promo2'] != 0]
print(f'Promo2_is_1_null_Promo2SinceWeek\n{Promo2_is_1_null_Promo2SinceWeek.shape}')
print(f'Promo2_is_1_null_Promo2SinceYear\n{Promo2_is_1_null_Promo2SinceYear.shape}')
print(f'Promo2_is_1_null_PromoInterval\n{Promo2_is_1_null_PromoInterval.shape}')
"""
Promo2SinceWeek/year、PromoInterval缺失值分析：
1、Promo2SinceWeek/year、PromoInterval的空值数据打印可知，这部分数据是缺失了导致的空值，同时Promo2SinceWeek/year、PromoInterval的空值均为544
这544个数据为空值，但Promo2均为0也就是没做持续促销，由于Promo2与三个特征强相关，所以全部按0来填充
"""
#3.5.4.2 0填充
store['Promo2SinceWeek'].fillna(0,inplace=True)
store['Promo2SinceYear'].fillna(0,inplace=True)
store['PromoInterval'].fillna(0,inplace=True)
#查看处理完后的store数据集
null_1 = store.isnull().sum()
print(f'store空值预览：\n{null_1}')

#3.6 train、store数据合并
print('以store列为索引合并train和store数据集')
train_store = pd.merge(train,store,how='inner',on='Store')
print(f'train_store尺寸：{train_store.shape}')#(844338, 22)全部都匹配上了
print(train_store.head())

#4、特征分析
#4.1 Store types:透视分析商店类型与销售额关系
StoreType_Sales = train_store.groupby('StoreType')['Sales'].describe()
print(f'StoreType_Sales\n{StoreType_Sales}')
StoreType_Sales_cust_sum = train_store.groupby('StoreType')[['Sales','Customers']].sum()
print(f'StoreType_Sales_cust_sum\n{StoreType_Sales_cust_sum}')
"""
Store types分析：将StoreType按Sales透视发现，b类型商店的平均销售额最高，但是对Sales和Customers进行求和透视发现，a和d类型商店分别在总销售
和顾客数排前两名，商店b在销售和顾客规模排名最后,原因是商店b的数量远低于其他店铺（只是a店铺规模的3.2%)
"""

#4.2 trends
#4.2.1 sales trends:商店类型和促销对sales趋势影响
sns.catplot(data=train_store,
            x='Month',
            y='Sales',
            col='StoreType',
            row='Promo',
            palette='plasma',
            hue='StoreType',
            kind='point')

#4.2.2 sales trends:星期几和商店类型对sales趋势影响
sns.catplot(data=train_store,
            x='Month',
            y='Sales',
            col='DayOfWeek',
            row='StoreType',
            palette='plasma',
            hue='StoreType',
            kind='point')

#4.2.3 Customers trends:商店类型和促销对Customers趋势影响
sns.catplot(data=train_store,
            x='Month',
            y='Customers',
            col='StoreType',
            row='Promo',
            palette='plasma',
            hue='StoreType',
            kind='point')

#4.2.4 sale per customer trends:商店类型和促销对SalePerCustomer趋势影响
sns.catplot(data=train_store,
            x='Month',
            y='SalePerCustomer',
            col='StoreType',
            row='Promo',
            palette='plasma',
            hue='StoreType',
            kind='point')

plt.tight_layout()
plt.show()
"""
trends散点图分析
一、商店类型和促销对sales趋势影响：
1、商店类型和促销不会影响整体的销售额趋势，但是有促销能提升销售额规模
2、同时12月份圣诞节销售额在不同店铺都会很高，后面要专门进行时间序列分析季节性和趋势
二、星期几和商店类型对sales趋势影响：
1、周一不同店铺的销售额会高于其他日子
2、c类型店铺周日不开业，d类店铺11月的周日不开业
三、商店类型和促销对Customers趋势影响：商店类型和促销不会影响整体的客流趋势，但是有促销能提升客流规模
四、商店类型和促销对SalePerCustomer趋势影响：同样不会客单销售额有趋势影响，但是促销能提升客单销售额规模，其中店铺d的客单销售额最高，无促销10元，有促销12元
"""

#4.3 竞争和促销分析 competition Promo2
#竞争店铺开业时间（按月）
train_store['CompetitionOpen'] = 12*(train_store['Year'] - train_store['CompetitionOpenSinceYear']) + \
                                 (train_store['Month'] - train_store['CompetitionOpenSinceMonth'])
#Promo2持续促销持续时间
train_store['Promo2Open'] = 12*(train_store['Year'] - train_store['Promo2SinceYear']) + \
                            (train_store['WeekOfYear'] - train_store['Promo2SinceWeek'])/4

Promo2_Compet = train_store.loc[:,['StoreType','Sales','Customers','Promo2Open','CompetitionOpen']].groupby('StoreType').mean()
print(Promo2_Compet)
"""
竞争和促销分析
1、销售额和客流规模最大的a类型商店，在持续促销时间远低于最高的b,在竞争对手开业时间上也低于b
2、b类型商店的日均销售额和客流最高，同时在持续促销时间和竞争也是最高的
"""
"""
一、分类型数据和数值型数据分析方式
1、相关性矩阵分析通常用于数值型数据，因为它基于数学计算（如皮尔逊相关系数、斯皮尔曼秩相关系数等）来衡量变量之间的线性关系。
数值型数据（定量数据）有连续的数值范围，可以计算出它们之间的相关程度。

2、分类数据（定性数据）通常不适用于相关性矩阵分析，因为相关性系数是针对连续变量设计的，而分类变量是由离散的类别组成，没有连续的数值范围，因此无法计算传统意义上的相关性系数。
分类数据，以下是一些常用的分析方法：
画图：使用条形图、饼图、堆叠条形图等来可视化分类变量的分布或不同类别之间的关系。
透视表（Pivot Tables）：通过透视表可以汇总数据，查看不同类别之间的统计信息，如计数、总和、平均值等。
交叉表（Cross-tabulation）：用于分析两个或多个分类变量之间的关系，常用于计算列联表（contingency tables）。
卡方检验（Chi-Square Test）：用于检验两个分类变量之间是否独立。
逻辑回归（Logistic Regression）：虽然不是专门用于分类数据的分析，但逻辑回归可以用来预测分类因变量（通常是二分类）与一组自变量之间的关系。
机器学习模型：分类数据可以通过独热编码转换为数值型数据，然后用于训练机器学习模型。
总的来说，分类数据通常需要不同的统计方法或数据可视化技术来进行分析，而不是使用相关性矩阵。
"""
#4.4 相关性分析（聚焦数值型数据）
print('检查train_store数据类型，非数值不能进行相关性矩阵构建')
print(train_store.dtypes)
"""
StateHoliday、StoreType、Assortment、PromoInterval都是分类变量，处理方式分析
1、独热编码（One-Hot Encoding）适合无顺序的类别数据，因为它将每个类别值转换成一个独立的二进制特征，这样就不会强加任何隐含的顺序。
2、标签编码（Label Encoding）适用于有自然顺序（时间顺序）的类别数据，它将每个类别映射到一个整数，并保留这些类别之间的顺序信息。
3、StateHoliday、StoreType、Assortment都是无顺序类别信息，适合单独分析；PromoInterval里面是是月份是有顺序的，但是由于每个数据都是多个月份，
所以这里相关性矩阵计算统一剔除这四列,Open代表是否开门在这里也剔除
"""
#4.4.1 剔除open和4个分类标签，计算特征相关性矩阵
corr_all = train_store.drop(['Open','StateHoliday','StoreType','Assortment','PromoInterval'],axis=1).corr()
mask = np.zeros_like(corr_all,dtype=np.bool_)
mask[np.triu_indices_from(mask)] = True
f,ax = plt.subplots(figsize=(11,9))
sns.heatmap(corr_all,mask=mask,square=True,linewidths=.5,ax=ax,cmap='BuPu')
plt.tight_layout()
plt.show()
"""
相关性矩阵热力图分析
一、正相关性
1、Sales和Customers：客流量和销售额强正相关
2、Promo与Sales、Customers：促销与销售额和客流正相关
二、负相关性
1、Promo2与Sales、Customers：持续促销会导致销售额和客流下降
"""

#4.4.2 Promo和Promo2对销售额影响
sns.catplot(data=train_store,
            x='DayOfWeek',
            y='Sales',
            col='Promo',
            row='Promo2',
            hue='Promo2',
            palette='RdPu',
            kind='point')
plt.tight_layout()
plt.show()
"""
Promo和Promo2对销售额影响
1、当完全没有促销的时候（Promo、Promo2都为0），销售额在周日达到峰值，前面分析可知c类店铺周日不开门，所以这里贡献销售额主要是abd店铺
2、在有促销但是没有持续促销Promo2的时候，销售额峰值出现在周一（同时有促销和持续促销也呈现这个趋势）
2、当只有持续促销Promo2的时候，整体销量低于其他三种情况，对销量提升影响不明显，这个在热度图里也有显示
"""

"""
一、数据分析总结
1、a类型店铺销售总额和客流总量最高
2、d类店铺的客均销售额最高，顾客购买额具有优势，其中在有促销但没持续促销的时候，客户购买额更高；公式可以考虑给d类店铺提供更广的商品多样性
3、b类店铺的客均销售额和销售总额规模最低，但是平均每日的销售额和客流是最高的，说明客户购买的物品价值较低，销售规模低原因之一是是因为店铺数量较少，
b店铺在持续促销和竞争上都是做的最多和竞争最大的，在客流上和购买转化上具有潜力和优势
4、促销获得有利于提升销售额和客流，在有促销的时候顾客会在周一买的更多，如果没有任何促销会在周日买的更多
5、如果只有持续促销，并不能有效提升销售额
"""

#二、查看不同类型店铺的销售额时间序列趋势Seasonality
#1、按店铺类型分组，查看销售数据的汇总情况
grouped = train_store.groupby('StoreType').agg({'Sales':['mean','sum','count',lambda x:x.sum()/train_store['Sales'].sum()]})
print(f'按店铺类型分组查看销售汇总情况\n{grouped}')

#2、分店铺绘制时间序列趋势图
"""
按日来显示太密集，将日期按周聚合，先用dt.to_period('W')转换成时间区间例如 2024-01-01/2024-01-07
然后用apply(lambda x:x.start_time)将开始时间2024-01-01作为同一个周的表示方式，然后按周来分组画图
"""
train_store['year_week'] = train_store['Date'].dt.to_period('W').apply(lambda x:x.start_time)
StoreType_sale_trends = train_store.groupby(['StoreType','year_week']).agg({'Sales':'sum'}).reset_index()
StoreType_sale_trends.columns = ['StoreType','year_week','Sales_sum']
print(StoreType_sale_trends.head())

# Matplotlib 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
store_types = StoreType_sale_trends['StoreType'].unique()

fig,ax = plt.subplots(4,1,figsize=(10,6))
for i,type in enumerate(store_types):
    subset = StoreType_sale_trends[StoreType_sale_trends['StoreType'] == type]
    ax[i].plot(subset['year_week'],subset['Sales_sum'],label=f'店铺类型{type}')
    ax[i].set_title(f'店铺类型{type}的销售额趋势')
    ax[i].set_xlabel('日期')
    ax[i].set_ylabel('销售额')
    ax[i].legend()#图例
plt.tight_layout()
plt.show()
"""
一、表面趋势分析：
1、画图可知acd店铺销售趋势较为一直，b店铺由于销售额整体较小，波动更明显但是整体趋势也相差不大
2、这种方法无法分离趋势（trend）、季节性（seasonal）和残差（residual），只能观察到表面趋势
深入分析长期趋势或周期性规律，seasonal_decompose 将时间序列分解为趋势 (trend)、季节性 (seasonal)、残差 (residual)三部分
可以清晰看到长期变化趋势和季节性变动
"""

#三、seasonal_decompose深入分析时间序列趋势
StoreType_sale_trends['Sales_sum'] = StoreType_sale_trends['Sales_sum']*1.0#转换维浮点数
#提取不同类型店铺销售额数据
sale_a = StoreType_sale_trends[StoreType_sale_trends['StoreType']=='a'].set_index('year_week')['Sales_sum']
sale_b = StoreType_sale_trends[StoreType_sale_trends['StoreType']=='b'].set_index('year_week')['Sales_sum']
sale_c = StoreType_sale_trends[StoreType_sale_trends['StoreType']=='c'].set_index('year_week')['Sales_sum']
sale_d = StoreType_sale_trends[StoreType_sale_trends['StoreType']=='d'].set_index('year_week')['Sales_sum']
print(sale_a.head())

c = 'blue'
f,(ax1,ax2,ax3,ax4) = plt.subplots(4,1,figsize=(12,13))
decomposition_a = seasonal_decompose(sale_a,model='additive',period=52)#一年52周
decomposition_a.trend.plot(color=c,ax=ax1)#trend是提取分解函数里面的趋势，seasonal、resid就是提取季节性和残差
ax1.set_title('店铺类型 a 的销售额趋势')

decomposition_b = seasonal_decompose(sale_b,model='additive',period=52)
decomposition_b.trend.plot(color=c,ax=ax2)
ax2.set_title('店铺类型 b 的销售额趋势')

decomposition_c = seasonal_decompose(sale_c,model='additive',period=52)
decomposition_c.trend.plot(color=c,ax=ax3)
ax3.set_title('店铺类型 c 的销售额趋势')

decomposition_d = seasonal_decompose(sale_d,model='additive',period=52)
decomposition_d.trend.plot(color=c,ax=ax4)
ax4.set_title('店铺类型 d 的销售额趋势')

plt.tight_layout()
plt.show()
"""
用seasonal_decompose进行时间序列分解，发现b店铺整体销售额呈现上升趋势，c店铺销售额2014年7月跌倒谷底后目前上升接近历史最高点
店铺a和d均出现销售额降低，其中a下降最大
"""

#四、自相关和偏自相关：Autocorrelation Function (ACF)  Partial Autocorrelation Function (PACF)
"""
两个函数核心是拟合ARIMA模型时选择合适的模型参数
一、ACF作用
1、可以发现时间序列数据中是否存在显著的自相关性，以及自相关性会持续到多少滞后阶（lags）
2、ACF图用来显示不同滞后期的 整体相关性，帮助识别周期性和趋势。
二、PACF作用：PACF 测量的是时间序列与其特定滞后项之间的相关性，但去除了中间滞后项对相关性的影响。
1、有助于识别时间序列中真正重要的滞后项，而不是中间项的间接影响。
2、PACF图用来展示滞后期的 独立相关性，帮助确定AR模型中的滞后期数量。
三、由于竞赛要求预测每天的数据，所以这里调整为每天
"""
StoreType_sale_dayTrends = train_store.groupby(['StoreType','Date']).agg({'Sales':'sum'}).reset_index()
StoreType_sale_dayTrends.columns = ['StoreType','Date','Sales_sum']
print(StoreType_sale_dayTrends.head())

sale_a = StoreType_sale_dayTrends[StoreType_sale_dayTrends['StoreType']=='a'].set_index('Date')['Sales_sum']
sale_b = StoreType_sale_dayTrends[StoreType_sale_dayTrends['StoreType']=='b'].set_index('Date')['Sales_sum']
sale_c = StoreType_sale_dayTrends[StoreType_sale_dayTrends['StoreType']=='c'].set_index('Date')['Sales_sum']
sale_d = StoreType_sale_dayTrends[StoreType_sale_dayTrends['StoreType']=='d'].set_index('Date')['Sales_sum']
c = 'blue'
plt.figure(figsize=(12,8))
#ACF PACF分析和可视化
plt.subplot(421)
plot_acf(sale_a,lags=50,ax=plt.gca(),color=c)
plt.title('a店铺的ACF')
plt.subplot(422)
plot_pacf(sale_a,lags=50,ax=plt.gca(),color=c)
plt.title('a店铺的PACF')

plt.subplot(423)
plot_acf(sale_b,lags=50,ax=plt.gca(),color=c)
plt.title('b店铺的ACF')
plt.subplot(424)
plot_pacf(sale_b,lags=50,ax=plt.gca(),color=c)
plt.title('b店铺的PACF')

plt.subplot(425)
plot_acf(sale_c,lags=50,ax=plt.gca(),color=c)
plt.title('c店铺的ACF')
plt.subplot(426)
plot_pacf(sale_c,lags=50,ax=plt.gca(),color=c)
plt.title('c店铺的PACF')

plt.subplot(427)
plot_acf(sale_d,lags=50,ax=plt.gca(),color=c)
plt.title('d店铺的ACF')
plt.subplot(428)
plot_pacf(sale_d,lags=50,ax=plt.gca(),color=c)
plt.title('d店铺的PACF')

plt.tight_layout()
plt.show()
"""
ACF PACF可视化分析
1、每个图表都呈现的2个特点：时间序列具有非随机性（当前数据和滞后期的数据之间存在显著的相关性），滞后 1 阶的相关性较高
2、a、b、d店铺：都呈现出季节性特征，对于a类型店铺呈现出周的趋势,在8、15、22、29、36、43、50都出现正的峰值；
b和d类型店铺类似也出现周的趋势
3、c类型店铺较为复杂，看起来每个观测值都与其相邻的观测值存在相关性，整体滞后正值峰值出现在13、24、36、48，并且收敛
"""


#五、用SARIMA进行时间序列分析和预测
# Matplotlib 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号
#1、训练数据处理
#1.1 挑选销售规模最大的店铺
choose_store = train_store.groupby('Store')['Sales'].sum()
choose_store = choose_store.sort_values(ascending=False)#降序
#print(choose_store)#b店铺262 销售额最大

#2.2 选择262店铺的数据
sales = train_store[train_store['Store']==262].loc[:,['Date','Sales']]
sales = sales.sort_index(ascending=False)#降序，2013-2015
print(f'sale数据\n{sales.head()}')
#print(sales.dtypes)#Date必须是datetime64格式

#2.3 普通ACF和PACF:order (p, d, q) 参数依据
"""
普通ACF和PACF设置适中的滞后期（如 30 或 60）来分析数据的短期自相关性
"""
c = 'blue'
plt.figure(figsize=(12,8))

plt.subplot(211)
plot_acf(sales['Sales'],lags=50,ax=plt.gca(),color=c)#只需要传递销售额列
plt.title('262店铺ACF')
plt.subplot(212)
plot_pacf(sales['Sales'],lags=50,ax=plt.gca(),color=c)
plt.title('262店铺PACF')
"""
p 表示 自回归（AR） 部分的阶数，通常通过 PACF（偏自相关函数） 图来确定
q 表示 滑动平均（MA） 部分的阶数，通常通过 ACF（自相关函数） 图来确定
都显示在1阶的时候达到峰值，pq先设置1
"""


#2.4 季节性ACF 和 PACF 图：seasonal_order (P, D, Q, s)参数依据
"""
季节性 ACF 和 PACF 图 用于选择季节性部分的 P、Q 和 D，滞后期应选择季节性周期的倍数（例如，365 天对于年周期，7 天对于周周期）
"""
plt.figure(figsize=(12,8))

plt.subplot(211)
plot_acf(sales['Sales'],lags=365,ax=plt.gca(),color=c)#只需要传递销售额列
plt.title('262店铺季节性ACF')
plt.subplot(212)
plot_pacf(sales['Sales'],lags=365,ax=plt.gca(),color=c)
plt.title('262店铺季节性PACF')

plt.tight_layout()
plt.show()

"""
季节性 P：自回归（AR） 部分的阶数 根据 季节性 PACF 图选择。如果季节性滞后期的偏自相关显著且在某个位置截断，选择该阶数。
季节性 Q：滑动平均（MA） 部分的阶数  根据 季节性 ACF 图选择。如果季节性滞后期的自相关显著且在某个位置截断，选择该阶数。
季节性差分 D：通常根据季节性 ADF 检验结果来选择，判断数据是否有季节性单位根。如果有季节性趋势，可以选择 D=1，否则为 D=0

季节性ACF和PACF都显示在1阶的时候达到峰值，pq先设置1，两个图都没有明显截尾现象（即没有在某一阶滞后后迅速衰减到 0 附近），不容易直接确定阶数。
"""

#3、ADF 测试（单位根检验）：决定d差分阶数
"""
d 表示差分阶数，目的是使数据平稳。如果数据的时间序列呈现趋势或季节性模式，通常我们需要对数据进行差分，使其变为平稳序列
"""
result = adfuller(sales['Sales'])
print(f'ADF Statistic:{result[0]}')#ADF统计量值
print(f'p-value:{result[1]}')
print(f'Critical Values:{result[4]}')#临界值 1%、5%、10% 三个显著性水平下的临界值。

if result[1] < 0.05:#p值小于0.05就是平稳的
    print('数据是平稳的，无需差分')
else:
    print('数据是非平稳的，需要差分')#结果数据平稳不需要差分

#3.1 季节性差分（Seasonal Differencing）
"""
从 ACF 和 PACF 图判断季节性平稳性
季节性 ACF 图分析
季节性 ACF 图来看，图中的自相关系数并没有在某一阶后迅速衰减到 0 附近，而是呈现出较为缓慢的衰减趋势，这暗示数据可能存在季节性非平稳性。
季节性 PACF 图分析
季节性 PACF 图来看，其系数同样没有明显的截尾现象，而是逐渐衰减，这进一步支持了数据可能存在季节性非平稳性的推测。
"""
sales_seasonal_diff = sales['Sales'] - sales['Sales'].shift(365)
# 一阶季节性差分（周期为 365）,sales['Sales'].shift(365)就是数据向前移动365天（季节性变化通常是以 年 为周期的）
sales_seasonal_diff.dropna(inplace=True)#去掉缺失值
#print(sales_seasonal_diff)

result_season = adfuller(sales_seasonal_diff)
print(f'ADF Statistic:{result_season[0]}')#ADF统计量值
print(f'p-value:{result_season[1]}')
print(f'Critical Values:{result_season[4]}')#临界值 1%、5%、10% 三个显著性水平下的临界值。

if result_season[1] < 0.05:#p值小于0.05就是平稳的
    print(' 一阶季节性差分后，数据是平稳的，无需差分')
else:
    print('一阶季节性差分后，数据是非平稳的，需要差分')#结果数据平稳，所有D选择1

"""
参数选择
一、order (p, d, q)：根据普通ACF和PACF图，同时进行ADF 测试，参数为（1，0，1）
二、seasonal_order (P, D, Q, s)：根据季节性ACF 和 PACF 图，同时进行季节性差分，参数为（1，1，1，7），因为按周有明显规律
"""

#4、SARIMA模型训练
sales.index = pd.to_datetime(sales['Date'])# 将 'Date' 列设置为时间索引,预测结果以时间索引返回
model = SARIMAX(sales['Sales'],order=(1,0,1),seasonal_order=(1,1,1,7))
sarima_result = model.fit()

forecast = sarima_result.get_forecast(steps=6*7,alpha=0.05)#置信区间95%
forecast_ci = forecast.conf_int()# 获取预测值的置信区间，会返回一个包含两个列的数据框，分别是 lower 和 upper，表示置信区间的下限和上限
#获取预测值：上下限均值
forecast_value = forecast.predicted_mean
#返回的是一个 Series 对象，而不是一个方法，因此它不能像函数一样被调用（即不能加括号）
print(f'forecast_ci\n{forecast_ci}')
print(f'forecast_value\n{forecast_value}')
#创建未来6周日期索引
#last_date = sales['Date'].max()#获取数据集最后一天日期
#future_date = pd.date_range(last_date,periods=6*7+1,freq='D')[1:]## 未来42天的日期，排除当前日期

#将日期、预测结果结合
forecast_df = pd.DataFrame({
    'Forecast':forecast_value,
    'Lower Bound':forecast_ci['lower Sales'],
    'Upper Bound':forecast_ci['upper Sales']
})
print(f'forecast_df\n{forecast_df}')

#可视化
plt.figure(figsize=(40,6))
plt.plot(sales['Date'],sales['Sales'],label='历史数据',color='blue')
plt.plot(forecast_df.index,forecast_df['Forecast'],label='262店铺未来6周预测销售额',color='red')
plt.fill_between(forecast_df.index,
                 forecast_df['Lower Bound'],
                 forecast_df['Upper Bound'],
                 color='pink',
                 alpha=0.3,
                 label='预测区间')
plt.xlabel('日期')
plt.ylabel('销售额')
plt.legend(loc='upper left')
plt.title('SARIMA模型销售预测')
plt.show()


#进一步分析趋势、季节性等成分
fig,ax = plt.subplots(2,1,figsize=(12,8))

#趋势
ax[0].plot(sarima_result.fittedvalues,label='趋势',color='green')
ax[0].plot(sales['Sales'],label='实际数据',color='blue',alpha=0.3)
ax[0].set_title('SARIMA模型拟合趋势')
ax[0].legend()

#残差
residuals = sarima_result.resid
ax[1].plot(residuals,label='残差',color='orange')
ax[1].set_title('SARIMA 模型残差')
ax[0].legend()
plt.show()