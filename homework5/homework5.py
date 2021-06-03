import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori  # apriori算法
from mlxtend.frequent_patterns import association_rules  # 计算关联规则的函数

data1 = pd.read_csv("../data/archive/winemag-data_first150k.csv")
data2 = pd.read_csv("../data/archive/winemag-data-130k-v2.csv")
data_wine1 = data1[['country', 'designation', 'points', 'price', 'variety', 'winery']]
data_wine2 = data2[['country', 'designation', 'points', 'price', 'variety', 'winery']]
# 对两个数据集进行拼接，我们发现只有points和price两列是数值型数据
data_wine = pd.concat([data_wine1, data_wine2], axis=0, ignore_index=True)
data_wine.head()

df = data_wine[(data_wine.designation == 'Reserve') & (data_wine.country == 'Australia')][
    ['price', 'variety', 'winery']]
# 利用get_dummies()方法转换成适合进行关联规则挖掘的形式
df = pd.get_dummies(df)
df = df.drop('price', 1)
df.head()

# 导出频繁项集，由于数据集较大，设置最小支持度为0.025
frequent_itemsets = apriori(df, use_colnames=True, min_support=0.025)

# 导出关联规则，计算支持度和置信度，并使用lift、leverage、conviction指标对规则进行评价，设置lift 最小值为1.25
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.25)
# 对挖掘结果进行分析：将lift值大于6的降序输出
rules[(rules.lift > 6)].sort_values(by=['lift'], ascending=False)
# 可视化展示：绘制关联规则支持度和置信度的散点图
plt.xlabel('support')
plt.ylabel('confidence')
plt.scatter(rules.support, rules.confidence)

df1 = data_wine[(data_wine.variety == 'Shiraz-Viognier')][['price', 'country', 'winery']]
df1.head(5)

for index, i in enumerate(df1.price.values):
    if i >= 10.0 and i < 30.0:
        df1.iat[index, 0] = 5
    elif i >= 30.0 and i < 50.0:
        df1.iat[index, 0] = 4
    elif i >= 50.0 and i < 100.0:
        df1.iat[index, 0] = 3
    elif i >= 100.0 and i < 200.0:
        df1.iat[index, 0] = 2
    else:
        df1.iat[index, 0] = 1
class_mapping = {5: 'D', 4: 'C', 3: 'B', 2: 'A', 1: 'S'}
price_level = df1.price.map(class_mapping)
df1 = pd.concat([df1, price_level], axis=1)
df1.columns = ['price_float', 'country', 'winery', 'price_level']
df1.head(10)

df1 = df1.drop('price_float', 1)
# 将price一列离散化后的数据如下
df1.head(10)

# 利用get_dummies()方法转换成适合进行关联规则挖掘的形式
df1 = pd.get_dummies(df1)
df1.head(10)

pd.options.display.max_colwidth = 100
# 导出频繁项集，设置最小支持度为0.05
frequent_itemsets1 = apriori(df1, use_colnames=True, min_support=0.05)

# 导出关联规则，计算支持度和置信度，并使用lift、leverage、conviction指标对规则进行评价，设置lift 最小值为1.5
rules1 = association_rules(frequent_itemsets1, metric='lift', min_threshold=1.5)

# 对挖掘结果进行分析：将lift值大于9的降序输出
rules1[(rules1.lift > 9)].sort_values(by=['lift'], ascending=False)

# 可视化展示：绘制关联规则支持度和置信度的散点图
plt.xlabel('support')
plt.ylabel('confidence')
plt.scatter(rules1.support, rules1.confidence)
