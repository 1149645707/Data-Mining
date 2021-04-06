import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("../data/archive/cbg_patterns.csv")
data.head()

# 统计distance_from_home一列每个可能聚会的频数
distance = data.distance_from_home.value_counts()
plt.figure(figsize=(20, 8))
plt.hist(distance)
# 给出raw_visit_count一列的五数概括
data.raw_visit_count.describe()
# 统计每一列缺失值的个数
print(data.isnull().sum())
# 对于census_block_group缺失的数据，由于census_block_group是人口普查区组唯一的12位FIPS码，无法进行填补，故选择直接将该行数据剔除
data[data['census_block_group'].isnull()]
data.drop(220734,inplace=True)  # 将索引为220734的数据剔除
# 对于raw_visit_count和raw_visitor_count两列的缺失值，分别使用他们的均值进行填充
raw_visit_count_mean = np.ceil( np.mean(data.raw_visit_count))
data.raw_visit_count = data.raw_visit_count.fillna(raw_visit_count_mean)
raw_visitor_count_mean = np.ceil( np.mean(data.raw_visitor_count))
data.raw_visitor_count = data.raw_visitor_count.fillna(raw_visitor_count_mean)
# 对于distance_from_home一列使用最高频率值来填补缺失值
data.distance_from_home.value_counts()
data.distance_from_home = data.distance_from_home.fillna(8345.0)  # 使用最高频率值进行填充
# 至此所有缺失值填补完毕
print(data.isnull().sum())
# 绘制盒图
plt.boxplot(data.raw_visit_count)
# 绘制直方图
plt.hist(data.census_block_group, bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
