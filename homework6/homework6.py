import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import datetime

data_origin = pd.read_csv("../data/archive/hotel_bookings.csv")
pd.set_option('display.max_columns', None)  # 显示所有列

# 选择有用的列进行后续的数据分析
data = pd.DataFrame(data_origin[
                        ['hotel', 'is_canceled', 'arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month',
                         'meal', 'previous_bookings_not_canceled', 'reserved_room_type', 'customer_type', 'adr',
                         'reservation_status', 'reservation_status_date']])

data.replace('Undefined', 'SC', inplace=True)  # 将meal一列值为Undefined全部替换为SC
data_rh = data[(data.hotel == 'Resort Hotel')]
data_ch = data[(data.hotel == 'City Hotel')]
# 预订房间类型比较
plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.bar(data_rh.reserved_room_type.value_counts().index, data_rh.reserved_room_type.value_counts(), width=0.6,
        color='b', label='Resort Hotel')
plt.xlabel('reserved_room_type')
plt.ylabel('numbers')
plt.title('Resort Hotel')
plt.subplot(1, 2, 2)
plt.bar(data_ch.reserved_room_type.value_counts().index, data_ch.reserved_room_type.value_counts(), width=0.6,
        color='g', label='City Hotel')
plt.xlabel('reserved_room_type')
plt.ylabel('numbers')
plt.title('City Hotel')

# 入住率比较   入住率=预定未取消记录数/总记录数
occupancy_rh = data_rh[data_rh.is_canceled == 0].is_canceled.count() / data_rh.is_canceled.count()
occupancy_ch = data_ch[data_ch.is_canceled == 0].is_canceled.count() / data_ch.is_canceled.count()
print('假日酒店的入住率={:.4}'.format(occupancy_rh))
print('城市酒店的入住率={:.4}'.format(occupancy_ch))

data_lead = data_origin[['hotel', 'is_canceled', 'lead_time']]
data_lead_rh = data_lead[(data_lead.hotel == 'Resort Hotel') & data_lead.is_canceled == 0]
data_lead_ch = data_lead[(data_lead.hotel == 'City Hotel') & data_lead.is_canceled == 0]
# 绘制假日酒店和城市酒店提前预订时间与对应记录条数的散点图
plt.figure(figsize=(15, 8))
plt.scatter(data_lead_rh.lead_time.value_counts()[0:20].index, data_lead_rh.lead_time.value_counts()[0:20], marker='*',
            label='Resort Hotel')
plt.scatter(data_lead_ch.lead_time.value_counts()[0:20].index, data_lead_ch.lead_time.value_counts()[0:20], marker='o',
            label='City Hotel')
plt.legend()
plt.xlabel('lead_time')
plt.ylabel('counts')

data_bookdate = pd.DataFrame(
    data[['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month', 'reservation_status_date']])

# 将到达月份做字符串到数字的映射
class_mapping = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8,
                 'September': 9, 'October': 10, 'November': 11, 'December': 12}
data_bookdate.arrival_date_month = data_bookdate.arrival_date_month.map(class_mapping)
data_bookdate['reservation_status_date'] = pd.to_datetime(data_bookdate['reservation_status_date'])  # 转换为日期格式
data_bookdate['reservation_status_date_year'] = data_bookdate['reservation_status_date'].dt.year
data_bookdate['reservation_status_date_month'] = data_bookdate['reservation_status_date'].dt.month
data_bookdate['reservation_status_date_day'] = data_bookdate['reservation_status_date'].dt.day
data_bookdate.drop(columns=['reservation_status_date'], inplace=True)

book_days_list = []
for i in range(0, len(data_bookdate)):
    a_date = datetime.date(data_bookdate.iloc[i]['arrival_date_year'], data_bookdate.iloc[i]['arrival_date_month'],
                           data_bookdate.iloc[i]['arrival_date_day_of_month'])
    r_date = datetime.date(data_bookdate.iloc[i]['reservation_status_date_year'],
                           data_bookdate.iloc[i]['reservation_status_date_month'],
                           data_bookdate.iloc[i]['reservation_status_date_day'])
    book_days_list.append((r_date - a_date).days)  # 计算每条记录的入住时长
book_days = pd.Series(book_days_list)
book_days.value_counts()[0:20]

# 绘制入住时长与对应记录数量的条形统计图
plt.figure(figsize=(15, 8))
plt.bar(book_days.value_counts()[0:20].index, book_days.value_counts()[0:20], width=0.5, color='orange')
plt.xlabel('reservation_date - arrive_date')
plt.ylabel('counts')

data_repeat = pd.DataFrame(data_origin[['is_repeated_guest', 'agent', 'reservation_status_date']])
# 以agent ID=240的客户为例：
data_repeat = pd.DataFrame(data_repeat[(data_repeat.is_repeated_guest == 1) & (data_repeat.agent == 240)])
data_repeat.drop_duplicates('reservation_status_date', inplace=True)  # 删除所有列值相同的记录
data_repeat['reservation_status_date'] = pd.to_datetime(data_repeat['reservation_status_date'])  # 转换为日期格式
data_repeat['year'] = data_repeat['reservation_status_date'].dt.year
data_repeat['month'] = data_repeat['reservation_status_date'].dt.month
data_repeat['day'] = data_repeat['reservation_status_date'].dt.day
data_repeat.drop(columns=['reservation_status_date'], inplace=True)
book_repeat_list = []
for i in range(0, len(data_repeat)):
    repeat = datetime.date(int(data_repeat.iloc[i]['year']), int(data_repeat.iloc[i]['month']),
                           int(data_repeat.iloc[i]['day']))
    book_repeat_list.append(repeat)
book_repeat_list.sort()
gap = {}
for index, i in enumerate(book_repeat_list):
    if index == len(book_repeat_list) - 1:
        break
    gap[index] = (book_repeat_list[index + 1] - book_repeat_list[index]).days
# 绘制每次预订相隔天数的折线图
plt.figure(figsize=(15, 8))
keys = list(gap.keys())
values = list(gap.values())
plt.plot(keys, values)
plt.xlabel('number of times')
plt.ylabel('gap numbers')

# 订餐类型比较
bar_width = 0.3  # 条形宽度
index_rh = np.arange(len(data_rh.meal.value_counts().index))
index_ch = index_rh + bar_width
plt.figure(figsize=(15, 8))
plt.bar(index_rh, data_rh.meal.value_counts(), width=bar_width, color='orange', label='Resort Hotel')
plt.bar(index_ch, data_ch.meal.value_counts(), width=bar_width, color='c', label='City Hotel')
plt.legend()
plt.xticks(index_rh + bar_width / 2,
           data_rh.meal.value_counts().index)  # 让横坐标轴刻度显示data_rh.meal.value_counts().index， index_rh + bar_width/2 为横坐标轴刻度的位置
plt.xlabel('meal')
plt.ylabel('numbers')
plt.title('comparison of meal')

data_bestBooking = pd.DataFrame(
    data_origin[['hotel', 'is_canceled', 'arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']])
# 首先看假日酒店和城市酒店的最佳预订月份
data_bestBooking_rh = data_bestBooking[(data_bestBooking.hotel == 'Resort Hotel') & (data_bestBooking.is_canceled == 0)]
data_bestBooking_ch = data_bestBooking[(data_bestBooking.hotel == 'City Hotel') & (data_bestBooking.is_canceled == 0)]
plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.bar(data_bestBooking_rh.arrival_date_month.value_counts().index,
        data_bestBooking_rh.arrival_date_month.value_counts(), width=0.6, color='b', label='Resort Hotel')
plt.xlabel('booking month')
plt.xticks(rotation=90)
plt.ylabel('numbers')
plt.title('Resort Hotel')
plt.subplot(1, 2, 2)
plt.bar(data_bestBooking_ch.arrival_date_month.value_counts().index,
        data_bestBooking_ch.arrival_date_month.value_counts(), width=0.6, color='g', label='City Hotel')
plt.xticks(rotation=90)
plt.xlabel('booking month')
plt.ylabel('numbers')
plt.title('City Hotel')

data_bestBooking_rh_day = data_bestBooking[
    (data_bestBooking.hotel == 'Resort Hotel') & (data_bestBooking.is_canceled == 0) & (
                data_bestBooking.arrival_date_month == 'August')]
data_bestBooking_ch_day = data_bestBooking[
    (data_bestBooking.hotel == 'City Hotel') & (data_bestBooking.is_canceled == 0) & (
                data_bestBooking.arrival_date_month == 'August')]
plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.bar(data_bestBooking_rh_day.arrival_date_day_of_month.value_counts()[0:10].index,
        data_bestBooking_rh.arrival_date_day_of_month.value_counts()[0:10], width=0.6, color='b', label='Resort Hotel')
plt.xlabel('booking day')
plt.ylabel('numbers')
plt.title('Resort Hotel')
plt.subplot(1, 2, 2)
plt.bar(data_bestBooking_ch_day.arrival_date_day_of_month.value_counts()[0:10].index,
        data_bestBooking_ch.arrival_date_day_of_month.value_counts()[0:10], width=0.6, color='g', label='City Hotel')
plt.xlabel('booking day')
plt.ylabel('numbers')
plt.title('City Hotel')

data_LR = pd.DataFrame(data_origin[['hotel', 'is_canceled', 'lead_time', 'adults', 'children', 'babies', 'meal',
                                    'previous_cancellations', 'previous_bookings_not_canceled', 'booking_changes',
                                    'adr', 'reservation_status']])
data_LR.replace('Undefined', 'SC', inplace=True)  # 将meal一列值为Undefined全部替换为SC
class_mapping1 = {'Resort Hotel': 0, 'City Hotel': 1}
class_mapping2 = {'Check-Out': 0, 'Canceled': 1, 'No-Show': 2}
data_LR.hotel = data_LR.hotel.map(class_mapping1)
data_LR.reservation_status = data_LR.reservation_status.map(class_mapping2)
meal_onehot = pd.get_dummies(data_LR.meal)

data_LR = pd.concat([data_LR, meal_onehot, data_bookdate], axis=1)
data_LR.drop(['meal'], axis=1, inplace=True)
data_LR.adr = data_LR.adr.astype('int')
data_LR.dropna(inplace=True)

y = data_LR['reservation_status']  # 用reservation_status一列的值做标签
x = data_LR.drop('reservation_status', axis=1)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
logreg = LogisticRegression().fit(X_train, y_train)
print("Training set score:{:.4f}".format(logreg.score(X_train, y_train)))
print("Test set score:{:.4f}".format(logreg.score(X_test, y_test)))

plt.figure(figsize=(12, 8))
plt.plot(logreg.coef_.T, 'o')
plt.xticks(range(x.shape[1]), x.columns, rotation=90)
plt.ylim(-5, 5)
plt.xlabel('columns names')
plt.ylabel('para values')
