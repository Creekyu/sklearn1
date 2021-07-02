import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# 解决绘图中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# [-1,1]归一化
def normalization(data):
    _range = np.max(abs(data))
    return data / _range


# 分类排序函数
def sort_area(data, area):
    total = [data, area]
    sort_1 = [[], []]
    sort_2 = [[], []]
    sort_3 = [[], []]
    sort_4 = [[], []]
    sort_5 = [[], []]
    # 按分类分别放进sort_1,sort_2,sort_3,sort_4,sort_5
    # 分几类就用到几个
    for i, j in zip(data, area):
        if (i == 0):
            sort_1[0].append(i)
            sort_1[1].append(j)
        if (i == 1):
            sort_2[0].append(i)
            sort_2[1].append(j)
        if (i == 2):
            sort_3[0].append(i)
            sort_3[1].append(j)
        if (i == 3):
            sort_4[0].append(i)
            sort_4[1].append(j)
        if (i == 4):
            sort_5[0].append(i)
            sort_5[1].append(j)
    return [sort_1, sort_2, sort_3, sort_4, sort_5]


if __name__ == "__main__":
    # 考虑到如果对每一消费分类下的31地区进行聚类，工作量巨大
    # 故我们对每一地区的消费总和进行聚类
    df = pd.read_excel("work.xlsx")
    df.index = df["地区"]  # 改行名为地区列
    del df["地区"]  # 删除地区列
    df['平均消费总额'] = df.apply(lambda x: x.sum(), axis=1)  # 列加总
    # 删除其他列
    d = list(range(0, 8))  # 1-8列
    df.drop(df.columns[d], axis=1, inplace=True)
    print(df)

    # K-means聚类
    x = df["平均消费总额"].tolist()  # 提取数据集
    arr = np.array(x)
    arr_ = normalization(arr)  # 归一化
    arr = [[i] for i in arr_]  # 将列表中每一个元素都转换成单元素列表（Kmeans数据集要求）
    # print(arr)

    # 三类别
    y_pred = KMeans(n_clusters=3, random_state=9).fit_predict(arr)
    area = df.index.tolist()  # 获取行名
    sort = sort_area(y_pred, area)  # 调用sort_area函数
    plt.scatter(sort[0][1], sort[0][0],marker="<")
    plt.scatter(sort[1][1], sort[1][0],marker="^")
    plt.scatter(sort[2][1], sort[2][0],marker="v")
    plt.xticks(rotation=70)  # x轴标签旋转70度
    plt.xlabel("地区")
    plt.ylabel("分类")
    plt.title("K-means聚类平均消费水平")
    plt.show()

    # 四类别
    y_pred_1 = KMeans(n_clusters=4, random_state=9).fit_predict(arr)
    sort_1 = sort_area(y_pred_1, area)
    plt.scatter(sort_1[0][1], sort_1[0][0],marker="<")
    plt.scatter(sort_1[1][1], sort_1[1][0],marker="^")
    plt.scatter(sort_1[2][1], sort_1[2][0],marker="v")
    plt.scatter(sort_1[3][1], sort_1[3][0],marker=">")
    plt.xticks(rotation=70)  # x轴标签旋转70度
    plt.xlabel("地区")
    plt.ylabel("分类")
    plt.title("K-means聚类平均消费水平")
    plt.show()

    # 系统聚类
    # 3类
    y_pred_2 = AgglomerativeClustering(n_clusters=3).fit_predict(arr)
    sort_2 = sort_area(y_pred_2,area)
    plt.scatter(sort_2[0][1], sort_2[0][0],marker="<")
    plt.scatter(sort_2[1][1], sort_2[1][0],marker="^")
    plt.scatter(sort_2[2][1], sort_2[2][0],marker="v")
    plt.xticks(rotation=70)  # x轴标签旋转70度
    plt.xlabel("地区")
    plt.ylabel("分类")
    plt.title("系统聚类平均消费水平")
    plt.show()

    # 4类
    y_pred_3 = AgglomerativeClustering(n_clusters=4).fit_predict(arr)
    sort_3 = sort_area(y_pred_3, area)
    plt.scatter(sort_3[0][1], sort_3[0][0],marker="<")
    plt.scatter(sort_3[1][1], sort_3[1][0],marker="^")
    plt.scatter(sort_3[2][1], sort_3[2][0],marker="v")
    plt.scatter(sort_3[3][1], sort_3[3][0],marker=">")
    plt.xticks(rotation=70)  # x轴标签旋转70度
    plt.xlabel("地区")
    plt.ylabel("分类")
    plt.title("系统聚类平均消费水平")
    plt.show()

