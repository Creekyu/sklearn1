import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# 解决绘图中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def pca(data, dim=0):
# sklearn中PCA模型没有使用协方差矩阵分析，而是采用SVD(奇异值分解)
# 所以要使用协方差矩阵分析，需要自己定义
# 引用来自：https://blog.csdn.net/qq_43409560/article/details/117535668
# 数据 降维大小（mle为auto）
    if dim == 0:
        dim = 'mle'
    pca_ = PCA(n_components=dim)
    # 降维数据
    after_data = pca_.fit_transform(data)
	# 协方差矩阵
    cov_matritx = pca_.get_covariance()
    # 特征值计算
    lambda_EA = np.linalg.eigvals(cov_matritx)
    return after_data, cov_matritx, lambda_EA


if __name__ == "__main__":
    df = pd.read_excel("work.xlsx")
    df.index = df["地区"]  # 改行名为地区列
    del df["地区"]  # 删除地区列

    # 提取每列数据
    data = []
    data.append(df["食品"].tolist())
    data.append(df["衣着"].tolist())
    data.append(df["居住"].tolist())
    data.append(df["家庭设备及服务"].tolist())
    data.append(df["交通和通讯"].tolist())
    data.append(df["文教娱乐用品及服务"].tolist())
    data.append(df["医疗保健"].tolist())
    data.append(df["其他商品及服务"].tolist())
    data1 = np.array(data)
    print(data1)

    # 主成分分析
    result = pca(data1,2)
    print("降维数据：\n",result[0],end="\n")
    print("协方差矩阵：\n",result[1],end="\n")
    print("特征值：",result[2])






