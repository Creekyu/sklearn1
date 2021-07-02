import pandas as pd
import matplotlib.pyplot as plt

# 解决绘图中文乱码
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

# 直方图函数
def plot_bar(df,start,end):
    df[start:end].plot(kind="bar")
    plt.xlabel("地区")
    plt.ylabel("平均消费支出(单位：元)")
    plt.title("各地区各类平均消费水平直方图")
    plt.xticks(rotation=60)  # x轴标签旋转60度
    plt.figure(dpi=400)  # 调整画布大小
    plt.show()

if __name__ == "__main__":
    df = pd.read_excel("work.xlsx")
    df.index = df["地区"] # 改行名为地区列
    del df["地区"] # 删除地区列
    # 直方图
    plot_bar(df,0,10)
    plot_bar(df,10,20)
    plot_bar(df,20,31)

    # 线图
    area = df.index.tolist()
    df.plot()
    plt.xlabel("地区")
    plt.ylabel("平均消费水平（单位：元）")
    x = range(0, 31)
    plt.xticks(x, area, rotation=70)  # 替换x轴坐标,并旋转70度
    plt.show()


