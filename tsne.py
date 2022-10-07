import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets

# digits = datasets.load_digits(n_class=6)
# X, y = digits.data, digits.target
# n_samples, n_features = X.shape
#
# '''显示原始数据'''
# n = 20  # 每行20个数字，每列20个数字
# img = np.zeros((10 * n, 10 * n))
# for i in range(n):
#     ix = 10 * i + 1
#     for j in range(n):
#         iy = 10 * j + 1
#         img[ix:ix + 8, iy:iy + 8] = X[i * n + j].reshape((8, 8))
# plt.figure(figsize=(8, 8))
# plt.imshow(img, cmap=plt.cm.binary)
# plt.xticks([])
# plt.yticks([])
# plt.show()



def visualization(X, y, figname, len_=-1, show=False):
    # np.random(100)

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501, learning_rate='auto', method='barnes_hut',
                         early_exaggeration=20)
    X_tsne = tsne.fit_transform(X)
    # print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    dict_ = {}
    plt.figure(figsize=(8, 8))
    poi_pots = []
    user_pots = []
    for i in range(X_norm.shape[0]):
        # plt.text(X_norm[i, 0], X_norm[i, 1], '·', color=plt.cm.Set1(y[i]),
        #          fontdict={'weight': 'bold', 'size': 9})  # str(y[i])
        # print(y[i])
        if y[i] < 9:
            co = plt.cm.Set1(y[i])
        elif y[i] < 9+8:
            co = plt.cm.Set2(y[i]-9)
        else:
            co = plt.cm.Set3(y[i] - 9 - 8)
        if len_ != -1 and i >= len_:
            if show is True:
                for pair in dict_[y[i]]:
                    plt.plot([pair[0], X_norm[i, 0]], [pair[1], X_norm[i, 1]], alpha=0.09, color=co, linewidth=0.5) #, linestyle='--'
            user_pots.append([X_norm[i, 0], X_norm[i, 1], co])
        else:
            if y[i] not in dict_:
                dict_[y[i]] = [[X_norm[i, 0], X_norm[i, 1]]]
            else:
                dict_[y[i]].append([X_norm[i, 0], X_norm[i, 1]])
            poi_pots.append([X_norm[i, 0], X_norm[i, 1], co])
    for poi_pot in poi_pots:
        plt.scatter(poi_pot[0], poi_pot[1], color=poi_pot[2], marker='x')
    if show is True:
        for user_pot in user_pots:
            plt.scatter(user_pot[0], user_pot[1], color=user_pot[2], marker='o', linewidths=6)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(fname='visualization/{}.png'.format(figname), format='png')
    print("Saved visualization/{}.eps successfully. ".format(figname))
    plt.show()

    # ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    # #  将数据点分成三部分画，在颜色上有区分度
    # for i in range(X_norm.shape[0]):
    #     if y[i] < 9:
    #         co = plt.cm.Set1(y[i])
    #     elif y[i] < 9+8:
    #         co = plt.cm.Set2(y[i]-9)
    #     else:
    #         co = plt.cm.Set3(y[i] - 9 - 8)
    #     if len_ != -1 and i > len_:
    #         ax.scatter(X_norm[i, 0], X_norm[i, 1], X_norm[i, 2], color=co, marker='o', linewidths=12)  # 绘制数据点
    #     else:
    #         ax.scatter(X_norm[i, 0], X_norm[i, 1], X_norm[i, 2], color=co, marker='x')  # 绘制数据点
    #
    # ax.set_zlabel('Z')  # 坐标轴
    # ax.set_ylabel('Y')
    # ax.set_xlabel('X')
    # plt.savefig(fname='visualization/{}.eps'.format(figname), format='eps')
    # plt.show()


# X = np.random.rand(500,100)
# y = np.random.randint(0, 10, (500))
# visualization(X, y, 'test')
#
# X = X[:,:10]
# # y = np.random.randint(0, 10, (1500))
# visualization(X, y, 'test2')