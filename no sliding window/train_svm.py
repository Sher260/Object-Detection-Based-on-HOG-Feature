from sklearn.svm import LinearSVC, SVC
from skimage.feature import hog
# import SVC
from skimage.io import imread
from skimage import transform, data
import joblib
import matplotlib.pyplot as plt
import glob
import os
import random
import time as tm
from random import shuffle
from sklearn.preprocessing import StandardScaler

from config import *
import numpy as np
model_path = '../data/models/svm_model_fenlei'
# 测试代码
def test():
    # ===========1 读取数据===============
    # 获取标注数据地址
    clf = joblib.load(model_path)
    path = "C:/Users/hitzo/Desktop/datami/test"
    file_name_list = os.listdir(path)
    file_name_list_new = []  # 图片所在文件夹
    for file_name in file_name_list:
        tmp_file = path + '/' + file_name
        a = os.path.isdir(tmp_file)
        # b = os.path.isfile(tmp_file)
        if os.path.isdir(tmp_file):
            file_name_list_new.append(tmp_file)
    N_file = len(file_name_list_new)  # 标注图片数量
    # shuffle(file_name_list_new)

    N_win = 80  # 块大小
    N_x = 480 // N_win
    N_y = 800 // N_win
    N_block = N_x * N_y  # 快的数量
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ion()
    plt.show()

    for i_file, file_name in enumerate(file_name_list_new):
        x_list = []  # 非常多的小块
        y_list = []  # 对应的标签
        t1 = tm.time()

        x_path = file_name + '/img.jpg'
        y_path = file_name + '/label.jpg'

        x_im = imread(x_path, as_gray=True)
        y_im = imread(y_path, as_gray=True)

        x_dst = transform.resize(x_im, (480, 800))  # 小图
        y_dst = transform.resize(y_im, (480, 800))  # 小图
        y_dst = y_dst / np.max(y_dst)  # 归一化

        target_area = np.sum(y_dst)  # 目标面积大小

        # 划分小块
        position_list = np.zeros([N_block, 2])  # 每一个小块的坐标
        tmp = 0
        for i in range(N_x):
            for j in range(N_y):
                position_list[tmp, :] = [i * N_win, j * N_win]
                x_tmp = x_dst[i * N_win:(i + 1) * N_win, j * N_win:(j + 1) * N_win]
                fd = hog(x_tmp, orientations=9, pixels_per_cell=[20, 20], cells_per_block=[4, 4],
                         visualize=False, transform_sqrt=True)

                y_tmp = y_dst[i * N_win:(i + 1) * N_win, j * N_win:(j + 1) * N_win]

                y_flag = np.sum(y_tmp) > target_area * 0.3  # 包含目标的小块是正样本
                x_list.append(fd)
                y_list.append(y_flag)
                tmp += 1

        pred = clf.predict(x_list)
        pred1 = clf.decision_function(x_list) > 0.4
        N_target = np.sum(np.array(y_list) == 1)  # 总目标数量
        N_detet = np.sum(pred[np.array(y_list) == 1])  # 检测到的目标
        pd = N_detet / N_target

        detect_position = position_list[pred1 == 1, :]
        ax.cla()
        ax.imshow(x_dst)
        for i in range(detect_position.shape[0]):
            x0 = detect_position[i, 0]
            y0 = detect_position[i, 1]

            rect = plt.Rectangle([y0, x0], N_win, N_win, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        a = 1
        filename = 'C:/Users/hitzo/Desktop/pichog/' + str(i_file) +'.jpg'
        plt.savefig(filename)


# 训练代码
def train_svm():
    model_path = '../data/models/svm_model_sky'

    N_train_img = 0  # 训练图片
    N_train_block = N_train_img * 60 # 训练的块数量
    fds = np.load('data_y_skysift (2).npy')
    labels = np.load('data_y_skysift (2).npy')

    # 将数据随机打乱
    N_block_all = len(labels) # 总的块的数量
    ID = np.random.choice(list(range(N_block_all)), N_block_all)
    fds = fds[ID,:]
    labels = labels[ID,:]


    # 训练数据
    x_train =fds[:N_train_block, :]
    y_train = labels[:N_train_block, 0]

    # 测试数据
    x_test = fds[N_train_block:, :]
    y_test = labels[N_train_block:, 0]

    R = np.zeros(20)
    Pre = np.zeros(20)
    Acc = np.zeros(20)
    FA = np.zeros(20)
    MA = np.zeros(20)
    FA_block = np.zeros(20)
    Rall = 0
    Pall = 0
    Aall = 0
    FAall = 0
    MAall = 0

    t1 = tm.time()
    # clf = LinearSVC(class_weight={1: 110}, max_iter=5000)
    clf = SVC(class_weight={1: 70}, max_iter=50000)
    # ==========样本重采样==========
    for i in range(20):
        [posive_ID] = np.where(y_train == 1)
        [negative_ID] = np.where(y_train == 0)
        negative_ID = np.random.choice(negative_ID,70*posive_ID.size)#正负样本数量比： 1:60
        all_ID = np.append(posive_ID, negative_ID)
        scaler = StandardScaler()
        x_train1 = x_train[all_ID,:]
        # x_train1 = scaler.fit_transform(x_train1)
        y_train1 = y_train[all_ID]
        # clf.fit(x_train1, y_train1) # 网络训练

        x_input = x_test
        y_input = y_test

        # joblib.dump(clf, model_path)  # 保存模型
        model_path = '../data/models/svm_model_fenleisift'
        clf = joblib.load(model_path)
        pred = clf.predict(x_input)
        P = np.sum(y_input == 1)  # 总目标数量 P
        N = np.sum(y_input == 0)  # 总杂波数量 N
        TP = np.sum(pred[y_input == 1])  # 检测到的目标 TP
        FP = np.sum(pred[y_input == 0])  # 杂波虚警个数
        TN = N - FP
        R[i] = TP / P  # 召回率 R
        Rall = Rall + R[i]
        Pre[i] = TP / (TP + FP)  # 精确度
        Pall = Pall + Pre[i]
        Acc[i] = (TP + TN) / (P + N)  # 准确率
        Aall = Aall + Acc[i]
        MA[i] = 1 - R[i]  # 漏警率
        MAall = MAall + MA[i]
        FA[i] = FP / (TP + FP)  # 虚警率
        FAall = FAall + FA[i]
        FA_block[i] = FP / (x_input.shape[0] / 144)
    a = 1
    t2 = tm.time()
    # print('R=', Rall/20, '\n', 'Pre=', Pall/20, '\n', 'Acc=', Aall/20, '\n', 'MA=', MAall/20, '\n', 'FA=', FAall/20)
    x_axis_data = range(5)  # x
    print('R=', Rall*5, '\n', 'Pre=', Pall*5, '\n', 'Acc=', Aall*5, '\n', 'MA=', MAall*5, '\n', 'FA=', FAall*5)
    print(f'time:{(t2-t1)/60}')

    y_axis_data1 = R # y
    y_axis_data2 = Pre
    y_axis_data3 = Acc
    y_axis_data4 = MA
    y_axis_data5 = FA

    plt.plot(x_axis_data, y_axis_data1, 'b*--', alpha=0.5, linewidth=2, label='R',marker = 's',markersize = 4)
    plt.plot(x_axis_data, y_axis_data2, 'rs--', alpha=0.5, linewidth=2, label='Pre',marker = 'o',markersize = 4)
    plt.plot(x_axis_data, y_axis_data3, 'go--', alpha=0.5, linewidth=2, label='Acc',marker = '^',markersize = 4)
    plt.plot(x_axis_data, y_axis_data4, 'k--', alpha=0.5, linewidth=2, label='MA',marker = '+',markersize = 4)
    plt.plot(x_axis_data, y_axis_data5, 'tab:purple', alpha=0.5, linewidth=2, label='FA',marker = '*',markersize = 4)

    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)  # 显示上面的label
    plt.xlabel('time',fontsize=20)
    plt.ylabel('result',fontsize=20)  # accuracy

    plt.xticks(range(0, 20),fontsize=20)
    plt.show()


if __name__ == '__main__':
    # train_svm()
    test()
