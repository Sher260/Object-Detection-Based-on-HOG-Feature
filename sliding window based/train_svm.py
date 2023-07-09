from sklearn.svm import LinearSVC,SVC
from skimage.feature import hog
from skimage.io import imread
from skimage import transform, data
from sklearn.externals import joblib
# import joblib
import matplotlib.pyplot as plt
import glob
import os
import random
import time as tm
from random import shuffle

from config import *
import numpy as np
model_path = '../data/models/svm_model'

# 测试代码
def test():
    # ===========1 读取数据===============
    # 获取标注数据地址
    clf = joblib.load(model_path)
    path = "C:/Users/hitzo/Desktop/datami/test"
    file_name_list = os.listdir(path)
    file_name_list_new = []  # 图片所在文件夹
    for file_name in file_name_list:
        tmp_file = path + '\\' + file_name
        a = os.path.isdir(tmp_file)
        # b = os.path.isfile(tmp_file)
        if os.path.isdir(tmp_file):
            file_name_list_new.append(tmp_file)
    N_file = len(file_name_list_new)  # 标注图片数量
    shuffle(file_name_list_new)

    N_win = 60  # 块大小
    N_x = 480 // N_win *2
    N_y = 780 // N_win *2
    N_block = (N_x-1) * (N_y-1)  # 快的数量
    sum_target = 0
    sum_detect = 0
    nms_target = 0
    x_list = []  # 非常多的小块
    y_list = []  # 对应的标签

    fig = plt.figure()

    ax1 = fig.add_subplot(121)
    plt.ion()
    plt.show()
    ax2 = fig.add_subplot(122)
    plt.ion()
    plt.show()

    for i_file, file_name in enumerate(file_name_list_new):

        t1 = tm.time()

        x_path = file_name + '/img.jpg'
        y_path = file_name + '/label.jpg'

        x_im = imread(x_path, as_gray=True)
        y_im = imread(y_path, as_gray=True)

        x_dst = transform.resize(x_im, (480, 780))  # 小图
        y_dst = transform.resize(y_im, (480, 780))  # 小图
        y_dst = y_dst / np.max(y_dst)  # 归一化

        target_area = np.sum(y_dst)  # 目标面积大小

        # 划分小块
        position_list = np.zeros([N_block, 2])  # 每一个小块的坐标
        tmp = 0
        for i in range(N_x-1):
            for j in range(N_y-1):
                position_list[tmp, :] = [N_win//2 * i, N_win//2 * j]
                x_tmp = x_dst[N_win//2 * i : (N_win//2 * i)+ N_win ,  N_win//2 * j : (N_win//2 * j) + N_win]
                fd = hog(x_tmp, orientations=9, pixels_per_cell=[20, 20], cells_per_block=[2, 2],
                         visualize=False, transform_sqrt=True)

                y_tmp = y_dst[N_win//2 * i : (N_win//2 * i)+ N_win ,  N_win//2 * j : (N_win//2 * j) + N_win]

                y_flag = np.sum(y_tmp) > target_area * 0.4  # 包含目标的小块是正样本
                x_list.append(fd)
                y_list.append(y_flag)
                tmp += 1

        pred = clf.predict(x_list)
        prob = np.array(clf.predict_proba(x_list))
        prob_now = prob[i_file*375 : (i_file+1)*375 ,:]
        NMS_locat = np.argmax(prob_now[:, 1]) + i_file*375
        if max(prob_now[:,1])>0.5:
            if y_list[NMS_locat] == 0:
                nms_target +=1 # NMS算法虚警数

        ax1.cla()
        ax2.cla()
        detect_position = position_list[pred[i_file*375:(i_file+1)*375] == 1, :]
        ax2.imshow(x_dst)
        if max(prob_now[:,1])>0.5:
            max_prob_pos = np.argmax(prob_now[:,1])
            a = 1
            x0 = position_list[max_prob_pos, 0]
            y0 = position_list[max_prob_pos, 1]
            rect = plt.Rectangle([y0, x0], N_win, N_win, linewidth=1, edgecolor='r', facecolor='none' )
            ax2.add_patch(rect)

        ax1.imshow(x_dst)
        for i in range(detect_position.shape[0]):
            x0 = detect_position[i, 0]
            y0 = detect_position[i, 1]
            rect = plt.Rectangle([y0, x0], N_win, N_win, linewidth=1, edgecolor='r', facecolor='none' )
            ax1.add_patch(rect)

        N_target = np.sum(np.array(y_list) == 1)  # 总目标数量
        # sum_target = sum_target + N_target
        # N_detet = np.sum(pred[np.array(y_list) == 1])
        TP = np.sum(pred[np.array(y_list) == 1])  # 检测到的目标
        FP = np.sum(pred[np.array(y_list) == 0])  # 杂波虚警个数
        FA = FP / (TP + FP)
        # sum_detect = sum_detect + N_detet
        # pd = N_detet / N_target  # 召回
        #
        # NMS_FA = nms_target / N_detet

        print(i_file , nms_target, FA)

# 训练代码
def train_svm():

    model_path = '../data/models/svm_model'

    N_train_img = 300 # 训练图片
    N_train_block = N_train_img * 375 # 训练的块数量
    fds = np.load('data_x.npy')
    labels = np.load('data_y.npy')

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


    # clf = SVC(class_weight={1: 70},probability=True)
    # # ==========样本重采样==========
    # [posive_ID] = np.where(y_train==1)
    # [negative_ID] = np.where(y_train==0)
    # negative_ID = np.random.choice(negative_ID,70*posive_ID.size)#正负样本数量比： 70：1
    # all_ID = np.append(posive_ID ,negative_ID)
    # x_train1 = x_train[all_ID,:]
    # y_train1 = y_train[all_ID]
    # clf.fit(x_train1, y_train1) # 网络训练

    # if not os.path.isdir(os.path.split(model_path)[0]):#用于判断model_path上一级是不是个路径
    #     os.makedirs(os.path.split(model_path)[0])#创建目录
    # # print('Classifier save to {}'.format(model_path))
    # joblib.dump(clf, model_path)#保存模型
    # print('Classifier save to {}'.format(model_path))
    clf = joblib.load(model_path)
    x_input = x_test
    y_input = y_test

    # x_input = x_train
    # y_input = y_train
    pred = clf.predict(x_input)
    P = np.sum(y_input == 1) # 总目标数量 P
    N = np.sum(y_input == 0) # 总杂波数量 N
    TP = np.sum(pred[y_input == 1])  # 检测到的目标 TP
    FP = np.sum(pred[y_input == 0])  # 杂波虚警个数
    TN = N - FP
    R = TP/P # 召回率 R
    Pre = TP/(TP+FP) #精确度
    Acc = (TP + TN)/(P + N) #准确率
    MA = 1-R #漏警率
    FA = FP /(TP + FP) #虚警率

    # pfa = N_false / N_clutter
    pfa_block = FP/ (x_input.shape[0]/375)
    joblib.dump(clf, model_path)  # 保存模型
    test()




if __name__ == '__main__':
    #train_svm()
    test()