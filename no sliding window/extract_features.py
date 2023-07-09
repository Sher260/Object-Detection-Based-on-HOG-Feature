from skimage.feature import hog
from skimage.io import imread
from skimage import transform, data
import joblib
import numpy as np
import glob
import os
from config import *
import matplotlib.pyplot as plt
import time as tm
import shutil

# ===========1 读取数据===============
# 获取标注数据地址
path = 'C:/Users/hitzo/Desktop/datami/playground'

file_name_list = os.listdir(path)
file_name_list_new = []  # 图片所在文件夹
for file_name in file_name_list:##_json文件夹
    tmp_file = path + '\\' + file_name
    a = os.path.isdir(tmp_file)
    # b = os.path.isfile(tmp_file)
    if os.path.isdir(tmp_file):#判断是否为目录
        file_name_list_new.append(tmp_file)
N_file = len(file_name_list_new)  # 标注图片数量

# 读取标签数据
# plt.figure()
# plt.ion()
# plt.show()
N_win = 80  # 块大小
N_x = 480 // N_win
N_y = 800 // N_win
N_block = N_x * N_y  # 快的数量
# x_matrix = np.zeros((N_file*N_block,N_win,N_win))
# y_matrix = np.zeros((N_file*N_block,1))
x_list = []  # 非常多的小块
y_list = []  # 对应的标签
for i_file,file_name in enumerate(file_name_list_new):
    t1 = tm.time()

    x_path = file_name + '\\img.jpg'
    y_path = file_name + '\\label.jpg'

    x_im = imread(x_path, as_gray=True)
    y_im = imread(y_path, as_gray=True)

    x_dst = transform.resize(x_im, (480, 800)) # 小图
    y_dst = transform.resize(y_im, (480, 800)) # 小图
    y_dst = y_dst / np.max(y_dst) # 归一化

    target_area = np.sum(y_dst)  # 目标面积大小

    # 划分小块

    for i in range(N_x):
        for j in range(N_y):
            x_tmp = x_dst[i * N_win:(i + 1) * N_win, j * N_win:(j + 1) * N_win]
            fd = hog(x_tmp, orientations=9, pixels_per_cell=[20, 20], cells_per_block=[4, 4],
                     visualize=False, transform_sqrt=True)

            y_tmp = y_dst[i * N_win:(i + 1) * N_win, j * N_win:(j + 1) * N_win]

            y_flag = np.sum(y_tmp) > target_area * 0.4 # 包含目标的小块是正样本
            x_list.append(fd)
            y_list.append(y_flag)

    t2 = tm.time()
    print(f'文件：{i_file}/{N_file} | 时间：{t2-t1}')

    # if i_file > 128:
    #     break
    # a = np.sum(np.array(y_list))

# # 将list格式--->矩阵格式
N_block_all = len(x_list) # 总块数量
N_feature = x_list[0].size
x_list1 = np.zeros((N_block_all, N_feature))
y_list1 = np.zeros((N_block_all, 1))
for i_block in range(N_block_all):
    x_list1[i_block,:] = x_list[i_block]
    y_list1[i_block,:] = y_list[i_block]

np.save('data_x_playground.npy', x_list1)
np.save('data_y_playground.npy', y_list1)

aa = 1

# if __name__ == '__main__':

# extract_features()
