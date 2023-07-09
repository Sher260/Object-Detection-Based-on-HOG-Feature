from skimage.feature import hog
from skimage.io import imread
from skimage import transform, data
# from sklearn.externals import joblib
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
path = 'C:/Users/hitzo/Desktop/datami/datamini'

file_name_list = os.listdir(path)
file_name_list_new = []  # 图片所在文件夹
for file_name in file_name_list:
    tmp_file = path + '\\' + file_name
    a = os.path.isdir(tmp_file)
    # b = os.path.isfile(tmp_file)
    if os.path.isdir(tmp_file):
        file_name_list_new.append(tmp_file)
N_file = len(file_name_list_new)  # 标注图片数量

# 读取标签数据
# plt.figure()
# plt.ion()
# plt.show()
N_win = 60  # 块大小  ##---修改
N_x = 480 // N_win *2
N_y = 780 // N_win *2
N_block = (N_x-1) * (N_y-1)  # 快的数量
# x_matrix = np.zeros((N_file*N_block,N_win,N_win))
# y_matrix = np.zeros((N_file*N_block,1))
x_list = []  # 非常多的小块
y_list = []  # 对应的标签
for i_file,file_name in enumerate(file_name_list_new):
    t1 = tm.time()

    x_path = file_name + '\\img.png'
    y_path = file_name + '\\label.png'

    x_im = imread(x_path, as_gray=True)
    y_im = imread(y_path, as_gray=True)

    x_dst = transform.resize(x_im, (480, 780)) # 小图  ##----修改
    y_dst = transform.resize(y_im, (480, 780)) # 小图
    y_dst = y_dst / np.max(y_dst) # 归一化
    plt.imshow(y_dst)
    target_area = np.sum(y_dst)  # 目标面积大小

    # 划分小块
    for i in (range(N_x-1)):
        for j in (range(N_y-1)):
            x_tmp = x_dst[N_win//2 * i : (N_win//2 * i)+ N_win ,  N_win//2 * j : (N_win//2 * j) + N_win ] ##---修改
            fd = hog(x_tmp, orientations=9, pixels_per_cell=[20, 20], cells_per_block=[2, 2],
                     visualize=False, transform_sqrt=True)

            y_tmp = y_dst[N_win//2 * i : (N_win//2 * i)+ N_win ,  N_win//2 * j : (N_win//2 * j) + N_win ]
            y_sum = np.sum(y_tmp)
            y_flag = y_sum > target_area * 0.4 # 包含目标的小块是正样本 ##---修改
            x_list.append(fd)
            y_list.append(y_flag)
            a = 1

    t2 = tm.time()
    print(f'文件：{i_file}/{N_file} | 时间：{t2-t1}')
    a = 1
    # if i_file > 2:
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

np.save('data_x.npy',x_list1)
np.save('data_y.npy',y_list1)

aa = 1

# if __name__ == '__main__':

# extract_features()
