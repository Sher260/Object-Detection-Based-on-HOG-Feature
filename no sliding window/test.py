# from PIL import Image
# from PIL import ImageEnhance
# import os
#
# rootdir = r'D:\programthings\pycharm\mastersecond\object_detector\test_image'  # 指明被遍历的文件夹
# for parent, dirnames, filenames in os.walk(rootdir):
#     for filename in filenames:
#         currentPath = os.path.join(parent, filename)
#         im = Image.open(currentPath)
#         for j in range(72):
#             #im_rotate = im.rotate(j * 10)  # 每张图像都10°旋转一次
#
#             enh_con = ImageEnhance.Contrast(im)  # 增加对比度 得到1440张(20*72=1440)
#             image_contrasted = enh_con.enhance(2)
#             newname1 = r"D:\programthings\pycharm\mastersecond\object_detector\test_image" + filename
#             image_contrasted.save(newname1)
# R = (0.12923,0.09231,0.11077,0.09231,0.09231,0.09231,0.07385,0.11077,0.09231,0.09231,0.09231,0.09231,0.11077,0.12923,0.09231,0.09231,0.12923,0.07385,0.11077,0.09231
#
# )
# t = 0
# for i in range(20):
#     t = R[i] + t
# print(t/20)


# sky
# [0.9691358  0.9691358  0.96296296 0.9691358  0.9691358  0.96296296, 0.9691358  0.9691358  0.9691358  0.96296296 0.9691358  0.9691358, 0.9691358  0.96296296 0.9691358  0.9691358  0.9691358  0.9691358, 0.96296296 0.96296296]
# [0.96319018 0.9691358  0.9689441  0.96319018 0.9691358  0.9689441, 0.9691358  0.96319018 0.9691358  0.9689441  0.9691358  0.9691358, 0.9691358  0.9689441  0.9691358  0.96319018 0.9691358  0.95731707, 0.9689441  0.9689441 ]
# [0.99858974 0.99871795 0.99858974 0.99858974 0.99871795 0.99858974, 0.99871795 0.99858974 0.99871795 0.99858974 0.99871795 0.99871795, 0.99871795 0.99858974 0.99871795 0.99858974 0.99871795 0.99846154, 0.99858974 0.99858974]
# [0.0308642  0.0308642  0.03703704 0.0308642  0.0308642  0.03703704, 0.0308642  0.0308642  0.0308642  0.03703704 0.0308642  0.0308642, 0.0308642  0.03703704 0.0308642  0.0308642  0.0308642  0.0308642, 0.03703704 0.03703704]
# [0.03680982 0.0308642  0.0310559  0.03680982 0.0308642  0.0310559, 0.0308642  0.03680982 0.0308642  0.0310559  0.0308642  0.0308642, 0.0308642  0.0310559  0.0308642  0.03680982 0.0308642  0.04268293, 0.0310559  0.0310559 ]
a = 1.213
print('R=\n',a/20,'\n','.')