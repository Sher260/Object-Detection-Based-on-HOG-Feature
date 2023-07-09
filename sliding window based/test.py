from skimage.feature import hog
from skimage.io import imread
from skimage import transform, data
import matplotlib.pyplot as plt

# img=imread('0100325.png',as_gray=True)
# x_dst = transform.resize(img, (480, 780))
# fd = hog(x_dst, orientations=9, pixels_per_cell=[20, 20], cells_per_block=[2, 2],
#                      visualize=True, transform_sqrt=True)
# plt.imshow(fd[1],cmap=plt.cm.gray)
#
# #plt.title = ("未划分子图的HOG特征图像",fontdict = myfontdict)
# #plt.title("未划分子图的HOG特征图像",fontsize='large',fontweight='bold')
# plt.show()

# im=imread('010032.png',as_gray=True)
# f = hog(im, orientations=9, pixels_per_cell=[20, 20], cells_per_block=[2, 2],
#                      visualize=True, transform_sqrt=True)
# plt.imshow(f[1],cmap=plt.cm.gray)
#
# plt.title = ("子图的HOG特征图像")
# plt.show()
x_im = imread('0100325.png', as_gray=True)
x_dst = transform.resize(x_im, (480, 780))
plt.imshow(x_dst)
plt.show()
a=1