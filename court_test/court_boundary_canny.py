import cv2
import numpy as np
from matplotlib import pyplot as plt

dist_tau = 3
intensity_threshold = 10#1-254去任何数一样结果

#Filter pixels by using the court line structure
# def _filter_pixels(gray):
#     """
#     Filter pixels by using the court line structure（取轮廓）
#     """
#     for i in range(dist_tau, len(gray) - dist_tau):
#         for j in range(dist_tau, len(gray[0]) - dist_tau):
#             if gray[i, j] == 0:
#                 continue
#             if (gray[i, j] - gray[i + dist_tau, j] > intensity_threshold and gray[i, j] - gray[
#                 i - dist_tau, j] > intensity_threshold):
#                 continue
#             if (gray[i, j] - gray[i, j + dist_tau] > intensity_threshold and gray[i, j] - gray[
#                 i, j - dist_tau] > intensity_threshold):
#                 continue
#             gray[i, j] = 0
#     return gray

def highlight_boundary(image_path):
    # 读取图像
    original_image = cv2.imread(image_path)

    # 将图像转换为灰度图
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    #gray_image = cv2.threshold(gray_image, 175, 255, cv2.THRESH_BINARY)[1]

    # 使用高斯滤波进行平滑处理
    blurred_image = cv2.GaussianBlur(gray_image, (11, 11), 1)

    # 使用 Canny 边缘检测
    edges = cv2.Canny(blurred_image, 50, 150)

    #使用膨胀
    edges = cv2.dilate(edges, np.ones((10, 10), np.uint8), iterations=1)

    gray = edges

    # """
    #    Filter pixels by using the court line structure（取轮廓）
    #    """
    # for i in range(dist_tau, len(gray) - dist_tau):
    #     for j in range(dist_tau, len(gray[0]) - dist_tau):
    #         if gray[i, j] == 0:
    #             continue
    #         if (gray[i, j] - gray[i + dist_tau, j] > intensity_threshold and gray[i, j] - gray[
    #             i - dist_tau, j] > intensity_threshold):
    #             continue
    #         if (gray[i, j] - gray[i, j + dist_tau] > intensity_threshold and gray[i, j] - gray[
    #             i, j - dist_tau] > intensity_threshold):
    #             continue
    #         gray[i, j] = 0

    # image = _filter_pixels(edges)

    # 反色处理，使边缘变为白色
    #edges = cv2.bitwise_not(edges)

    # 显示原始图像、灰度图和突显边缘的图像
    # plt.figure(figsize=(12, 4))
    #
    # plt.subplot(1, 3, 1)
    # plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    # plt.title('Original Image')
    #
    # plt.subplot(1, 3, 2)
    # plt.imshow(gray_image, cmap='gray')
    # plt.title('Gray Image')
    #
    # plt.subplot(1, 3, 3)
    # plt.imshow(edges, cmap='gray')
    # plt.title('Highlighted Edges')
    cv2.imshow('origin', gray_image)
    cv2.imshow('Highlighted Edges',gray)

    plt.show()


# 替换为您的图像文件路径
image_path = 'greencourt96.png'
highlight_boundary(image_path)
# 等待用户按键，关闭窗口
cv2.waitKey(0)

# 关闭所有打开的窗口
cv2.destroyAllWindows()