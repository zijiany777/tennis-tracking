import cv2
import numpy as np
import os
import pandas as pd
from sympy import Line
from itertools import combinations
from court_reference import CourtReference

from click_pixelposition import clickpos


# # 读取图像
# img = cv2.imread("../tennis_1_2024-3-2/test/2024.4.10 test/2.png")
# ori_img = img

# image_size = image.shape[:3]  # 返回 (height, width)
# print("Image Size:", image_size)
# print("Min pixel value:", np.min(image))
# print("Max pixel value:", np.max(image))

# 定义全局变量
mask_points = []  # 用于存储选定区域的顶点坐标

def point_cut(img):

    # 鼠标回调函数
    def mouse_callback(event, x, y, flags, param):
        global mask_points, img

        if event == cv2.EVENT_LBUTTONDOWN:
            mask_points.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("image", img)

    # 读取图像
    # img = cv2.imread("greencourt96.png")

    # 获取图像大小
    height, width, _ = img.shape

    # 创建一个窗口并设置大小为图像大小
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", width, height)

    # 创建窗口并设置鼠标回调函数
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_callback)

    while True:
        cv2.imshow("image", img)
        key = cv2.waitKey(1) & 0xFF

        # 按 'esc' 键退出循环
        if key == 27:
            break

    # 创建图像掩码
    mask = np.zeros_like(img[:, :, 0])

    # 将选定区域设为255（白色）
    cv2.fillPoly(mask, [np.int32(mask_points)], 255)

    # 将原图与掩码进行与运算，保留选定区域的像素值
    result = cv2.bitwise_and(img, img, mask=mask)

    # 获取图像大小
    height, width, _ = result.shape

    # 创建一个窗口并设置大小为图像大小
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Result", width, height)


    # 显示结果
    cv2.imshow("Result", result)
    return result


# 设置图像文件夹的路径
image_folder_path = '../court_test/image2'

# 准备一个空列表来收集图像信息
image_info_list = []

# 获取文件夹内所有文件的名称
image_files = [f for f in os.listdir(image_folder_path) if os.path.isfile(os.path.join(image_folder_path, f))]

for image_file in image_files:
    # 构建图像的完整路径
    image_path = os.path.join(image_folder_path, image_file)

    # 读取图像
    img = cv2.imread(image_path)
    ori_img = img

    # 检查是否成功读取图像
    if img is None:
        print("Error: Unable to read the image.")
    else:
        img = point_cut(img)  #是不是因为有全局变量mask_points的缘故导致无法多图片