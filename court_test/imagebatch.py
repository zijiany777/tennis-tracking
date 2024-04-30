import cv2
import os
import pandas as pd
import numpy as np
from court_test import mask_points,point_cut,_threshold,_filter_pixels,_detect_lines,draw_lines,_find_homography,homo_outer_court_point
from court_reference import CourtReference

# 设置图像文件夹的路径
image_folder_path = '../court_test/image2'

# 准备一个空列表来收集图像信息
image_info_list = []

# 获取文件夹内所有文件的名称
image_files = [f for f in os.listdir(image_folder_path) if os.path.isfile(os.path.join(image_folder_path, f))]

# 遍历所有图像文件
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
        # 定义全局变量
        mask_points = []  # 用于存储选定区域的顶点坐标
        img = point_cut(img)
#         img = _threshold(img)
#         img = _filter_pixels(img)
#         # cv2.imshow("image", img)
#         horizontal_lines, vertical_lines = _detect_lines(img)
#
#         # 计算并打印水平线和垂直线的数量
#         num_horizontal_lines = len(horizontal_lines) if horizontal_lines is not None else 0
#         num_vertical_lines = len(vertical_lines) if vertical_lines is not None else 0
#
#         print(f"Number of horizontal lines: {num_horizontal_lines}")
#         print(f"Number of vertical lines: {num_vertical_lines}")
#
#         # 将灰度图转换为BGR彩色图像
#         color_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#
#         # 合并垂线时的基准线
#         v_height, v_width = img.shape[:2]
#
#         # 将基准线放入列表中
#         base_line = [(0, int(v_height * 6 / 7), v_width, int(v_height * 6 / 7))]
#         # 使用 draw_lines 绘制基准线
#         draw_lines(color_image, base_line, color=(0, 255, 0), thickness=2)  # 用绿色绘制水平线
#
#         draw_lines(color_image, horizontal_lines, color=(255, 0, 0), thickness=2)  # 用红色绘制水平线
#         draw_lines(color_image, vertical_lines, color=(0, 0, 255), thickness=2)  # 用蓝色绘制垂直线
#
#         # print(image)
#         # cv2.imshow("Gray Image", image)
#
#         outer_max_mat, outer_max_inv_mat, outer_max_score, k = _find_homography(horizontal_lines, vertical_lines, 1)
#         print(f"外框的最高分: {outer_max_score}")
#
#         inner_max_mat, inner_max_inv_mat, inner_max_score, k = _find_homography(horizontal_lines, vertical_lines, 0)
#         print(f"内框的最高分: {inner_max_score}")
#
#         print(f"计算了多少次循环:{k}")
#
#         # 通过投射变化，把直线投到原图中
#         p = np.array(CourtReference().get_important_lines(), dtype=np.float32).reshape((-1, 1, 2))
#         lines = cv2.perspectiveTransform(p, inner_max_mat).reshape(-1)
#         for i in range(0, len(lines), 4):
#             x1, y1, x2, y2 = lines[i], lines[i + 1], lines[i + 2], lines[i + 3]
#             pt1 = (int(round(x1)), int(round(y1)))
#             pt2 = (int(round(x2)), int(round(y2)))
#             cv2.line(ori_img, pt1, pt2, (255, 0, 0), 3)
#
#         # 得到投射变换后的外框四个顶点
#         court_points = homo_outer_court_point(inner_max_mat)
#
#         # #显示透射变换后的court和原图的二值图相交
#         # court = cv2.warpPerspective(CourtReference().court, inner_max_mat, img.shape[1::-1])
#         # height, width = img.shape
#         # court[court > 0] = 1
#         # gray = img.copy()
#         # gray[gray > 0] = 1
#         # correct = court * gray
#         # correct[correct > 0] = 255
#         # court[court > 0] = 255
#         # gray[gray > 0] = 255
#
#         # 获取图像大小
#         height, width = ori_img.shape
#
#         # 创建一个窗口并设置大小为图像大小
#         # cv2.namedWindow("image1", cv2.WINDOW_NORMAL)
#         # cv2.namedWindow("image2", cv2.WINDOW_NORMAL)
#         #
#         # cv2.resizeWindow("image1", width, height)
#         # cv2.resizeWindow("image2", width, height)
#         #
#         # cv2.imshow("image1", color_image)
#         # cv2.imshow("image2", ori_img)
#
#         # cv2.namedWindow("correct", cv2.WINDOW_NORMAL)
#         # cv2.resizeWindow("correct", width, height)
#         # cv2.imshow("correct", correct)
#         #
#         # cv2.namedWindow("court", cv2.WINDOW_NORMAL)
#         # cv2.resizeWindow("court", width, height)
#         # cv2.imshow("court", court)
#         #
#         # cv2.namedWindow("gray", cv2.WINDOW_NORMAL)
#         # cv2.resizeWindow("gray", width, height)
#         # cv2.imshow("gray", gray)
#         # #cv2.imshow("Gray Image2", image2)
#         #
#         # # 等待用户按键，关闭窗口
#         # cv2.waitKey(0)
#         #
#         # # 关闭所有打开的窗口
#         # cv2.destroyAllWindows()
#
#     # 确保图像被正确读取
#         height, width, channels = img.shape
#         size = os.path.getsize(image_path)
#
#         # 添加信息到列表
#         image_info_list.append({
#             "Image": image_file,
#             "Resolution": f"{width}x{height}",
#             "Channels": channels,
#             "Size (bytes)": size,
#             "court_points":court_points,
#             "click_points":mask_points
#         })
#
# # 创建数据框
# df_images = pd.DataFrame(image_info_list)
#
# # 将数据框写入Excel文件
# excel_path = 'image_info.xlsx'
# df_images.to_excel(excel_path, index=False, engine='openpyxl')
#
# print(f"Image information has been saved to {excel_path}")
