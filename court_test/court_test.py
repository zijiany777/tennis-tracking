import cv2
import numpy as np
import os
import pandas as pd
from sympy import Line
from itertools import combinations
from court_reference import CourtReference
import math
import time

start_time = time.time()  # 获取开始时间

#from click_pixelposition import clickpos


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












dist_tau = 3
intensity_threshold = 10#1-254去任何数一样结果

#Filter pixels by using the court line structure
def _filter_pixels(gray):
    """
    Filter pixels by using the court line structure（取轮廓）
    """
    for i in range(dist_tau, len(gray) - dist_tau):
        for j in range(dist_tau, len(gray[0]) - dist_tau):
            if gray[i, j] == 0:
                continue
            if (gray[i, j] - gray[i + dist_tau, j] > intensity_threshold and gray[i, j] - gray[
                i - dist_tau, j] > intensity_threshold):
                continue
            if (gray[i, j] - gray[i, j + dist_tau] > intensity_threshold and gray[i, j] - gray[
                i, j - dist_tau] > intensity_threshold):
                continue
            gray[i, j] = 0
    return gray

def _threshold(gray):
    """
    Simple thresholding for white pixels
    """
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]
    return gray

# def high_pass_filter(image, sigma=1000):
#     # 使用高斯滤波进行平滑
#     blurred = cv2.GaussianBlur(image, (3, 3), sigma)
#
#     # 高频通过滤，原始图像减去平滑后的图像
#     high_pass = image - blurred
#
#     return high_pass



#blurred = cv2.GaussianBlur(image, (5, 5), 10)
#image1 = _threshold(image)

#img1 = point_cut(img)



#clickpos(img1)   //确定两点的距离


#img1 = cv2.equalizeHist(img1)




#image1 =_filter_pixels(image1)

#horizontal, vertical = _detect_lines(image1)
# print("horizontal:",horizontal)
# print("vertical:",vertical)
# lines = _detect_lines(image1)
# print(lines.shape)
# print(lines)


#image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Gray Image", gray)


#image2 = cv2.threshold(image1, 200, 255, cv2.THRESH_BINARY)[1]



#image3 = _filter_pixels(image2)

#检测直线
def _detect_lines(gray):
    """
    Finds all line in frame using Hough transform
    """
    minLineLength = 100 #可以过滤掉较短的线
    maxLineGap = 20
    # Detect all lines
    lines = cv2.HoughLinesP(gray, 1, np.pi / 180, 80, minLineLength=minLineLength, maxLineGap=maxLineGap)
    #gray：输入图像，必须是灰度图。
    # 1：rho 参数，表示霍夫空间中的径向距离分辨率，以像素为单位。这里设置为 1 表示每个像素都会被考虑。
    # np.pi / 180：theta 参数，表示霍夫空间中的角度分辨率，以弧度为单位。这里设置为 np.pi / 180 表示每一度都会被考虑。
    # 80：threshold 参数，表示判断直线的最低票数（即直线上最少的点数）。这里设置为 80，意味着被认为是直线的点的集合至少需要80票。
    # minLineLength：直线的最小长度。这个参数可以帮助过滤掉那些太短的线段，只有长度大于 minLineLength 的线段才会被检测到。这里通过变量 minLineLength 指定。
    # maxLineGap：线段上两点之间的最大允许间隔。如果间隔小于 maxLineGap，这两点之间的空隙将被视为同一条线段的一部分。这里通过变量 maxLineGap 指定。
    lines = np.squeeze(lines)


    # Classify the lines using their slope
    horizontal, vertical = _classify_lines(lines)


    # Merge lines that belong to the same line on frame
    horizontal, vertical = _merge_lines(horizontal, vertical)


    return horizontal, vertical


#找垂线和水平线
def _classify_lines(lines):
    """
    Classify line to vertical and horizontal lines
    """
    horizontal = []
    vertical = []
    highest_vertical_y = np.inf
    lowest_vertical_y = 0
    for line in lines:
        x1, y1, x2, y2 = line
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        if dx > 2 * dy:  #如果 dx 是 dy 的两倍以上，线条被认为是水平的；否则，认为是垂直的。
            horizontal.append(line)
        else:
            vertical.append(line)
            highest_vertical_y = min(highest_vertical_y, y1, y2)
            lowest_vertical_y = max(lowest_vertical_y, y1, y2)

    # Filter horizontal lines using vertical lines lowest and highest point
    #代码计算出基于垂直线的最高点和最低点定义的一个区域。它稍微扩大这个区域（lowest_vertical_y 增加，highest_vertical_y 减少），目的是创建一个包含主要水平线的y坐标范围。
    #然后，它遍历所有水平线，只有当水平线的 y 坐标在这个调整后的区域内时，才将其加入到 clean_horizontal 列表中。
    clean_horizontal = []
    h = lowest_vertical_y - highest_vertical_y
    lowest_vertical_y += h / 15
    highest_vertical_y -= h * 2 / 15
    for line in horizontal:
        x1, y1, x2, y2 = line
        if lowest_vertical_y > y1 > highest_vertical_y and lowest_vertical_y > y1 > highest_vertical_y:
            clean_horizontal.append(line)
    return clean_horizontal, vertical

def line_intersection(line1, line2):
    """
    Find 2 lines intersection point
    """
    l1 = Line(line1[0], line1[1])
    l2 = Line(line2[0], line2[1])

    intersection = l1.intersection(l2)
    return intersection[0].coordinates
def _merge_lines(horizontal_lines, vertical_lines):
    """
    Merge lines that belongs to the same frame`s lines
    """

    # Merge horizontal lines
    horizontal_lines = sorted(horizontal_lines, key=lambda item: item[0])
    mask = [True] * len(horizontal_lines)
    new_horizontal_lines = []
    for i, line in enumerate(horizontal_lines):
        if mask[i]:
            for j, s_line in enumerate(horizontal_lines[i + 1:]):
                if mask[i + j + 1]:
                    x1, y1, x2, y2 = line
                    x3, y3, x4, y4 = s_line
                    dy = abs(y3 - y2)
                    if dy < 20: #可以调整水平线合并的敏感度
                        points = sorted([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], key=lambda x: x[0])
                        line = np.array([*points[0], *points[-1]])
                        mask[i + j + 1] = False
            new_horizontal_lines.append(line)

    # Merge vertical lines
    vertical_lines = sorted(vertical_lines, key=lambda item: item[1])
    v_height, v_width = img.shape[:2]
    # 设置一条水平参考线，用于帮助判断垂直线是否应该合并。这条参考线的位置由图像的高度和宽度决定，这里假设为图像高度的 6/7。
    # 以判断和合并在相似高度上的垂直线。
    xl, yl, xr, yr = (0, v_height * 6 / 7, v_width, v_height * 6 / 7)
    mask = [True] * len(vertical_lines)
    new_vertical_lines = []
    for i, line in enumerate(vertical_lines):
        if mask[i]:
            for j, s_line in enumerate(vertical_lines[i + 1:]):
                if mask[i + j + 1]:
                    x1, y1, x2, y2 = line
                    x3, y3, x4, y4 = s_line
                    xi, yi = line_intersection(((x1, y1), (x2, y2)), ((xl, yl), (xr, yr)))
                    xj, yj = line_intersection(((x3, y3), (x4, y4)), ((xl, yl), (xr, yr)))

                    dx = abs(xi - xj)
                    if dx < 20:  #可以调整垂直线合并的敏感度
                        points = sorted([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], key=lambda x: x[1])
                        line = np.array([*points[0], *points[-1]])
                        mask[i + j + 1] = False

            new_vertical_lines.append(line)
    return new_horizontal_lines, new_vertical_lines

#找两条线的交点


#在图中画线
def draw_lines(image, lines, color=(0, 255, 0), thickness=2):
    """
    Draws lines on an image.
    """
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)



#找透射变换矩阵，并且计分（大框和小框）
def _find_homography(horizontal_lines, vertical_lines,outer,g_or_c):
    """
    Finds transformation from reference court to frame`s court using 4 pairs of matching points
    """
    max_score = -np.inf
    max_mat = None
    max_inv_mat = None
    k = 0
    max_mat_points = []
    # Loop over every pair of horizontal lines and every pair of vertical lines
    for horizontal_pair in list(combinations(horizontal_lines, 2)):
        for vertical_pair in list(combinations(vertical_lines, 2)):
            h1, h2 = horizontal_pair
            v1, v2 = vertical_pair
            # Finding intersection points of all lines
            i1 = line_intersection((tuple(h1[:2]), tuple(h1[2:])), (tuple(v1[0:2]), tuple(v1[2:])))
            i2 = line_intersection((tuple(h1[:2]), tuple(h1[2:])), (tuple(v2[0:2]), tuple(v2[2:])))
            i3 = line_intersection((tuple(h2[:2]), tuple(h2[2:])), (tuple(v1[0:2]), tuple(v1[2:])))
            i4 = line_intersection((tuple(h2[:2]), tuple(h2[2:])), (tuple(v2[0:2]), tuple(v2[2:])))
            # i1_int = (int(round(i1[0])), int(round(i1[1])))
            # i2_int = (int(round(i2[0])), int(round(i2[1])))
            # i3_int = (int(round(i3[0])), int(round(i3[1])))
            # i4_int = (int(round(i4[0])), int(round(i4[1])))
            # cv2.circle(color_image, i1_int, 5, (0, 255, 0), 5)
            # cv2.circle(color_image, i2_int, 5, (0, 255, 0), 5)
            # cv2.circle(color_image, i3_int, 5, (0, 255, 0), 5)
            # cv2.circle(color_image, i4_int, 5, (0, 255, 0), 5)

            intersections = [i1, i2, i3, i4]
            intersections = sort_intersection_points(intersections)
            if(outer == 1):
                configuration = CourtReference().court_conf[1]
            else:
                configuration = CourtReference().court_conf[5]
            # Find transformation
            matrix, _ = cv2.findHomography(np.float32(configuration), np.float32(intersections), method=0)
            inv_matrix = cv2.invert(matrix)[1]
            # Get transformation score
            confi_score = _get_confi_score(matrix,g_or_c)

            if max_score < confi_score:
                max_score = confi_score
                max_mat = matrix
                max_inv_mat = inv_matrix
                best_conf = 1
                max_mat_points = intersections



            k += 1 #计算循环次数

    return max_mat, max_inv_mat, max_score, k, max_mat_points

def _get_confi_score(matrix,g_or_c):
    """
    Calculate transformation score
    """
    court = cv2.warpPerspective(CourtReference().court, matrix, img.shape[1::-1])
    court[court > 0] = 1
    gray = img.copy()  #这里的img是哪里的？filter之后？？
    gray[gray > 0] = 1
    correct = court * gray
    if g_or_c == 1:
        wrong = gray - correct
    else:
        wrong = court - correct #这里应该是gray - correct合理些
    c_p = np.sum(correct)
    w_p = np.sum(wrong)
    return c_p - 0.5 * w_p

#求线的交点
def line_intersection(line1, line2):
    """
    Find 2 lines intersection point
    """
    l1 = Line(line1[0], line1[1])
    l2 = Line(line2[0], line2[1])

    intersection = l1.intersection(l2)
    return intersection[0].coordinates

#对交点进行排序
def sort_intersection_points(intersections):
    """
    sort intersection points from top left to bottom right
    """
    y_sorted = sorted(intersections, key=lambda x: x[1])
    p12 = y_sorted[:2]
    p34 = y_sorted[2:]
    p12 = sorted(p12, key=lambda x: x[0])
    p34 = sorted(p34, key=lambda x: x[0])
    return p12 + p34


#获取投射变换后的外框的四个交点
def homo_outer_court_point(matrix):
    src_points = np.array([[286, 561], [1379, 561], [286, 2935],[1379, 2935]], dtype=np.float32)
    transformed_points = cv2.perspectiveTransform(src_points.reshape(-1, 1, 2), matrix)
    # 输出变换后的点
    print(transformed_points)
    return transformed_points

def court_click_points_distance(court_points,mask_points):
    distances = []
    for i in range(len(court_points)):
        distance = math.sqrt((court_points[i][0][0] - mask_points[i][0])**2 + (court_points[i][0][1] - mask_points[i][1])**2)
        distances.append(distance)
    return distances





if __name__ == '__main__':
    # 设置图像文件夹的路径
    image_folder_path = "../court_test/image3"

    # 准备一个空列表来收集图像信息
    image_info_list = []

    # 获取文件夹内所有文件的名称
    image_files = [f for f in os.listdir(image_folder_path) if os.path.isfile(os.path.join(image_folder_path, f))]

    # 创建新文件夹存储图片
    output_folder = "../court_test/output_images"
    os.makedirs(output_folder, exist_ok=True)

    # 遍历所有图像文件
    for image_file in image_files:
        mask_points = []
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
            img = _threshold(img)
            img = _filter_pixels(img)
            #cv2.imshow("image", img)
            horizontal_lines,vertical_lines = _detect_lines(img)

            # 计算并打印水平线和垂直线的数量
            num_horizontal_lines = len(horizontal_lines) if horizontal_lines is not None else 0
            num_vertical_lines = len(vertical_lines) if vertical_lines is not None else 0

            print(f"Number of horizontal lines: {num_horizontal_lines}")
            print(f"Number of vertical lines: {num_vertical_lines}")

            #将灰度图转换为BGR彩色图像
            color_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            #合并垂线时的基准线
            v_height, v_width = img.shape[:2]


            # 将基准线放入列表中
            base_line = [(0, int(v_height * 6 / 7), v_width, int(v_height * 6 / 7))]
            # 使用 draw_lines 绘制基准线
            draw_lines(color_image, base_line, color=(0, 255, 0), thickness=2)  # 用绿色绘制水平线

            #把直线和垂线画出来
            draw_lines(color_image, horizontal_lines, color=(255, 0, 0), thickness=2)  # 用红色绘制水平线
            draw_lines(color_image, vertical_lines, color=(0, 0, 255), thickness=2)  # 用蓝色绘制垂直线



            #print(image)
            #cv2.imshow("Gray Image", image)

            outer_max_mat, outer_max_inv_mat, outer_max_score, k, outer_max_mat_points = _find_homography(horizontal_lines, vertical_lines,1,1)
            print(f"外框的最高分: {outer_max_score}")

            inner_max_mat, inner_max_inv_mat, inner_max_score, k, inner_max_mat_points = _find_homography(horizontal_lines, vertical_lines,0,1)
            print(f"内框的最高分: {inner_max_score}")

            print(f"计算了多少次循环:{k}")

            print(f"内框的matrix: {inner_max_mat}")

            #标出分值最高的四个点（内框）
            i1_int = (int(round(inner_max_mat_points[0][0])), int(round(inner_max_mat_points[0][1])))
            i2_int = (int(round(inner_max_mat_points[1][0])), int(round(inner_max_mat_points[1][1])))
            i3_int = (int(round(inner_max_mat_points[2][0])), int(round(inner_max_mat_points[2][1])))
            i4_int = (int(round(inner_max_mat_points[3][0])), int(round(inner_max_mat_points[3][1])))
            cv2.circle(color_image, i1_int, 5, (0, 255, 0), 5)
            cv2.circle(color_image, i2_int, 5, (0, 255, 0), 5)
            cv2.circle(color_image, i3_int, 5, (0, 255, 0), 5)
            cv2.circle(color_image, i4_int, 5, (0, 255, 0), 5)

            #通过投射变化，把直线投到原图中
            matrix = inner_max_mat
            p = np.array(CourtReference().get_important_lines(), dtype=np.float32)
            p = p.reshape(-1, 1, 2)
            lines = cv2.perspectiveTransform(p, matrix).reshape(-1)
            for i in range(0, len(lines), 4):
                x1, y1, x2, y2 = lines[i], lines[i + 1], lines[i + 2], lines[i + 3]
                pt1 = (int(round(x1)), int(round(y1)))
                pt2 = (int(round(x2)), int(round(y2)))
                cv2.line(ori_img, pt1, pt2, (255, 0, 0), 3)


            #存储直线和交点的照片
            image_file_without_extension = image_file.split(".")[0]
            new_image_path = os.path.join(output_folder, image_file_without_extension+"_line_image.jpg")
            cv2.imwrite(new_image_path, color_image)

            #存储投射变换后的照片
            image_file_without_extension = image_file.split(".")[0]
            new_image_path = os.path.join(output_folder, image_file_without_extension+"_homo_image.jpg")
            cv2.imwrite(new_image_path, ori_img)


            #得到投射变换后的外框四个顶点
            court_points = homo_outer_court_point(inner_max_mat)
            #计算投射变换后的外框四个顶点和点击的四个点的距离
            distances = court_click_points_distance(court_points,mask_points)



            # #显示透射变换后的court和原图的二值图相交
            # court = cv2.warpPerspective(CourtReference().court, inner_max_mat, img.shape[1::-1])
            # height, width = img.shape
            # court[court > 0] = 1
            # gray = img.copy()
            # gray[gray > 0] = 1
            # correct = court * gray
            # correct[correct > 0] = 255
            # court[court > 0] = 255
            # gray[gray > 0] = 255


            # # 获取图像大小
            # height, width= img.shape
            #
            # # 创建一个窗口并设置大小为图像大小
            # cv2.namedWindow("image1", cv2.WINDOW_NORMAL)
            # cv2.namedWindow("image2", cv2.WINDOW_NORMAL)
            #
            # cv2.resizeWindow("image1", width, height)
            # cv2.resizeWindow("image2", width, height)
            #
            # cv2.imshow("image1", color_image)
            # cv2.imshow("image2", ori_img)


            # cv2.namedWindow("correct", cv2.WINDOW_NORMAL)
            # cv2.resizeWindow("correct", width, height)
            # cv2.imshow("correct", correct)
            #
            # cv2.namedWindow("court", cv2.WINDOW_NORMAL)
            # cv2.resizeWindow("court", width, height)
            # cv2.imshow("court", court)
            #
            # cv2.namedWindow("gray", cv2.WINDOW_NORMAL)
            # cv2.resizeWindow("gray", width, height)
            # cv2.imshow("gray", gray)
            # #cv2.imshow("Gray Image2", image2)



            # # 等待用户按键，关闭窗口
            # cv2.waitKey(0)
            #
            # # 关闭所有打开的窗口
            # cv2.destroyAllWindows()
            # 确保图像被正确读取
            #height, width, channels = img.shape
            size = os.path.getsize(image_path)

            # 添加信息到列表
            image_info_list.append({
                "Image": image_file,
                #"Resolution": f"{width}x{height}",
                #"Channels": channels,
                "Size (bytes)": size,
                "max_confi_inner_court_points": inner_max_score,
                "max_confi_outer_court_points": outer_max_score,
                "court_points": court_points,
                "counting_times": k,
                #"click_points": mask_points,#是不是又是因为全局变量
                "court_click_points_distance":distances
            })

    # 创建数据框
    df_images = pd.DataFrame(image_info_list)

    # 将数据框写入Excel文件
    excel_path = 'image_info.xlsx'
    df_images.to_excel(excel_path, index=False, engine='openpyxl')

    print(f"Image information has been saved to {excel_path}")



end_time = time.time()  # 获取结束时间
elapsed_time = end_time - start_time  # 计算运行时间
print(f"The code ran for {elapsed_time} seconds.")